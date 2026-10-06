"""python -m ur5e_real.adapters.metiswam4d: prepare, smoke, train, offline infer."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .checkpoint import frozen_digest, load, save
from .contract import FPS, HORIZON, VERSION, fit_stats, to_pi05
from .data import prepare, read_rgb
from .native import build_model, load_source, model_config
from .policy import ActionPolicy, WanRgbEncoder


def _diagnostic_latent(head, wrist, device):
    """Pixel pooling ONLY for smoke tests; never silently substitutes for a Wan VAE."""
    import cv2

    image = np.concatenate([cv2.resize(view, (8, 4)) for view in (head, wrist)], axis=0)
    rgb = torch.as_tensor(image.copy(), device=device, dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1
    return torch.cat((rgb, rgb.mean(dim=0, keepdim=True)), dim=0)[None, :, None]


def smoke(args, source):
    torch.manual_seed(7)
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = model_config(tiny=True)
    model = build_model(cfg).to(device=args.device, dtype=getattr(torch, args.dtype))
    model.config.gradient_checkpointing = True
    state = np.zeros((1, 7), dtype=np.float32)
    action = np.zeros((1, HORIZON, 7), dtype=np.float32)
    action[0, :, 0] = np.linspace(0, 0.1, HORIZON)
    action[0, :, 6] = np.repeat([1, 0, 1, 1, 1], 10)
    valid = np.ones((1, HORIZON), dtype=bool)
    if args.dataset:
        data = np.load(args.dataset / "actions.npz", allow_pickle=False)
        state, action, valid = data["state"][:1], data["actions"][:1], data["valid"][:1]
        info = json.loads((args.dataset / "images.json").read_text())[0]
        head, wrist = read_rgb(info["head"]), read_rgb(info["wrist"])
    else:
        rng = np.random.default_rng(7)
        head, wrist = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(2)]
    latent = _diagnostic_latent(head, wrist, args.device)
    stats = fit_stats(state, action, valid)
    policy = ActionPolicy(model, stats)
    before = frozen_digest(model)
    initial_action = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    losses, start = [], time.monotonic()
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        loss = policy.loss(latent, state, action, valid)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite smoke loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    changed = [n for n, p in model.named_parameters() if p.requires_grad and not torch.equal(p, initial_action[n])]
    assert any(n.startswith("action.") for n in changed)
    assert any(n.startswith("proprio_encoder.") for n in changed)
    assert before == frozen_digest(model)
    prediction = policy.predict(latent, state, rounds=4)
    changed_image = policy.predict(-latent, state, rounds=4)
    sensitivity = float(np.max(np.abs(prediction - changed_image)))
    assert sensitivity > 0, "Action lost its image condition"
    contract = {"version": VERSION, "fps": FPS, "horizon": HORIZON, "slots": list(range(10, 17)), "stats": stats}
    save(args.output / "diagnostic.pt", model, cfg, contract, source)
    restored, _ = load(args.output / "diagnostic.pt", source, device=args.device, allow_tiny=True)
    np.testing.assert_array_equal(prediction, restored.predict(latent, state, rounds=4))
    report = {"status": "passed", "scope": "native tiny Video + Action; diagnostic pixel pooling, NOT a pretrained VAE",
              "uses_real_episode": bool(args.dataset), "device": args.device, "dtype": args.dtype,
              "gradient_checkpointing": True, "torch": str(torch.__version__),
              "source_sha256": source, "steps": 8, "losses": losses,
              "frozen_before_sha256": before, "frozen_after_sha256": frozen_digest(model),
              "updated_parameters": len(changed), "trainable_groups": sorted(model.trainable_groups()),
              "image_perturbation_max_action_change": sensitivity, "save_reload_exact": True,
              "physical_action_shape": list(prediction.shape), "pi05_action_shape": list(to_pi05(prediction).shape),
              "elapsed_s": time.monotonic() - start, "robot_connected": False}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def train(args, source):
    from metiswam4d.experts.alpha import find_checkpoint, load_alpha_action, load_alpha_proprio, load_alpha_video

    if args.steps < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("positive steps and learning rate required")
    checkpoint = find_checkpoint(args.alpha_checkpoint)  # fail before allocating the large model
    if not (args.vae / "config.json").is_file():
        raise FileNotFoundError("local Wan VAE config missing")
    contract = json.loads((args.dataset / "contract.json").read_text())
    if contract["version"] != VERSION:
        raise ValueError("dataset contract mismatch")
    with np.load(args.dataset / "actions.npz", allow_pickle=False) as data:
        state, actions, valid = data["state"], data["actions"], data["valid"]
    images = json.loads((args.dataset / "images.json").read_text())
    if not len(state) or len(state) != len(images):
        raise ValueError("image/action sample count mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(7)
    cfg = model_config()
    model = build_model(cfg).to(device=args.device, dtype=torch.bfloat16)
    init = {"video": load_alpha_video(model.video, checkpoint), "action": load_alpha_action(model.action, checkpoint),
            "proprio": load_alpha_proprio(model.proprio_encoder, checkpoint)}
    encoder = WanRgbEncoder(args.vae, device=args.device)
    policy = ActionPolicy(model, contract["stats"])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    frozen = frozen_digest(model)
    cache, losses, rng = {}, [], np.random.default_rng(7)
    model.train()
    for step in range(args.steps):
        idx = int(rng.integers(len(state)))
        pair = images[idx]
        key = (pair["head"], pair["wrist"])
        if key not in cache:
            cache[key] = encoder.encode(read_rgb(key[0]), read_rgb(key[1])).cpu()
        optimizer.zero_grad(set_to_none=True)
        loss = policy.loss(cache[key], state[idx:idx+1], actions[idx:idx+1], valid[idx:idx+1])
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 10 == 0:
            print(json.dumps({"step": step + 1, "loss": losses[-1]}), flush=True)
    if frozen != frozen_digest(model):
        raise RuntimeError("frozen parameters changed")
    save(args.output / "policy.pt", model, cfg, contract, source, vae_path=args.vae.resolve())
    report = {"status": "trained", "steps": args.steps, "losses": losses, "initialization": init,
              "trainable_groups": sorted(model.trainable_groups()), "frozen_unchanged": True,
              "task": contract["task"], "source_sha256": source, "robot_connected": False}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def infer(args, source):
    policy, saved = load(args.checkpoint, source, device=args.device)
    encoder = WanRgbEncoder(saved["vae_path"], device=args.device)
    latent = encoder.encode(read_rgb(args.head), read_rgb(args.wrist))
    prediction = policy.predict(latent, np.asarray([args.state], dtype=np.float32), rounds=args.rounds)[0]
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, joint_gripper=prediction, pi05_joint14=to_pi05(prediction), dt_s=1 / FPS)
    return {"status": "offline_prediction", "shape": list(prediction.shape), "task": saved["contract"]["task"],
            "robot_connected": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[4] / ".third_party/MetisWAM4D")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("prepare", help="full-task measured-joint training windows; no VAE needed")
    data.add_argument("--data-root", required=True, type=Path)
    data.add_argument("--runs", nargs="+", required=True)
    data.add_argument("--task", required=True)
    data.add_argument("--output", required=True, type=Path)
    test = commands.add_parser("smoke", help="native tiny structural test; never a trained robot policy")
    test.add_argument("--dataset", type=Path)
    test.add_argument("--device", default="cpu")
    test.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    test.add_argument("--output", required=True, type=Path)
    training = commands.add_parser("train", help="single-task Action/proprio fine-tuning from explicit local Alpha weights")
    training.add_argument("--dataset", required=True, type=Path)
    training.add_argument("--alpha-checkpoint", required=True, type=Path)
    training.add_argument("--vae", required=True, type=Path)
    training.add_argument("--steps", default=1000, type=int)
    training.add_argument("--lr", default=1e-5, type=float)
    training.add_argument("--device", default="cuda")
    training.add_argument("--output", required=True, type=Path)
    prediction = commands.add_parser("infer", help="offline RGB + measured joint/gripper -> 50 actions")
    prediction.add_argument("--checkpoint", required=True, type=Path)
    prediction.add_argument("--head", required=True, type=Path)
    prediction.add_argument("--wrist", required=True, type=Path)
    prediction.add_argument("--state", required=True, nargs=7, type=float, metavar="VALUE")
    prediction.add_argument("--rounds", default=16, type=int)
    prediction.add_argument("--device", default="cuda")
    prediction.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.data_root, args.runs, args.task, args.output)
    else:
        source = load_source(args.source)
        result = {"smoke": smoke, "train": train, "infer": infer}[args.command](args, source)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
