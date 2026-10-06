"""RynnBrain numerics check in the InternW0 environment: the same image + instruction through the backbone with the
causal-conv1d fast path and with its PyTorch fallback; prints the relative difference of ``last_hidden_state``.

    CUDA_VISIBLE_DEVICES=1 /ytech_milm_intern/danglingwei/envs/internw0-delta/bin/python \
        metiswam4d_inspired_by_internw0/scripts/check_rynnbrain_kernels.py
"""
import io

import h5py
import numpy as np
import torch
from PIL import Image

VLM = "/ytech_milm_intern/danglingwei/model_zoos/InternW0-Delta/RynnBrain1.1-2B"
EPISODE = "/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/datas/RDJ_MetisWAM4D/cover_blocks/train_4d/episode0/source.hdf5"


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(VLM, trust_remote_code=True, max_pixels=65536)
    model = AutoModelForImageTextToText.from_pretrained(VLM, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    with h5py.File(EPISODE) as f:
        img = Image.fromarray(np.asarray(Image.open(io.BytesIO(bytes(f["observation/head_camera/rgb"][100]))).convert("RGB"))[..., ::-1])
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Pick up the red block by 10 cm."}]}]
    text = proc.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    inputs = {k: v.cuda() for k, v in proc(text=[text], images=[img], return_tensors="pt").items()}
    fast = [m for m in model.modules() if getattr(m, "causal_conv1d_fn", None) is not None]
    print("modules with causal_conv1d fast path:", len(fast))
    with torch.no_grad():
        a = model.model(**inputs).last_hidden_state.float()
        saved = [m.causal_conv1d_fn for m in fast]
        for m in fast:
            m.causal_conv1d_fn = None
        b = model.model(**inputs).last_hidden_state.float()
        for m, fn in zip(fast, saved):
            m.causal_conv1d_fn = fn
        c = model.model(**inputs).last_hidden_state.float()
    rel = lambda x, y: float((x - y).norm() / y.norm())
    print(f"tokens {a.shape[1]}  |h| {a.norm(dim=-1).mean():.2f}")
    print(f"fast vs fallback relative diff {rel(a, b):.4e}   fast vs fast (repeat) {rel(a, c):.4e}")
    print(f"per-token cosine fast/fallback min {torch.nn.functional.cosine_similarity(a[0], b[0], dim=-1).min():.4f}")


if __name__ == "__main__":
    main()
