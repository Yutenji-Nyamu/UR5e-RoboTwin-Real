# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (see integrations/rlt/LICENSE-RLinf).
# Losses adapted from fsdp_rlt_ac_policy_worker.py at the locked donor commit.
"""Single-device RLT losses; physical rollout is never performed by the learner."""

from __future__ import annotations

from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F

from .models import heads


def masked_bc(actions, reference, lengths):
    mask = torch.arange(actions.shape[1], device=actions.device)[None] < lengths[:, None]
    errors = (actions - reference).square().mean(-1)
    return (errors * mask).sum() / mask.sum().clamp_min(1)


def td_target(rewards, lengths, terminated, truncated, next_q, gamma, bootstrap_on_timeout=True):
    indices = torch.arange(rewards.shape[1], device=rewards.device)
    valid = indices[None] < lengths[:, None]
    discounted = (rewards * valid * gamma**indices).sum(-1)
    done = terminated.bool() | (truncated.bool() & (not bootstrap_on_timeout))
    return discounted + (~done) * (gamma**lengths) * next_q


class RLTTrainer:
    def __init__(self, config, device="cpu"):
        self.config, self.device = config, torch.device(device)
        actor, critic = heads(config)
        self.actor, self.critic = actor.to(self.device), critic.to(self.device)
        self.target = deepcopy(self.critic).requires_grad_(False)
        lr = config["algorithm"]["learning_rate"]
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_updates = self.actor_updates = self.bc_updates = 0
        self.online_start_update = None

    def weights(self, phase):
        cfg = self.config["algorithm"]
        if phase == "warmup":
            progress = 0.0
        elif phase == "online":
            if self.online_start_update is None:
                self.online_start_update = self.critic_updates
            progress = min(1.0, (self.critic_updates - self.online_start_update + 1) / max(1, cfg["ramp_updates"]))
        else:
            raise ValueError("learner phase must be warmup or online")
        return tuple(
            cfg[f"warmup_{name}_weight"] + progress * (cfg[f"online_{name}_weight"] - cfg[f"warmup_{name}_weight"])
            for name in ("bc", "q")
        )

    def _batch(self, batch):
        return {key: torch.as_tensor(value, device=self.device) for key, value in batch.items()}

    def bc_step(self, batch):
        batch = self._batch(batch)
        self.actor_optimizer.zero_grad(set_to_none=True)
        prediction = self.actor(batch["z"], batch["proprio"], batch["reference"], deterministic=True)
        loss = F.mse_loss(prediction, batch["reference"])
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.config["algorithm"]["gradient_clip"])
        self._finite(loss, norm)
        self.actor_optimizer.step()
        self.bc_updates += 1
        return {"bc_updates": self.bc_updates, "bc_loss": loss.item(), "actor_grad_norm": float(norm)}

    @staticmethod
    def _finite(*values):
        if not all(torch.isfinite(value).all() for value in values):
            raise FloatingPointError(
                "non-finite RLT loss or gradient; refusing this optimizer step; resume the last paired checkpoint"
            )

    def update(self, data, *, phase="warmup"):
        batch = self._batch(data)
        cfg = self.config["algorithm"]
        with torch.no_grad():
            following = self.actor(batch["next_z"], batch["next_proprio"], batch["next_reference"])
            following_q = self.target(batch["next_z"], batch["next_proprio"], following).min(-1).values
            targets = td_target(
                batch["rewards"],
                batch["lengths"],
                batch["terminated"],
                batch["truncated"],
                following_q,
                cfg["gamma"],
                cfg["bootstrap_on_timeout"],
            )
        self.critic_optimizer.zero_grad(set_to_none=True)
        q = self.critic(batch["z"], batch["proprio"], batch["action"])
        q_loss = F.mse_loss(q, targets[:, None].expand_as(q))
        q_loss.backward()
        q_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), cfg["gradient_clip"])
        self._finite(q_loss, q_norm)
        self.critic_optimizer.step()
        metrics = {
            "critic_loss": q_loss.item(),
            "td_abs": (q.detach() - targets[:, None]).abs().mean().item(),
            "q_mean": q.detach().mean().item(),
            "q_min": q.detach().min().item(),
            "q_max": q.detach().max().item(),
            "target_q_mean": targets.mean().item(),
            "twin_q_gap": (q[:, 0] - q[:, 1]).detach().abs().mean().item(),
            "critic_grad_norm": float(q_norm),
        }
        update_actor = self.critic_updates % cfg["critic_actor_ratio"] == 0
        if update_actor:
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic.requires_grad_(False)
            try:
                action = self.actor(
                    batch["z"], batch["proprio"], batch["reference"], reference_dropout=cfg["reference_dropout"]
                )
                policy_q = self.critic(batch["z"], batch["proprio"], action)[:, 0]
                bc = masked_bc(action, batch["reference"], batch["lengths"])
                bc_weight, q_weight = self.weights(phase)
                loss = bc_weight * bc - q_weight * policy_q.mean()
                loss.backward()
                norm = nn.utils.clip_grad_norm_(self.actor.parameters(), cfg["gradient_clip"])
                self._finite(loss, norm)
                self.actor_optimizer.step()
                self.actor_updates += 1
                metrics.update(
                    actor_loss=loss.item(),
                    bc_loss=bc.item(),
                    q_actor=policy_q.mean().item(),
                    bc_weight=bc_weight,
                    q_weight=q_weight,
                    weighted_bc=bc_weight * bc.item(),
                    weighted_q=q_weight * policy_q.mean().item(),
                    actor_grad_norm=float(norm),
                    action_ref_abs=(action - batch["reference"]).detach().abs().mean().item(),
                )
            finally:
                self.critic.requires_grad_(True)
        with torch.no_grad():
            for target, current in zip(self.target.parameters(), self.critic.parameters(), strict=True):
                target.lerp_(current, cfg["tau"])
        self.critic_updates += 1
        metrics.update(
            critic_updates=self.critic_updates,
            actor_updates=self.actor_updates,
            actor_updated=update_actor,
            phase=phase,
        )
        return metrics

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target": self.target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "critic_updates": self.critic_updates,
            "actor_updates": self.actor_updates,
            "bc_updates": self.bc_updates,
            "online_start_update": self.online_start_update,
        }

    def load_state_dict(self, value):
        for name in ("actor", "critic", "target", "actor_optimizer", "critic_optimizer"):
            getattr(self, name).load_state_dict(value[name])
        for name in ("critic_updates", "actor_updates", "bc_updates", "online_start_update"):
            setattr(self, name, value[name])
