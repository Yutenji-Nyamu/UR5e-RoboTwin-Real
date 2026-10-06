#!/usr/bin/env python
"""Zero-WAM inference server with an attention probe (drop-in for ``python -m wan_va.wan_va_server``).

Patches ``ICLAttentionBackend`` so that, for every self-attention call, the attention of the *video* queries
(the only queries allowed to read the in-context human demo, ``query_type == 0``) and of the *action* queries
(``query_type == 1``) is recomputed explicitly with the same mask, and summarised as mass on key groups:

  icl_frame[f]   ICL demo tokens of demo latent frame f            (key_type == 2, grouped by key frame id)
  obs            observation / robot-video keys outside the current chunk (key_type == 0, older frames)
  self           keys of the current chunk (same frame ids as the queries)
  action         action keys (key_type == 1)

Per forward call we also store the query frame ids (robot progress) so the demo-frame attention can be laid out
as an alignment matrix (robot progress x demo time).  Records are flushed to ``$ZW_PROBE_OUT/<episode>.npz`` on
every episode reset and at exit.  Only the server is patched; the client and protocol are unchanged.
"""
from __future__ import annotations

import atexit
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ZW = Path("/m2v_intern_v3/danglingwei/codes/Zero-WAM_260923")
sys.path.insert(0, str(ZW))
OUT = Path(os.environ.get("ZW_PROBE_OUT", "/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/probes/zerowam/attn"))
OUT.mkdir(parents=True, exist_ok=True)

from wan_va.modules import icl_model as M  # noqa: E402


class Probe:
    def __init__(self):
        self.meta = None
        self.layer = -1
        self.records: list[dict] = []
        self.episode = 0
        self.forward_id = 0

    def flush(self):
        if not self.records:
            return
        path = OUT / f"episode_{self.episode:04d}_{int(time.time())}.npz"
        keys = sorted({k for r in self.records for k in r})
        np.savez_compressed(path, **{k: np.asarray([r.get(k, np.nan) for r in self.records], dtype=object) for k in keys})
        print(f"[probe] wrote {len(self.records)} attention records -> {path}", flush=True)
        self.records = []
        self.episode += 1


P = Probe()
_orig_build = M.ICLAttentionBackend.build_self_mask.__func__
_orig_apply = M.ICLAttentionBackend.apply.__func__


def build_self_mask(cls, query_type_ids, key_type_ids, query_seq_ids, key_seq_ids, query_frame_ids, key_frame_ids, window_size, device, compile_mask=True):
    P.meta = dict(qt=query_type_ids, kt=key_type_ids, qs=query_seq_ids, ks=key_seq_ids, qf=query_frame_ids, kf=key_frame_ids, w=int(window_size))
    P.layer = -1
    P.forward_id += 1
    return _orig_build(cls, query_type_ids, key_type_ids, query_seq_ids, key_seq_ids, query_frame_ids, key_frame_ids, window_size, device, compile_mask)


def apply(cls, query, key, value, *, cross_attention: bool, block_mask=None):
    out = _orig_apply(cls, query, key, value, cross_attention=cross_attention, block_mask=block_mask)
    if cross_attention or P.meta is None:
        return out
    m = P.meta
    if query.shape[1] != len(m["qt"]) or key.shape[1] != len(m["kt"]):
        return out
    P.layer += 1
    try:
        with torch.no_grad():
            qt, kt, qs, ks, qf, kf, w = m["qt"], m["kt"], m["qs"], m["ks"], m["qf"], m["kf"], m["w"]
            allowed = (kt[None, :] != -1) & (qs[:, None] == ks[None, :])
            win = (w == -1) | ((qf[:, None] - kf[None, :]).abs() <= w)
            allowed = allowed & ((win & (kt[None, :] != M.ICL_CACHE_TYPE)) | ((qt[:, None] == 0) & (kt[None, :] == M.ICL_CACHE_TYPE)))
            rec = dict(forward=P.forward_id, layer=P.layer, q_frame_min=int(qf[qt != -1].min()) if (qt != -1).any() else -1,
                       q_frame_max=int(qf[qt != -1].max()) if (qt != -1).any() else -1, n_icl_keys=int((kt == M.ICL_CACHE_TYPE).sum()),
                       n_obs_keys=int((kt == 0).sum()), n_action_keys=int((kt == 1).sum()))
            d = query.shape[-1]
            for qtype, name in ((0, "video"), (1, "action")):
                rows = torch.nonzero(qt == qtype).squeeze(-1)
                if rows.numel() == 0:
                    continue
                q = query[0, rows].transpose(0, 1).float()                                 # [H, Lq, D]
                k = key[0].transpose(0, 1).float()                                          # [H, Lk, D]
                logits = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)                   # [H, Lq, Lk]
                logits = logits.masked_fill(~allowed[rows][None], float("-inf"))
                probs = logits.softmax(-1)
                mass = probs.sum(dim=(0, 1))                                                 # [Lk] summed over heads & queries
                total = float(mass.sum())
                is_icl = kt == M.ICL_CACHE_TYPE
                cur = (kt == 0) & torch.isin(kf, qf[rows].unique())
                rec[f"{name}_mass_icl"] = float(mass[is_icl].sum() / total)
                rec[f"{name}_mass_obs"] = float(mass[(kt == 0) & ~cur].sum() / total)
                rec[f"{name}_mass_self"] = float(mass[cur].sum() / total)
                rec[f"{name}_mass_action"] = float(mass[kt == 1].sum() / total)
                rec[f"{name}_n_queries"] = int(rows.numel())
                if is_icl.any():
                    frames = kf[is_icl]
                    uniq = torch.unique(frames)
                    per_frame = torch.stack([mass[is_icl][frames == f].sum() for f in uniq]) / total
                    rec[f"{name}_icl_frames"] = uniq.cpu().numpy()
                    rec[f"{name}_icl_frame_mass"] = per_frame.cpu().numpy()
                    # per-head entropy over ICL keys (how concentrated on demo tokens each head is)
                    p_icl = probs[:, :, is_icl].sum(1)                                        # [H, n_icl]
                    p_icl = p_icl / p_icl.sum(-1, keepdim=True).clamp_min(1e-12)
                    ent = -(p_icl * (p_icl + 1e-12).log()).sum(-1) / np.log(p_icl.shape[-1])
                    rec[f"{name}_icl_head_entropy"] = ent.cpu().numpy()
                    rec[f"{name}_icl_head_mass"] = (probs[:, :, is_icl].sum((1, 2)) / probs.sum((1, 2))).cpu().numpy()
            P.records.append(rec)
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] skipped layer {P.layer}: {exc}", flush=True)
    return out


M.ICLAttentionBackend.build_self_mask = classmethod(build_self_mask)
M.ICLAttentionBackend.apply = classmethod(apply)

# episode boundary: flush when the server resets the ICL context
from wan_va import wan_va_server as S  # noqa: E402

_orig_reset_icl = S.WanVAServer._reset_icl if hasattr(S, "WanVAServer") else None
for _name in dir(S):
    _cls = getattr(S, _name)
    if isinstance(_cls, type) and hasattr(_cls, "_reset_icl"):
        _orig = _cls._reset_icl

        def _reset_icl(self, *a, _orig=_orig, **kw):
            P.flush()
            return _orig(self, *a, **kw)

        _cls._reset_icl = _reset_icl
        print(f"[probe] patched {_name}._reset_icl", flush=True)
atexit.register(P.flush)

if __name__ == "__main__":
    S.main()
