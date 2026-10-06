"""Online UMT5 prompt encoding (the Wan2.2 text encoder), GPU-resident next to the VAE.

Matches the diffusers Wan pipeline: tokenise with padding to ``max_len``, run the encoder, keep the
first ``len`` hidden states of every prompt.  Returns ``text_context [B, L, 4096]`` (bf16) and
``text_mask [B, L]`` with ``L`` = longest prompt of the batch.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor


class UMT5Online:
    def __init__(self, model_dir: str, device: torch.device, max_len: int = 256, dtype: torch.dtype = torch.bfloat16):
        from transformers import AutoTokenizer, UMT5EncoderModel
        from transformers.utils import logging as hf_logging
        hf_logging.disable_progress_bar()
        root = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
        self.encoder = UMT5EncoderModel.from_pretrained(str(root / "text_encoder"), torch_dtype=dtype)
        # The checkpoint stores only ``shared.weight`` with tie_word_embeddings=false; transformers 5.x then
        # leaves encoder.embed_tokens zero-initialised instead of sharing it.
        self.encoder.encoder.embed_tokens = self.encoder.shared
        self.encoder = self.encoder.eval().requires_grad_(False).to(device)
        self.device = device
        self.max_len = max_len
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, prompts: list[str]) -> tuple[Tensor, Tensor]:
        batch = self.tokenizer(prompts, padding="max_length", max_length=self.max_len, truncation=True,
                               add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        ids = batch.input_ids.to(self.device)
        mask = batch.attention_mask.to(self.device).bool()
        hidden = self.encoder(ids, attention_mask=mask.long()).last_hidden_state.to(self.dtype)
        lengths = mask.sum(dim=1)
        longest = int(lengths.max().clamp(min=1))
        hidden = hidden[:, :longest] * mask[:, :longest, None].to(hidden.dtype)
        return hidden, mask[:, :longest]


__all__ = ["UMT5Online"]
