# Shared transformer embedding cache.
#
# Both the classifier fallback (extract/classifier.py) and the dense retriever
# (retrieve/__init__.py) embed text with the same LegalBERT-style encoder.
# `getEncoder` caches the (tokenizer, model, device) triple per model name so
# repeated calls — one per section in the old per-section classify loop, one
# per finding in grounding — stop reloading the weights from disk.
from __future__ import annotations

from functools import lru_cache

import numpy as np


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=4)
def getEncoder(model_name: str, device: str):
    """Cached (tokenizer, model, device) for a model name."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def encodeMeanPooled(
    texts: list[str],
    model_name: str,
    batch_size: int = 16,
    device: str | None = None,
) -> np.ndarray:
    """Mean-pooled last-hidden-state embeddings, batched, raw (unnormalized)."""
    import torch

    device = device or default_device()
    tokenizer, model, device = getEncoder(model_name, device)

    vecs: list[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            token_embeds = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float().to(token_embeds.device)
            summed = (token_embeds * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            vecs.append((summed / counts).cpu().numpy())
    return np.vstack(vecs).astype("float32")


def clearEncoderCache() -> None:
    getEncoder.cache_clear()