# Retrieval: dense embeddings + FAISS over the content-hash document index.
#
# Consumes data/indexes/sections.jsonl (unique sections from the ingest index
# step), embeds them with a transformer (default nlpaueb/legal-bert-base-uncased)
# via mean-pooled last hidden state, and serves top-k cosine-similarity queries.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FAISS_INDEX = "sections.faiss"
IDS_JSONL = "section_ids.jsonl"
META_JSON = "meta.json"


def _mean_pool(model_output, attention_mask: np.ndarray) -> np.ndarray:
    """Mean-pool the last hidden state over non-padding tokens."""
    token_embeds = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float().to(token_embeds.device)
    summed = (token_embeds * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return (summed / counts).cpu().numpy()


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-9, None)


def embedTexts(texts: list[str], model_name: str, batch_size: int = 16) -> np.ndarray:
    """Embed a list of texts into L2-normalized dense vectors (n, dim)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

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
            )
            out = model(**enc)
            vecs.append(_mean_pool(out, enc["attention_mask"]))
    return _normalize(np.vstack(vecs).astype("float32"))


def buildEmbeddings(
    sections_path: Path,
    out_dir: Path,
    model_name: str = "nlpaueb/legal-bert-base-uncased",
    batch_size: int = 16,
) -> dict:
    """Embed unique sections from the index and write a FAISS index + id map."""
    import faiss

    out_dir.mkdir(parents=True, exist_ok=True)

    sections: list[dict] = []
    with sections_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                sections.append(json.loads(line))

    texts = [s["text"] for s in sections]
    vectors = embedTexts(texts, model_name, batch_size)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(out_dir / FAISS_INDEX))

    with (out_dir / IDS_JSONL).open("w", encoding="utf-8") as fh:
        for sec, vec in zip(sections, vectors):
            fh.write(
                json.dumps(
                    {
                        "id": sec["id"],
                        "text": sec["text"],
                        "n_occurrences": sec.get("n_occurrences", 1),
                        "sources": sec.get("sources", []),
                    }
                )
                + "\n"
            )

    meta = {
        "model": model_name,
        "dim": int(vectors.shape[1]),
        "n_sections": len(sections),
        "index": FAISS_INDEX,
    }
    (out_dir / META_JSON).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def queryEmbeddings(
    query: str,
    out_dir: Path,
    model_name: str = "nlpaueb/legal-bert-base-uncased",
    k: int = 5,
) -> list[dict]:
    """Search the built FAISS index; returns [{id, text, sources, score, rank}]."""
    import faiss

    index = faiss.read_index(str(out_dir / FAISS_INDEX))
    ids: list[dict] = []
    with (out_dir / IDS_JSONL).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                ids.append(json.loads(line))

    qvec = embedTexts([query], model_name)
    scores, idxs = index.search(qvec, k)
    hits: list[dict] = []
    for rank, (score, i) in enumerate(zip(scores[0], idxs[0])):
        if i < 0:
            continue
        sec = ids[int(i)]
        hits.append(
            {
                "rank": rank + 1,
                "id": sec["id"],
                "text": sec["text"],
                "sources": sec.get("sources", []),
                "score": round(float(score), 4),
            }
        )
    return hits