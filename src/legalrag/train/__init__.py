"""Themis trainable components: classifier, SLM fine-tune, grounding gate.

`themis train classify` fits the clause-classifier fallback on auto- or
gold-labeled sections; `themis train slm` LoRA-tunes the plain-language
model; `themis train ground` fits the statute-grounding relevance gate.
Segment (heuristics) and risk (hand-authored rules) are NOT trainable by
design - "training" them means editing rules.py triggers and anchors.
"""
from legalrag.train.classify import DEFAULT_MODEL as CLASSIFY_MODEL
from legalrag.train.classify import train_classifier
from legalrag.train.data import (
    build_statute_pairs,
    load_auto_labels,
    load_finetune_pairs,
    load_gold_labels,
    load_statute_chunks,
)
from legalrag.train.ground import train_gate
from legalrag.train.slm import (
    DEFAULT_DATA,
    DEFAULT_EVAL,
    DEFAULT_OUT,
    finetune,
)
from legalrag.train.slm import (
    DEFAULT_MODEL as SLM_MODEL,
)

__all__ = [
    "CLASSIFY_MODEL",
    "DEFAULT_DATA",
    "DEFAULT_EVAL",
    "DEFAULT_OUT",
    "SLM_MODEL",
    "build_statute_pairs",
    "finetune",
    "load_auto_labels",
    "load_finetune_pairs",
    "load_gold_labels",
    "load_statute_chunks",
    "train_classifier",
    "train_gate",
]