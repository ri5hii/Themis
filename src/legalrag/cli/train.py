# `themis train` — train the trainable components.
#
# Subcommands: classify (clause-classifier fallback), slm (plain-language
# LoRA fine-tune), ground (statute-grounding relevance gate).
#
# NOT trainable by design: segment (document segmentation heuristics) and
# risk (hand-authored trigger rules) — "training" those means editing
# rules.py triggers/anchors, so the CLI never implies otherwise.
from __future__ import annotations

import argparse
from pathlib import Path

from legalrag.train import (
    CLASSIFY_MODEL,
    DEFAULT_DATA,
    DEFAULT_EVAL,
    SLM_MODEL,
    load_auto_labels,
    load_gold_labels,
    load_statute_chunks,
    train_classifier,
    train_gate,
)
from legalrag.train.ground import DEFAULT_MODEL as GROUND_MODEL
from legalrag.train.slm import finetune

ROOT = Path(__file__).resolve().parents[3]
AUTO_LABELS = ROOT / "data" / "annotated" / "leivaditi_leases.jsonl"
ANNOTATED_DIR = ROOT / "data" / "annotated"
STATUTES_DIR = ROOT / "data" / "statutes"
CLASSIFIER_OUT = ROOT / "models"
GROUNDING_OUT = ROOT / "models" / "grounding"
LORA_OUT = ROOT / "models" / "finetuned" / "lora"


def build_classify_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(
        prog="themis train classify",
        description="Fit the clause-classifier fallback on frozen embeddings.",
    )
    parser.add_argument("--data", choices=("auto", "gold"), default="auto",
                        help="auto: fast-lane labels from leivaditi_leases.jsonl; "
                             "gold: redflag sentences + themis annotate sections (default auto)")
    parser.add_argument("--model", default=CLASSIFY_MODEL, help="LegalBERT-family model for embeddings")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", "-o", default=str(CLASSIFIER_OUT), help="artifact dir (default models/)")
    return parser


def run_classify(ns: argparse.Namespace) -> int:
    if ns.data == "gold":
        texts, labels = load_gold_labels(ANNOTATED_DIR)
    else:
        texts, labels = load_auto_labels(AUTO_LABELS)
    if len(texts) < 4:
        print(f"[error] not enough labeled sections ({len(texts)}); run `themis annotate` first")
        return 1
    train_classifier(
        texts, labels,
        model_name=ns.model,
        test_size=ns.test_size,
        seed=ns.seed,
        out_dir=Path(ns.output),
    )
    return 0


def run_classify_cli(argv: list[str] | None = None) -> int:
    return run_classify(build_classify_parser().parse_args(argv))


def build_slm_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(
        prog="themis train slm",
        description="LoRA fine-tune the plain-language SLM on aligned pairs.",
    )
    parser.add_argument("--model", default=SLM_MODEL, help="Base model")
    parser.add_argument("--train-data", default=DEFAULT_DATA)
    parser.add_argument("--eval-data", default=DEFAULT_EVAL)
    parser.add_argument("--output", "-o", default=str(LORA_OUT))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--load-8bit", action="store_true", help="QLoRA: 8-bit base (needs bitsandbytes, CUDA)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--eval-only", action="store_true", help="Skip training, eval only")
    return parser


def run_slm(ns: argparse.Namespace) -> int:
    return finetune(
        model_name=ns.model,
        train_data_path=ns.train_data,
        eval_data_path=ns.eval_data,
        out_dir=Path(ns.output),
        epochs=ns.epochs,
        lr=ns.lr,
        r=ns.r,
        alpha=ns.alpha,
        max_length=ns.max_length,
        load_8bit=ns.load_8bit,
        batch_size=ns.batch_size,
        grad_accum=ns.grad_accum,
        eval_only=ns.eval_only,
    )


def run_slm_cli(argv: list[str] | None = None) -> int:
    return run_slm(build_slm_parser().parse_args(argv))


def build_ground_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(
        prog="themis train ground",
        description="Fit the statute-grounding relevance gate (dense + BM25).",
    )
    parser.add_argument("--data", default=str(STATUTES_DIR), help="statute md sources (default data/statutes)")
    parser.add_argument("--out", default=str(GROUNDING_OUT), help="artifact dir (default models/grounding)")
    parser.add_argument("--model", default=GROUND_MODEL, help="embedding model for dense features")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neg-per-pos", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser


def run_ground(ns: argparse.Namespace) -> int:
    from legalrag.risk.rules import RULES

    chunks = load_statute_chunks(Path(ns.data))
    if not chunks:
        print(f"[error] no statute chunks under {ns.data}")
        return 1
    try:
        train_gate(
            chunks,
            RULES,
            model_name=ns.model,
            out_dir=Path(ns.out),
            seed=ns.seed,
            neg_per_pos=ns.neg_per_pos,
            test_size=ns.test_size,
        )
    except ValueError as e:
        print(f"[error] {e}")
        return 1
    return 0


def run_ground_cli(argv: list[str] | None = None) -> int:
    return run_ground(build_ground_parser().parse_args(argv))


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(
        prog="themis train",
        description=(
            "Train the trainable components. Segment (heuristics) and risk "
            "(hand-authored rules) are not trainable - editing rules.py "
            "triggers/anchors is the 'training' for those."
        ),
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    sub_classify = sub.add_parser("classify", help="Clause-classifier fallback (auto or gold labels)")
    build_classify_parser(sub_classify)

    sub_slm = sub.add_parser("slm", help="LoRA fine-tune the plain-language SLM")
    build_slm_parser(sub_slm)

    sub_ground = sub.add_parser("ground", help="Statute-grounding relevance gate")
    build_ground_parser(sub_ground)
    return parser


def main(args: argparse.Namespace) -> int:
    """Dispatch a parsed `themis train` namespace to the subcommand runner."""
    if args.subcommand == "classify":
        return run_classify(args)
    if args.subcommand == "slm":
        return run_slm(args)
    if args.subcommand == "ground":
        return run_ground(args)
    raise ValueError(f"unknown train subcommand: {args.subcommand}")