# Themis CLI: `themis analyze`, `themis annotate`, `themis train`.
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    from . import analyze, annotate, train

    parser = argparse.ArgumentParser(prog="themis", description="Themis lease-analysis toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    sub_analyze = sub.add_parser("analyze", help="End-to-end lease analysis")
    analyze.build_parser(sub_analyze)
    sub_analyze.set_defaults(func=analyze.main)

    sub_annotate = sub.add_parser("annotate", help="Interactive section re-annotation")
    annotate.build_parser(sub_annotate)
    sub_annotate.set_defaults(func=annotate.main)

    sub_train = sub.add_parser("train", help="Train trainable components (classify/slm/ground)")
    train.build_parser(sub_train)
    sub_train.set_defaults(func=train.main)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
