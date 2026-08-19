"""Repo verification: lint, tests, and dataset regeneration.

Runs ruff, pytest, then (if raw data is present) re-ingests the Leivaditi and
LEXDEMOD corpora and regenerates the EDA report so the tracked baseline can be
compared. If a sample document exists (data/samples/), also exercises the
document ingestion CLI. Finally, checks the SLM field-fidelity eval (when the
tuned GGUF is present) for parse integrity and prose quality. Exits non-zero
on any failure.

Usage:
    PYTHONUNBUFFERED=1 python scripts/verify_repo.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
REPORT = ROOT / "eval" / "eda_report.json"
SAMPLE = ROOT / "data" / "samples" / "sample_lease.txt"
FIELD_FIDELITY = ROOT / "eval" / "finetune" / "field_fidelity.json"
GGUF = ROOT / "models" / "qwen2.5-1.5b" / "qwen2.5-1.5b-instruct-tuned-q8_0.gguf"

STEPS: list[tuple[str, list[str]]] = [
    ("ruff", [PY, "-m", "ruff", "check", "src", "scripts", "tests"]),
    ("pytest", [PY, "-m", "pytest", "tests", "-q"]),
    ("ingest leivaditi", [PY, "scripts/ingest_leivaditi_full.py"]),
    ("ingest lexdemod", [PY, "scripts/ingest_lexdemod.py"]),
    ("eda report", [PY, "scripts/dataset_eda.py", "--json", str(REPORT)]),
    ("build index", [PY, "scripts/build_index.py"]),
    ("build embeddings", [PY, "scripts/build_embeddings.py"]),
    ("build statute index", [PY, "scripts/build_statute_index.py"]),
    ("train smoke", [PY, "scripts/train_smoke.py"]),
    ("eval fast-lane", [PY, "scripts/eval_fastlane.py"]),
    ("eval ood", [PY, "scripts/eval_ood.py"]),
]
if SAMPLE.exists():
    STEPS.append(("analyze sample", [PY, "scripts/analyze_document.py", str(SAMPLE)]))


def check_field_fidelity() -> bool:
    """Assert the SLM eval shows parse integrity and prose quality."""
    if not GGUF.exists():
        print("[verify] skip field fidelity (no tuned GGUF)", flush=True)
        return True
    if not FIELD_FIDELITY.exists():
        print("[verify] FAIL field fidelity (eval missing)", flush=True)
        return False
    data = json.loads(FIELD_FIDELITY.read_text())
    checks = {
        "parse_rate": data["parse_rate"] == 1.0,
        "template_impact_rate": data["template_impact_rate"] == 0.0,
        "prose_exact_rate": data["prose_exact_rate"] == 0.0,
        "field_fidelity": data["field_fidelity"] >= 0.7,
    }
    ok = all(checks.values())
    print(f"[verify] {'ok' if ok else 'FAIL'} field fidelity "
          f"(fid={data['field_fidelity']:.3f}, parse={data['parse_rate']}, "
          f"template={data['template_impact_rate']}, echo={data['prose_exact_rate']})",
          flush=True)
    if not ok:
        for name, passed in checks.items():
            if not passed:
                print(f"[verify]   bad {name}", flush=True)
    return ok


def main() -> int:
    failures = 0
    for name, cmd in STEPS:
        print(f"[verify] {name} ...", flush=True)
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            failures += 1
            print(f"[verify] FAIL {name}\n{proc.stdout}{proc.stderr}", flush=True)
        else:
            print(f"[verify] ok {name}", flush=True)
    if not check_field_fidelity():
        failures += 1
    if failures:
        print(f"[verify] {failures} step(s) failed", flush=True)
        return 1
    report = json.loads(REPORT.read_text())
    summary = {
        "leases_rows": report["leases"]["rows"],
        "redflags_rows": report["redflags"]["rows"],
        "full_docs": report["full_benchmark"]["docs"]["rows"],
        "full_redflag_positives": report["full_benchmark"]["redflags"]["positive"],
        "lexdemod_rows": report["lexdemod"]["rows"],
    }
    print("[verify] report summary:", json.dumps(summary), flush=True)
    print("[verify] all ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())