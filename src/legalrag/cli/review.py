# Interactive review: user verdicts on findings (accept / dismiss / edit).
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .output import colorize, format_finding


class ReviewAborted(Exception):
    """Raised when the user quits the review loop."""


def _read(prompt: Callable[[str], str], msg: str) -> str | None:
    """Prompt for input; None on EOF (non-interactive stdin)."""
    try:
        return prompt(msg)
    except EOFError:
        return None


def review_findings(
    findings: list[Any],
    prompt: Callable[[str], str] = input,
    colors: bool = False,
    max_findings: int | None = None,
) -> list[dict]:
    """Walk findings and collect user verdicts.

    Accepts Finding objects or plain dicts; converts once to dicts, mutates
    them in place (user_verdict / user_risk_level / user_note) and returns
    the dicts so the caller can persist the verdicts. Raises ReviewAborted
    when the user quits or stdin closes.
    """
    dicts = [f.toDict() if hasattr(f, "toDict") else f for f in findings]
    n = len(dicts)
    if max_findings is not None:
        n = min(n, max_findings)

    if n == 0:
        print("No findings to review.")
        return dicts

    print("Reviewing findings one at a time.")
    print("  [a] accept   [d] dismiss   [e] edit risk level   [n] add note")
    print("  [s] skip     [?] help      [q] quit")
    print("=" * 60)

    for i, d in enumerate(dicts[:n]):
        while True:
            print(f"\n--- Finding {i + 1}/{n} ---")
            print("\n".join(format_finding(d, colors)))
            if d.get("clause_text"):
                print(f"\n  clause:\n  {d['clause_text'][:400]}")
            if d.get("grounding"):
                print(f"  statute text: {d['grounding'][:300]}")
            choice = _read(prompt, "  action [a/d/e/n/s/?/q]: ")
            if choice is None:
                raise ReviewAborted()
            choice = choice.strip().lower()
            if choice == "q":
                raise ReviewAborted()
            if choice == "?":
                print("  [a] accept   [d] dismiss   [e] edit risk level")
                print("  [n] add note [s] skip      [q] quit")
                continue
            if choice == "s":
                d["user_verdict"] = "skipped"
                break
            if choice == "a":
                d["user_verdict"] = "accepted"
                break
            if choice == "d":
                d["user_verdict"] = "dismissed"
                break
            if choice == "e":
                level = _read(prompt, "  risk level [high/medium/low/info]: ")
                if level is None:
                    raise ReviewAborted()
                level = level.strip().lower()
                if level not in ("high", "medium", "low", "info"):
                    print("  invalid level, skipping edit")
                    continue
                d["user_risk_level"] = level
                d["user_verdict"] = "edited"
                break
            if choice == "n":
                note = _read(prompt, "  note: ")
                if note is None:
                    raise ReviewAborted()
                note = note.strip()
                d["user_note"] = note
                continue
            print("  unknown action")

    counts = {v: 0 for v in ("accepted", "dismissed", "edited", "skipped")}
    for d in dicts[:n]:
        counts[d.get("user_verdict", "skipped")] += 1
    print("\n" + "=" * 60)
    print(
        colorize(
            f"Review complete: {counts['accepted']} accepted, {counts['dismissed']} dismissed, "
            f"{counts['edited']} edited, {counts['skipped']} skipped",
            "info",
            colors,
        )
    )
    return dicts