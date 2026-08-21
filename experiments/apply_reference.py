"""Write a tuned reference block into a task file, with its provenance.

Splitting this out of tune_reference.py because the two jobs have different risk profiles:
tuning is a read-only search you may run a dozen times, while writing the anchor edits a
task file and changes the meaning of every score that task will ever produce.

The provenance comment is not decoration. The reference *is* the unit of measurement, so a
reader has to be able to see how it was obtained and argue with it — that is the whole
reason the gains are published rather than hidden.

    python experiments/apply_reference.py --task tasks/x_v1.yaml \\
        --kp 8.0 5.0 --ti 200 5 --beta 0.6 1.0 --pairing 0 1 \\
        --cost 0.0186 --flatness "+/-25% on kp costs 1.6% to 1.9%" --max-tv 0.00069
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def replace_block(text: str, header: str, new_block: str) -> str:
    """Replace a top-level YAML block, preserving everything around it."""
    pattern = re.compile(
        rf"(?:^#[^\n]*\n)*^{re.escape(header)}:.*?(?=^\S|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    if not pattern.search(text):
        raise ValueError(f"no top-level '{header}:' block found")
    return pattern.sub(new_block.rstrip() + "\n\n", text, count=1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--kp", type=float, nargs="+", required=True)
    ap.add_argument("--ti", type=float, nargs="+", required=True)
    ap.add_argument("--beta", type=float, nargs="+", default=None)
    ap.add_argument("--pairing", type=int, nargs="+", required=True)
    ap.add_argument("--cost", type=float, required=True)
    ap.add_argument("--flatness", default="")
    ap.add_argument("--max-tv", type=float, default=None)
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    path = Path(args.task)
    text = path.read_text(encoding="utf-8")
    beta = args.beta or [1.0] * len(args.kp)

    lines = [
        "# Provenance: experiments/tune_reference.py, coordinate descent over (kp, ti, beta)",
        f"# per loop. Best mean cost {args.cost:.6f} with zero gated scenarios.",
    ]
    if args.flatness:
        lines.append(f"# Flatness: {args.flatness} -- a broad basin, not a fit to the seeds.")
    if args.note:
        lines += [f"# {ln}" for ln in args.note.split("\n")]
    lines += [
        "# The reference defines 1.0 on this task, so an under-tuned one inflates every score",
        "# here forever. If a naive PI beats it, the reference is wrong, not the naive PI.",
        "reference:",
        "  kind: pid",
        f"  pairing: {args.pairing}",
        f"  kp: {[round(v, 4) for v in args.kp]}",
        f"  ti: {[round(v, 3) for v in args.ti]}",
        f"  beta: {[round(v, 3) for v in beta]}",
    ]
    text = replace_block(text, "reference", "\n".join(lines))

    if args.max_tv is not None:
        if re.search(r"^\s+max_total_variation:.*$", text, re.MULTILINE):
            text = re.sub(r"^(\s+)max_total_variation:.*$",
                          rf"\g<1>max_total_variation: {args.max_tv:.6g}"
                          rf"   # ~4x the tuned reference's tv",
                          text, count=1, flags=re.MULTILINE)
        else:
            text = re.sub(r"^(scoring:\n(?:[ \t]+\S[^\n]*\n)*)",
                          rf"\g<1>  max_total_variation: {args.max_tv:.6g}"
                          rf"   # ~4x the tuned reference's tv\n",
                          text, count=1, flags=re.MULTILINE)

    path.write_text(text, encoding="utf-8")
    print(f"applied reference to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
