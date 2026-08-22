"""Write a tuned reference block into a task's anchor file, with its provenance.

Split out of `tune_reference.py` because the two jobs have different risk profiles: tuning is
a read-only search you may run a dozen times, while writing the anchor changes the meaning of
every score that task will ever produce.

**The anchor is written to `tasks/references/<task>.yaml`, not to the task file.** The task
file is what a competitor is handed, and a submission with filesystem access will read it —
one already did, lifting `kp` and `ti` out of a task and reporting them as its own tuning.
Publishing the anchor is right, since a reference nobody can argue with is worse than a weak
one; publishing it in the competitor's line of sight is not. `rtcbench.task` merges the two
at load time.

`max_total_variation` is a *scoring* setting, so it stays in the task file. Only the gains
move.

    python experiments/apply_reference.py --task tasks/x_v1.yaml \\
        --pairing 0 1 --kp 8.0 5.0 --ti 200 5 --cost 0.0186 \\
        --flatness "+/-25% on kp costs 1.6% to 1.9%" --max-tv 0.00069
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SIDECAR_STUB = """# Reference anchor for {task_id}.
#
# Kept out of the task file on purpose -- see the note in rtcbench.task.Task.load.

reference:
  kind: pid
  pairing: [0]
  kp: [1.0]
  ti: [30.0]
"""


def replace_block(text: str, header: str, new_block: str) -> str:
    """Replace a top-level YAML block, preserving everything around it."""
    pattern = re.compile(
        rf"(?:^#[^\n]*\n)*^{re.escape(header)}:.*?(?=^\S|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    if not pattern.search(text):
        raise ValueError(f"no top-level '{header}:' block found")
    return pattern.sub(new_block.rstrip() + "\n\n", text, count=1)


def set_max_tv(task_path: Path, value: float) -> None:
    """Update (or insert) max_total_variation in the TASK file, where scoring lives."""
    text = task_path.read_text(encoding="utf-8")
    comment = "   # ~4x the tuned reference's tv"
    if re.search(r"^\s+max_total_variation:.*$", text, re.MULTILINE):
        text = re.sub(
            r"^(\s+)max_total_variation:.*$",
            rf"\g<1>max_total_variation: {value:.6g}{comment}",
            text, count=1, flags=re.MULTILINE,
        )
    else:
        text = re.sub(
            r"^(scoring:\n(?:[ \t]+\S[^\n]*\n)*)",
            rf"\g<1>  max_total_variation: {value:.6g}{comment}\n",
            text, count=1, flags=re.MULTILINE,
        )
    task_path.write_text(text, encoding="utf-8")


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

    task_path = Path(args.task)
    sidecar = task_path.parent / "references" / task_path.name
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    if not sidecar.exists():
        sidecar.write_text(SIDECAR_STUB.format(task_id=task_path.stem), encoding="utf-8")

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
    sidecar.write_text(
        replace_block(sidecar.read_text(encoding="utf-8"), "reference", "\n".join(lines)),
        encoding="utf-8",
    )
    print(f"applied reference to {sidecar}")

    if args.max_tv is not None:
        set_max_tv(task_path, args.max_tv)
        print(f"set max_total_variation in {task_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
