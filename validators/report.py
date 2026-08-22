"""Roll the oracles up into one fidelity table.

    python -m validators.report
    python -m validators.report --plant four_tank --verbose

Each plant earns a tier from what has actually checked it. A tier is a claim about
*evidence*, never about quality, and the ladder is not a ranking of how good a plant is — a
measured disagreement is a better outcome than no comparison at all.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validators.integration import check as convergence_check  # noqa: E402
from validators.spectral import dc_gain, rga, step_direction  # noqa: E402


def cfg_for(task_id: str) -> dict:
    c = dict(yaml.safe_load((ROOT / "tasks" / f"{task_id}.yaml").read_text(
        encoding="utf-8"))["plant"])
    c.pop("kind")
    return c


@dataclass
class Claim:
    plant_id: str
    task_id: str
    claim: str
    measure: str
    passed: bool


# -- one function per published characterisation ----------------------------------


def four_tank_minimum_phase() -> Claim:
    """Johansson: gamma1+gamma2 > 1 puts both transmission zeros in the LHP, and the
    diagonal pairing is right. RGA(1,1) > 1 is the standard indicator."""
    r = rga(dc_gain("four_tank", cfg_for("four_tank_v1"), settle=1500))[0, 0]
    return Claim("four_tank", "four_tank_v1", "diagonal pairing correct",
                 f"RGA(1,1) = {r:+.2f}", r > 0.5)


def four_tank_non_minimum_phase() -> Claim:
    """The same rig with the valves moved: gamma1+gamma2 < 1 puts a zero in the RHP and the
    diagonal pairing becomes WRONG. A negative RGA is the textbook signature.

    This is also an independent confirmation of a task-design decision that was previously
    only established empirically: tuning both pairings gave 0.0379 off-diagonal against
    0.0994 diagonal, and the RGA says the same thing structurally.
    """
    r = rga(dc_gain("four_tank", cfg_for("four_tank_nmp_v1"), settle=1500))[0, 0]
    return Claim("four_tank", "four_tank_nmp_v1", "diagonal pairing WRONG",
                 f"RGA(1,1) = {r:+.2f}", r < 0.5)


def column_a_ill_conditioned() -> Claim:
    """Skogestad's Column A is the canonical ill-conditioned plant: a large condition number
    means the two loops fight, and a controller ignoring directionality either crawls or
    amplifies small input errors into large composition swings."""
    G = dc_gain("column_a", cfg_for("column_a_v1"), bump=0.02, settle=2500)
    k = float(np.linalg.cond(G))
    return Claim("column_a", "column_a_v1", "severely ill-conditioned",
                 f"cond(G) = {k:.0f}", k > 50)


def boiler_drum_inverse_response() -> Claim:
    """Shrink: cold feedwater collapses bubbles, so indicated level falls before it rises.
    The trap is that a naive controller reacts to the wrong-way excursion."""
    early, final = step_direction("boiler_drum", cfg_for("boiler_drum_v1"),
                                  actuator=0, channel=0, early=4, settle=600)
    return Claim("boiler_drum", "boiler_drum_v1", "inverse response (RHP zero)",
                 f"early {early:+.1f} vs final {final:+.1f}", early * final < 0)


def unstable_cstr_open_loop_unstable() -> Claim:
    """The middle branch of the ignition/extinction curve does not hold itself."""
    from rtcbench.plants import build_plant

    cfg = cfg_for("unstable_cstr_v1")
    p = build_plant({"kind": "unstable_cstr", **cfg})
    p.reset(11)
    x_op = p._x_op.copy()
    p._x = x_op.copy()
    u = p.spec.initial_actuation()
    for _ in range(200):
        p.step(u)
    departure = abs(p.audit().x[1] - x_op[1])
    p.close()
    return Claim("unstable_cstr", "unstable_cstr_v1", "operating point unstable",
                 f"departs {departure:.1f} K held", departure > 25.0)


def van_de_vusse_gain_reversal() -> Claim:
    """The defining pathology: the steady-state gain from dilution rate to product B changes
    sign across the operating range, so a controller tuned on one side goes unstable on the
    other."""
    from rtcbench.plants import build_plant

    cfg = cfg_for("van_de_vusse_v1")
    signs = []
    for frac in (0.25, 0.85):
        p = build_plant({"kind": "van_de_vusse", **cfg})
        p.reset(11)
        lo, hi = p.spec.actuator_lo(), p.spec.actuator_hi()
        base = lo[0] + frac * (hi[0] - lo[0])
        ys = []
        for d in (+0.03, -0.03):
            q = build_plant({"kind": "van_de_vusse", **cfg})
            q.reset(11)
            u = q.spec.initial_actuation()
            u[0] = float(np.clip(base * (1 + d), lo[0], hi[0]))
            for _ in range(2000):
                obs = q.step(u)
            ys.append(float(obs.y[0]))
            q.close()
        signs.append(np.sign(ys[0] - ys[1]))
        p.close()
    return Claim("van_de_vusse", "van_de_vusse_v1", "gain reverses sign",
                 f"sign(dCb/dF) = {signs[0]:+.0f} then {signs[1]:+.0f}",
                 signs[0] != signs[1])


CLAIMS = [
    four_tank_minimum_phase,
    four_tank_non_minimum_phase,
    column_a_ill_conditioned,
    boiler_drum_inverse_response,
    unstable_cstr_open_loop_unstable,
    van_de_vusse_gain_reversal,
]

TASKS = ["four_tank_v1", "four_tank_nmp_v1", "four_tank_blind_v1", "column_a_v1",
         "ph_neutralization_v1", "deadtime_process_v1", "van_de_vusse_v1",
         "boiler_drum_v1", "unstable_cstr_v1", "shell_fractionator_v1"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", default=None)
    ap.add_argument("--skip-convergence", action="store_true")
    args = ap.parse_args(argv)

    print("T2 - integration convergence (substep refinement)\n")
    converged: dict[str, bool] = {}
    if not args.skip_convergence:
        seen = set()
        for task in TASKS:
            cfg = dict(yaml.safe_load((ROOT / "tasks" / f"{task}.yaml").read_text(
                encoding="utf-8"))["plant"])
            kind = cfg.pop("kind")
            if kind in seen or (args.plant and kind != args.plant):
                continue
            seen.add(kind)
            r = convergence_check(kind, cfg, n_steps=40)
            converged[kind] = r.passed
            print(r.line())

    print("\nT1 - published characterisation, measured through the public interface\n")
    confirmed: dict[str, list[bool]] = {}
    for fn in CLAIMS:
        if args.plant and args.plant not in fn.__name__:
            continue
        c = fn()
        confirmed.setdefault(c.plant_id, []).append(c.passed)
        mark = "confirmed" if c.passed else "NOT CONFIRMED"
        print(f"  {c.plant_id:<18} {c.claim:<28} {c.measure:<30} [{mark}]")

    print("\nfidelity\n")
    plants = sorted(set(converged) | set(confirmed))
    for p in plants:
        has_claim = p in confirmed
        ok_claim = has_claim and all(confirmed[p])
        conv = converged.get(p)
        if has_claim and ok_claim and conv is not False:
            tier = "T1 verified   its published pathology is measurably present"
        elif has_claim and not ok_claim:
            tier = "T1 GAP        a published claim did not reproduce -- investigate"
        elif conv:
            tier = "T2 converged  integration sound; equations only self-checked"
        else:
            tier = "T3 self-consistent"
        print(f"  {p:<20} {tier}")

    missing = {"ph_neutralization", "deadtime_process", "shell_fractionator"} - set(confirmed)
    if missing and not args.plant:
        print(f"\n  no published-characterisation oracle yet: {', '.join(sorted(missing))}")
        print("  (their pathologies are asserted in tests/ by their own authors only)")

    bad = [p for p, v in confirmed.items() if not all(v)]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
