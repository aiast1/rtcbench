# Validators

**The question this tree exists to answer: is a plant actually right?**

Everything in `tests/` is written by whoever wrote the plant, alongside the plant. That
catches typos and regressions and is worth having, but it is *self-consistency*, not
correctness — a sign error or a units slip would pass every one of those tests, and they
would faithfully confirm the wrong model is stationary at its own wrong steady state. A
benchmark built on a plant like that measures nothing, confidently.

So this tree checks the plants against something that did not come from the plant.

**Dev-only, always.** Nothing here is imported by `src/rtcbench`, nothing here ships in the
wheel, and `pip install rtcbench` stays a two-package install. The oracles are heavy
(scipy, Cantera) precisely because they are allowed to be.

```bash
pip install -e ".[validators]"
python -m validators.report                 # the fidelity table
python -m validators.report --plant van_de_vusse --verbose
```

## The two oracles, and what each can and cannot prove

**T2 — integration accuracy** (`integration.py`, scipy). Re-integrates the plant's *own*
right-hand side with an adaptive stiff solver at tight tolerance and compares trajectories.

This proves the fixed-step RK4 is converged — that the substep count is enough for the
dynamics. It is a real risk: a stiff plant with too few substeps produces a trajectory that
is smooth, plausible, reproducible, and wrong. It does **not** prove the right-hand side is
right, because it uses the same right-hand side.

**T1 — independent physics** (`cantera_cstr.py`, Cantera). Builds the same reactor from the
source paper's kinetics and thermodynamics in an independent implementation and compares.
Different code, different author, different numerics.

This is the one that catches a transcription error. It only applies where the plant is
genuinely chemical kinetics, which in this pack means the reacting CSTRs.

## The tier a plant earns

| Tier | Meaning |
|---|---|
| **T1 verified** | agrees with an independent physics implementation |
| **T1 gap** | compared against one, and the disagreement is measured and stated |
| **T2 converged** | integration is converged, but only self-consistency vouches for the equations |
| **T3 self-consistent** | nothing but its own tests |

A tier is a claim about evidence, never about quality — and a measured gap is a *better*
outcome than no comparison, so `T1 gap` outranks `T2 converged`. Do not collapse them.

## Rules

- An oracle failure is the **plant's** problem until proven otherwise. The temptation is to
  loosen the tolerance; do not.
- Never import from `validators/` in `src/`. There is a test that enforces this.
- If an oracle is wrong, say so in its docstring and keep it — a retired oracle with its
  reasoning recorded is worth more than a deleted one.
