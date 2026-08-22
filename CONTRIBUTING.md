# Contributing to RTCbench

Contributions are welcome — plants especially. Read this first; a benchmark has stricter
rules than an ordinary library because a careless change does not break the build, it
silently changes what every past number meant.

## Sign your commits off (DCO, not a CLA)

```bash
git commit -s -m "..."
```

That adds a `Signed-off-by:` line certifying you wrote the contribution or have the right to
submit it, per the [Developer Certificate of Origin](https://developercertificate.org/).
There is no CLA — they deter contributors and a DCO is sufficient.

Code is Apache-2.0; tasks, results and documentation are CC-BY-4.0. Apache carries an
explicit patent grant, which matters in a patented field like process control; it is
otherwise as permissive as MIT.

## Before you open a pull request

```bash
python -m pytest tests/ -q                      # must be green
rtcbench validate --task tasks/<anything you touched>.yaml   # must print PASS
```

Describe *what you measured*, not what you intended. "Tuned the reference, mean cost 0.0039,
cost basin flat to 3% at ±25% on kp" is a reviewable claim. "Improved the reference" is not.

---

## Adding a plant — the most valuable contribution

Follow [docs/AUTHORING_PLANTS.md](docs/AUTHORING_PLANTS.md). It is three files you own
outright and nothing shared, so several plants can be authored in parallel without
colliding.

**A plant must defeat a naive controller that the existing pack does not.** State which one
in the module docstring: inverse response, deadtime dominance, input multiplicity, open-loop
instability, ill-conditioning, sign-reversing gain. A plant a textbook PI handles comfortably
adds nothing — the suite already has nine, and the interesting question is which *new*
failure mode you are introducing.

**Reimplement from published equations and cite the source.** No property databases, no
equation-of-state solvers, no reimplementation of Cantera or DWSIM. Every plant here is a
small ODE with constants lifted from its paper; that is why the physics is 4–54 lines. If a
plant needs real thermodynamics, it belongs behind an adapter (FMI, CAPE-OPEN), not in the
scored pack.

**Ship tests that measure the pathology, not tests that assert it.** For an inverse-response
plant, step the input and assert the output moves the *wrong way first*, with the ordering
checked. For an unstable one, hold the actuators and assert the state departs and does not
return. A test that only checks the plant runs is worth nothing here.

---

## Changing scoring, the harness, or an existing task

This is where a benchmark can be damaged invisibly. Three questions, in order:

**1. Does this change what an existing published number means?** If yes it needs a version
bump and a note in `leaderboard/`, never a silent edit. Published task files are immutable —
their content hash appears in every run record. Cut `<id>_v2.yaml`.

**2. Does it make a task easier in a way that flatters submissions?** Weakening a
disturbance, loosening a constraint or shrinking the mismatch spread to get `validate` to
pass is fixing the thermometer. If a task fails validation, the task is mis-specified — fix
the specification and say in a comment what you changed and why.

**3. Could a submission detect and exploit it?** Task files are readable by anything with
filesystem access. In the first trial run, an agent with repository access lifted the
reference gains out of a task file and reported them as its own tuning; the anchor-copy
check in `experiments/leaderboard.py` exists because of that.

If you change a task's scoring weights or disturbances, **you must retune its reference** and
record the provenance above the `reference:` block in `tasks/references/<task>.yaml` — the
anchor lives there, not in the task file, so a competitor reading the task does not get
handed the answer. The reference defines 1.0; leaving it
stale re-denominates every score on that task.

---

## Submitting a result to the leaderboard

Submissions are sealed controller artifacts, scored by the ordinary harness with no special
case. Include the run record and the commissioning budget (tokens, wall-clock, sim time) —
score without cost is half a result.

**Conflict-of-interest rule, published up front:** contributions from anyone affiliated with
Acaysia are labelled as such on the leaderboard, and no maintainer scores their own
submission on a sealed set without a second maintainer countersigning. This is written down
before there is anything to be accused of, which is the only time writing it down is worth
much.

Results files in `leaderboard/` are never edited after the fact. A leaderboard you can
quietly revise is not evidence. Corrections are new files with a note explaining what changed
and why.

---

## What gets rejected

- A plant with no cited source, or one that is a re-parameterisation of an existing plant's
  pathology.
- Loosening a task to make a submission — including your own — score better.
- A reference anchor a naive PI can beat. That inflates every score on the task permanently;
  retune it before shipping.
- New dependencies in `src/rtcbench/`. Core keeps numpy and pyyaml. Tooling that needs more
  goes in `experiments/`, adapters go out of tree.
- Anything that widens `Observation` to include plant state.
- Results that cannot be recomputed from their own run record.

## Reporting a defect in the benchmark itself

The most useful issue you can file is evidence that a score is measuring the wrong thing —
an unwinnable scenario, an under-tuned reference, a task a submission can game, an anchor
pair too narrow to be meaningful. Those are bugs of the highest severity here, and several
of the design's current rules exist because someone found one.
