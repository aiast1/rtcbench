# Working in this repo

Orientation for coding agents. Humans should read [README.md](README.md) and
[DESIGN.md](DESIGN.md) first; this file is the operational layer on top of them.

RTCbench scores language models on closed-loop control: a model writes a controller, the
controller is wired to a dynamic process simulation it has never seen, and it is judged on
what the plant does. Everything here follows from that being a *measurement*, so the
overriding rule is that changes must not quietly alter what a number means.

---

## Layout

```
src/rtcbench/          core. two dependencies (numpy, pyyaml). keep it that way.
  plant.py             the Plant protocol -- the only thing core knows about an engine
  controller.py        what a submission implements; TaskBrief is what it is told
  harness.py           the sealed loop. one function does the actual benchmarking
  instruments.py       noise, deadtime, saturation, stiction -- separate from physics
  metrics.py score.py  cost decomposition, anchored scoring, the safety/duty gates
  task.py              tasks are data; content-hashed; immutable once published
  record.py trend.py   run records that can recompute their own score, and SVG trends
  baselines/           the hold and reference anchors -- these DEFINE 0.0 and 1.0
  plants/              one file per plant, auto-discovered
tasks/                 one YAML per task. IMMUTABLE once published -- cut a v2
tests/                 properties, not coverage. 160 tests, ~40 s
experiments/           tooling that is not part of the scored path (boto3 lives here)
leaderboard/           committed results + the rendered report. never edited after the fact
docs/                  AUTHORING_PLANTS.md is the contract for adding a plant
```

## Commands

```bash
python -m pytest tests/ -q                                  # ~40 s, must stay green
rtcbench validate --task tasks/<id>.yaml                     # must print PASS
rtcbench show     --task tasks/<id>.yaml                     # the brief a competitor gets
rtcbench score    --task tasks/<id>.yaml --controller x.py
python experiments/tune_reference.py --task tasks/<id>.yaml  # run in the FOREGROUND
python experiments/matrix.py --tasks tasks/*.yaml            # whole suite, slow
python experiments/report.py --out leaderboard/report.html
```

---

## Invariants — breaking one invalidates published scores

1. **A plant is a pure function of `(seed, control sequence)`.** Fixed-step RK4, seeded
   `np.random.default_rng((seed, <plant constant>))`, never the global RNG, never wall-clock.
   Every reproducibility claim reduces to this.
2. **`Observation` carries measurements only** — `t`, `y`, `quality`. Never state. Ground
   truth goes to `Plant.audit()`, which the harness calls and a controller cannot reach. If
   you find yourself widening `Observation`, stop: that is the benchmark quietly becoming
   about something else.
3. **A published task file is never edited.** Its content hash is in every run record. Cut
   `<id>_v2.yaml` instead. Changing a task's disturbance, weights, or seeds silently changes
   what every historical score on it meant.
4. **The reference anchor defines 1.0.** Retuning it re-denominates every score on that task.
   If you change a task's scoring weights or disturbances, you *must* retune, and record the
   provenance in a comment above the `reference:` block.
5. **Core keeps two dependencies.** numpy and pyyaml. No plotting library (see `trend.py`),
   no cloud SDK (see `experiments/`). The promise is `pip install` and go on a laptop.
6. **A failure is recorded, never dropped.** A controller that crashes, overruns, violates or
   exceeds its duty limit gets a gated record and stays in its ensemble. Dropping it would
   score a submission only on the scenarios it happened to survive.

---

## Gotchas that have already cost this project real time

**Run the reference tuner in the foreground.** It takes minutes. Seven plant-authoring agents
each backgrounded it, ended their turn waiting for a notification, got re-invoked, waited
again — five burned 240k–270k tokens producing nothing, and two tasks shipped with
placeholder gains. If a tool's runtime exceeds your turn budget, that is a reason to make the
tool faster, not to wait on it.

**A search grid that cannot express the answer fails with false confidence.** The tuner's
`TI_GRID` originally stopped at 200 s. On a plant with an RHP zero the correct integral time
is thousands of seconds, so every in-grid candidate violated and the tuner reported that *no
safe tuning existed* — which reads as "impossible task" and means "wrong grid". Boiler drum's
optimum is `ti = 2000 s`.

**Tune multi-loop plants on all 20 seeds, not the default 6.** A gain set safe on a sample is
not necessarily safe on the ensemble; Shell fractionator gated on one of the other 14.

**Size a task's mismatch against the initial condition.** If a parameter draw starts the
plant near a constraint, that scenario is unwinnable before the controller acts, and an
unwinnable scenario is noise. `four_tank_v1`'s first draft put some draws 1.6 cm from an
overflow at t=0.

**Feedback seeds must be disjoint from scored seeds.** `experiments/commission.py` defaults
to 1001–1005 for exactly this reason. Pointing feedback at a task's own ensemble is tuning on
the test set.

**Do not override the git identity.** Commit with plain `git commit`; the repo-local config
is already correct. Passing `-c user.name=...` once put an invented author and a work email
into the public history of all 12 commits and needed a rewrite plus a force-push to undo.

**Look at rendered output before believing it.** Two bugs in the results page survived source
review and died instantly on screenshot: a tooltip parked outside the token scope, painting
unstyled; and two conflicting `display` declarations on one rule.

---

## Reviewing a change to scoring or tasks

Ask, in order:

1. Does this change what an existing published number means? If yes, it needs a version bump
   and a note in `leaderboard/`, not a silent edit.
2. Does it make a task easier in a way that flatters submissions? Weakening a disturbance,
   loosening a constraint, or shrinking the mismatch to make `validate` pass is fixing the
   thermometer.
3. Would a submission be able to detect it and exploit it? The task file is readable by
   anything with filesystem access — which is how an agent once lifted the reference gains
   verbatim and reported them as its own tuning.

---

## House style

Comments explain *why*, especially why an obvious alternative was rejected — several modules
carry the measurement that settled a design question, and those are the most valuable lines
in the file. Record corrections rather than quietly fixing them: `tests/test_unstable_cstr.py`
documents an assumption its author got wrong, because the next reader will have the same
intuition. Tests assert measured behaviour, never assumed behaviour.
