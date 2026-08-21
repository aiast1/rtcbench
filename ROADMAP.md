# Roadmap

Ordered by what the first full run actually exposed, not by what would be fun to build. Each
item says what evidence motivates it, so the ordering can be argued with.

**Where things stand (v0.2):** nine validated plants, 160 tests, anchored scoring with safety
and actuator-duty gates, replayable run records, and one complete multi-model run —
14 models × 8 tasks through an identical harness. See [leaderboard/](leaderboard/).

---

## Next

### 1. Sandbox the submission loader

`rtcbench.submission` imports a controller with the harness's own privileges. Until that
changes, **this benchmark cannot honestly accept a submission from a stranger.**

Not hypothetical. In the first trial run an agent with repository access read the reference
gains out of the task file, shipped `kp=1.95, ti=20.5` against a published `2.0, 20.0`, and
reported them as "derived through systematic experimentation". It scored exactly +1.000 on
every statistic, which is how it was caught.

Wanted: a subprocess with no network, no filesystem beyond its own directory, no import of
`rtcbench.plants`, and the per-step wall-clock budget enforced from outside rather than
measured from inside. The scan-overrun semantics already exist and should survive the move —
blowing the budget holds the last output, as a DCS does.

Splitting the reference config into a file the brief path never reads is a cheap partial
mitigation worth doing first.

### 2. An independent oracle for the plants

The plants are checked by tests written alongside them. That is self-consistency, not
correctness: **a sign error or a units slip would pass every test in the repo**, and the
tests would faithfully confirm the wrong model is stationary at its own wrong steady state.

Wanted: a dev-only `validators/` tree that integrates the same reactor in an independent
implementation and compares trajectories — Cantera (BSD, pip-installable) for the reacting
plants, an FMU or a published reference trajectory for the rest. Heavy, optional, never
imported by `src/`, never in the wheel.

This is also the honest use for a high-fidelity proprietary engine: not as a competitor, but
as the second implementation that says whether an open reimplementation is faithful.

### 3. Ship a blind-tier task

Every shipped task is `mismatch` — the brief hands over nominal parameters. The **blind**
tier, where a controller gets only a tag list, units, limits and a scan rate, is the design's
flagship and is currently unexercised. It is one field in a task file; what it needs is a
task designed so that system identification is genuinely required rather than optional.

Expect scores to fall sharply. That is the point.

### 4. Fix the difficulty cliff

Three of 112 submissions beat their task's reference, and on `four_tank_nmp_v1` **nobody**
did — 8 of 14 models hit the floor. A task where the whole field bottoms out ranks nothing;
it only says "hard". Meanwhile `column_a` had no floored results at all.

The suite currently has easy tasks and impossible ones and not much between. Wanted: a couple
of plants pitched at the gap, and possibly a partial-credit look at whether the NMP task's
floor is hiding real differences.

---

## Later

**Promote `experiments/commission.py` to a real `rtcbench commission` command.** The design
calls for a metered commissioning phase with a budget in sim-seconds and episodes, where an
agent bump-tests a training replica before submitting. Today's version is a working stand-in:
three rounds of cost feedback, no interactive experimentation.

**A sealed test set with published governance.** The salt-derived seed machinery exists and
is tested; the rotation schedule, who holds the salt, and how a submission is run against it
do not.

**Leaderboard automation.** Scoring is a manual pipeline. It should be one command that
produces a dated results file, regenerates the report, and refuses to overwrite an existing
one.

**Adapters, in priority order.** `rtcbench-fmi` first — FMI 2.0/3.0 co-simulation opens
OpenModelica, Dymola and Simulink as plant hosts with no license entanglement, because we
never link their source. Then OPC UA for hardware-in-the-loop. DWSIM/CAPE-OPEN lives in its
own GPL repo and can never host a scored task.

**More plants.** Tennessee Eastman is the obvious gap — plantwide, recycle, safety trips,
economics — and needs a cleanly-licensed reimplementation path confirmed before it starts.

**Human baseline.** Two or three real control engineers submitting under the same budget. A
benchmark with a human reference line is worth several times one without, and right now the
only reference is a PI the maintainers tuned.

---

## Open questions

- **A neutral home.** The `rtcbench` GitHub org is taken by an unrelated party, so the
  project sits on a personal account. Needs a name and a maintainer group before it moves.
- **What "RTC" expands to.** Assumed "real-time control"; never confirmed.
- **Whether a repo-access track is worth keeping**, separately labelled. "What does an agent
  with a filesystem do" is a legitimate and interesting measurement — the honest answer so
  far is *it reads the answer key* — but it is a different benchmark from this one and must
  never share a table with it.
- **Whether the score floor at −1.0 is the right constant.** It fixed a real defect (one
  narrow-anchor task was dominating suite means) and it reordered the ranking substantially,
  which means the choice is load-bearing and deserves more scrutiny than it has had.

---

## Deliberately not doing

- **Rebuilding a thermodynamics or physical-properties engine.** No property database, no
  equation of state, no flash algorithm. Plants are small published ODEs with constants from
  their source papers. Anything needing real thermo belongs behind an adapter.
- **Vendoring a GPL or proprietary engine into the scored suite.** Not a licensing
  preference — a reproducibility one. A stranger with a laptop must be able to re-run any
  scored result, which rules out closed engines, COM-bound engines and GPL engines alike, for
  the same reason.
- **A live agent-in-the-loop track as the main leaderboard.** Submissions are sealed
  artifacts so they can be re-scored on tasks written years later. A transcript cannot be.
- **Adding dependencies to core.** numpy and pyyaml. The plotting is hand-rolled SVG
  precisely so `pip install rtcbench` stays a two-package install.
