# Roadmap

Ordered by what the first full run actually exposed, not by what would be fun to build. Each
item says what evidence motivates it, so the ordering can be argued with.

**Where things stand (v0.2):** nine validated plants, 178 tests, anchored scoring with safety
and actuator-duty gates, replayable run records, and one complete multi-model run —
14 models × 8 tasks through an identical harness. See [leaderboard/](leaderboard/).

---

## Next

### 1. Finish hardening the sandbox  *(first cut shipped)*

`--sandbox` now runs a submission in a spawned process that cannot import `rtcbench` at all
(the brief crosses as plain data and is rebuilt on the far side), cannot open a socket, and
has its step budget enforced by the parent. A late block is a scan overrun and holds its
previous output; a silent one is killed at 20x the budget. Eighteen tests pin those claims
individually.

Motivation, for the record: an agent with repository access read the reference gains out of
a task file, shipped `kp=1.95, ti=20.5` against a published `2.0, 20.0`, and reported them as
"derived through systematic experimentation". It scored exactly +1.000, which is how it was
caught.

Still open, and the reason this is not marked done:

- **Filesystem confinement.** Python cannot do it portably. A submission can still read the
  task file if it knows the path — so the sandbox does not yet close the exact hole that
  motivated it. Splitting the reference config out of the competitor-readable task file is
  the cheap fix and should land next.
- **`ctypes`, `subprocess` and friends** are untouched. The honest framing is an
  honest-mistake barrier; genuinely untrusted code belongs in a container.
- **Wiring into `experiments/commission.py`**, which still loads in-process.

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

### 3. Blind tier on a HARD plant  *(the easy half shipped)*

`four_tank_blind_v1` shipped and exercises the tier: same rig, same seeds, bit-identical
anchors, model withheld. It produced a genuine negative result — the **median effect of
withholding the model is -0.028**, four models better without it, six worse, ten unchanged.
On that plant, being handed the parameters is worth approximately nothing.

That is interesting, and it also means the task is not yet testing what the tier is FOR. A
four-tank is simple enough that a robustly-tuned PI does not need a model, so withholding one
costs nothing. The tier only bites where system identification is genuinely required.

Wanted: a blind variant of a plant where the model actually matters — `van_de_vusse` (whose
gain reverses sign, so a controller that has not found the reversal will be tuned on the
wrong side of it) or `deadtime_process` (where the delay must be estimated to be
compensated). Expect a large negative effect there, and if there is not one, that is a real
finding about how little models use the model.

### 4. A reference controller class beyond decentralized PID

Every reference anchor in the pack is a bank of independent PI loops, and on one task that
is demonstrably the wrong class. Shell fractionator is a constrained-MPC benchmark -- 3x3,
per-element deadtimes, interacting loops -- and its PID reference is only 37% better than
freezing the actuators, against 94% on four_tank.

That cannot be fixed by making the task harder: strengthening it scaled hold 0.064 -> 0.146
and the reference 0.0435 -> 0.0923, both by 2.3x, moving the ratio only 32% -> 37%. A harder
task hurts both anchors in proportion. The anchor is the thing that is wrong.

Wanted, cheapest first: a static decoupler plus PI (little more than a gain matrix inverse),
then a proper constrained MPC for the tasks that deserve one. The task file carries a
PROVISIONAL banner until this lands, and `shell_fractionator_v1` should not be quoted as a
ranking in the meantime.

### 5. Fill the gap between "hard" and "impossible"  *(partly resolved)*

Relocating the score floor did most of the work here. The ladder by median score is now
reasonably continuous down to zero:

    blind +0.84 · four_tank +0.66 · deadtime +0.35 · column_a +0.29 · pH +0.16
    boiler 0.00 · van_de_vusse -0.00 · unstable -0.05 · nmp -1.88 · shell -7.58

What remains is the tail. `four_tank_nmp` at a -1.88 median is a genuine cliff — the field
does not merely find it hard, most of it is worse than doing nothing — and `shell` is
provisional pending a better reference class (item 4).

Two open questions rather than one task:

- Is the NMP cliff real or an artefact of the wrong pairing being catastrophic rather than
  merely bad? A submission that pairs diagonally is not slightly wrong, it is fighting
  itself, so the distribution may be genuinely bimodal.
- Is a task where the median model scores below zero still useful? It ranks the top of the
  field fine. Arguably yes, and the aggregate is already protected by the per-task floor.

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

## Distribution — getting this in front of people

A reframe first, because it decides everything below: **RTCbench is not a heavy-compute
benchmark.** The whole suite runs on a laptop in minutes — that is a designed property, not
an accident. What costs money is *generating* a submission (API tokens); *verifying* one is
a few thousand cheap simulations.

That inverts the usual problem. MLPerf and SWE-bench need central evaluation because
verification is expensive; here anyone can verify anyone. RTCbench can be self-serve in a way
most benchmarks cannot, and the plan should exploit that rather than copying a pipeline built
for a different cost structure.

In the order worth doing them:

**1. A Zenodo DOI per release.** GitHub-integrated, one webhook. It is what makes the project
citable, and it costs an afternoon at most.

**2. An HF Dataset for run records.** This solves a problem the repo already has. Run records
carry full `(t, y, r, u)` traces and are what make a score falsifiable, but 200 episodes x 20
seeds x 600 steps would bloat `git clone` badly. A versioned Dataset repo (parquet, with the
built-in viewer) keeps the falsifiability promise without punishing everyone who checks out
the code. `leaderboard/` keeps the small summary JSON.

**3. An HF Space for the leaderboard.** Discoverability. It is where people look for LLM
benchmarks, and free CPU is plenty because the Space renders published results rather than
computing anything.

**Do NOT run submissions inside a Space.** Executing arbitrary submitted Python on hosted
infrastructure is precisely the threat model `rtcbench.sandbox` explicitly does not cover.
Keep the Space read-only over published results.

**4. A container image pinned per task-pack version.** "Reproducible on a laptop" is currently
true and undefended — nothing stops a numpy release changing a trajectory in the twelfth
decimal and silently re-denominating every published score. A pinned image makes the
reproducibility claim testable instead of aspirational, and it is also the honest answer to
the sandbox's filesystem hole.

**5. Submission by pull request, scored in CI.** The piece that fits this project uniquely
well: a PR adds `submissions/<name>/controller.py`, Actions runs the *dev* set in the sandbox
and posts the score on the PR, a maintainer runs the sealed set. Cheap verification is what
makes that possible at all, which is what turns the sandbox from a nicety into load-bearing
infrastructure.

A paper is the obvious eventual step. Note that PapersWithCode, the traditional channel, was
sunset — confirm its status before planning around it.

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
