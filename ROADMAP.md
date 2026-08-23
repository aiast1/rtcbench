# Roadmap

**Where things stand (v0.2):** ten validated plants, 186 tests, anchored scoring with safety
and actuator-duty gates, a submission sandbox, replayable run records, and two complete
multi-model runs — most recently 20 models x 10 tasks through an identical harness. See
[leaderboard/](leaderboard/).

---

## The arc

RTCbench is trying to become the thing you point at to answer *"can this model be trusted to
commission a control loop?"* — with an answer a control engineer would accept and a skeptic
could re-derive.

Three properties have to hold at once, and they fail independently:

1. **The plants must be right.** A benchmark built on a plant with a sign error measures
   nothing, confidently.
2. **The scale must be right.** An anchor that is the wrong controller class, or a gap too
   narrow to rank with, produces numbers that look like measurements and are not.
3. **The submissions must be honest.** Code that can read the answer key, or that is scored
   on the scenarios it was tuned against, produces a leaderboard about the harness rather
   than about control.

Milestones are ordered by which of those is currently weakest, not by what is most fun to
build. Coarse on purpose — the further out, the less the detail is worth writing down now.

---

## M1 — Trustworthy plants *(current)*

**The problem:** every plant is verified by tests written alongside it by the same author.
That is self-consistency, not correctness. A sign error or a units slip would pass all 186
tests, and the tests would faithfully confirm the wrong model is stationary at its own wrong
steady state. If that is true of any plant, everything published about it is wrong.

- Independent oracles: a dev-only `validators/` tree integrating the same system in a second
  implementation and comparing trajectories. Cantera (BSD, pip-installable) covers the
  reacting plants; FMUs or published reference trajectories cover the rest.
- A fidelity ladder per plant, carried as data and surfaced in the report — verified against
  an independent oracle / quantified gap / self-consistent only. Nobody should have to read a
  docstring to learn how much a plant has been checked.
- CI that fails a plant whose oracle comparison drifts.

**Done looks like:** every plant states what checked it and how far off it was, and nothing in
the scored suite rests on self-consistency alone.

## M1.5 — Error bars *(measured, and it is the biggest gap)*

**Every published number is one commissioning run.** Re-running three models three times
each:

    fable-5           +0.553 +0.620 +0.513   spread 0.107
    grok-4.6          +0.034 +0.227 +0.299   spread 0.265
    nemotron-super    -0.420 -0.443 -0.036   spread 0.407

Ranks 1-6 on the published leaderboard span 0.297. Nemotron's spread alone exceeds that, and
grok's mean would move it four places. The top of the table is real; the middle is not
reliably ordered, and a single-run ranking should not be quoted as one.

The spread has structure worth keeping: the consistent model is the one that never crashes.
A submission that crashes on a task *sometimes* swings by that whole task's worth of score,
which is why reliability and variance are the same finding seen twice.

- Three repeats minimum for any published run; report mean and spread, not a point.
- The report and the README must show the interval, not just the rank.
- Consider whether the headline should be the mean or the worst repeat -- the same argument
  that made CVaR the per-task headline applies again one level up.

## M2 — Trustworthy scale

**The problem:** all ten anchors are banks of independent PI loops, and on at least one plant
that is demonstrably the wrong controller class.

- Reference classes beyond decentralized PID. **The obvious first attempt is already
  refuted:** a static decoupler + PI is implemented (`baselines/decoupler.py`) and is WORSE
  than a tuned PI bank on every plant tried — shell -23%, column_a -13%, four_tank_nmp -12%,
  with equal per-loop tuning effort. Column A, condition number 273, is the strongest case
  for decoupling in the pack and the decoupler still loses. The likely reason is that the
  decoupler inverts a *nominal* gain matrix while every task draws its parameters per
  scenario, and inverting a matrix you do not have exactly amplifies exactly the directions
  where the plant responds weakly. What remains is a genuine constrained MPC with deadtime
  compensation, which is a much larger piece of work than it looked.
- Anchor health published per task. `validate` already computes the gap-to-hold ratio; it
  belongs in the report next to every score.
- A human baseline. Two or three real control engineers under the same budget. A benchmark
  with a human reference line is worth several times one without.

**Done looks like:** no task carries a PROVISIONAL banner, and every anchor is the best of a
class appropriate to its plant. `shell_fractionator` stays provisional until then; the cheap
fix for it does not exist.

## M3 — Trustworthy submissions

- Container isolation, the only honest answer to arbitrary submitted code. The Python sandbox
  is an honest-mistake barrier and says so in its own docstring.
- The reference configuration out of the competitor-readable task file.
- A sealed test set with published governance: who holds the salt, the rotation schedule, how
  a submission is run against it.
- Contamination canaries — a task whose known-optimal answer is a trap.

**Done looks like:** a stranger's submission can be accepted, run and scored without anyone
having to trust it.

## M4 — Commissioning as the benchmark

**The problem:** today a model gets three rounds of cost feedback. The design calls for
something much closer to the real job — an agent that can *experiment* on a training replica
before committing to a design.

- `rtcbench commission` as a first-class command with a metered budget in simulated seconds,
  episodes, wall-clock and tokens.
- A tool surface for the agent: bump tests, step tests, relay experiments, trend inspection —
  what a commissioning engineer actually does.
- Cost reported beside score, because a result without its budget is half a result.

**Done looks like:** the benchmark measures system identification plus control design rather
than PID recall, and the leaderboard is a Pareto frontier rather than a ranking.

## M5 — Breadth

- Tennessee Eastman: plantwide, recycle, safety trips, economics. The obvious gap, pending a
  cleanly-licensed reimplementation path.
- Plantwide tasks generally — several units coupled with recycle, where the snowball effect is
  the trap.
- Batch and grade-change tasks. Everything in the pack is continuous and near a steady state;
  startup, shutdown and transition are where real plants hurt.
- Blind variants of plants where the model actually matters.

## M6 — Beyond simulation

- `rtcbench-fmi`, opening OpenModelica, Dymola and Simulink as plant hosts with no licence
  entanglement, because we never link their source.
- OPC UA, then hardware in the loop. Four-tank rigs exist in a dozen university labs and are
  the natural first physical target.
- A correlation study against a high-fidelity engine: do open-suite scores predict scores on
  plants nobody can reproduce? That is the honest use for a proprietary backend.

## M7 — The questions the benchmark exists to answer

Once the instrument is trustworthy, these become answerable rather than speculative:

- Does control ability transfer across plants, or is each column its own skill?
- Does giving a model the plant model help? *(Partially answered, and so far: no. On
  `four_tank_blind` the median effect of withholding it was -0.028.)*
- Does more inference compute buy control quality, and where does that curve bend?
- Can models do system identification at all, or are they pattern-matching tuning rules?
- What is the failure taxonomy? Wrong pairing, windup, ignoring deadtime, chattering — a named
  catalogue of how models fail is more useful to the field than a ranking.

## M8 — Ecosystem

- Public leaderboard, DOI, and eventually a paper.
- Third-party plants arriving through `docs/AUTHORING_PLANTS.md` without maintainer help.
- A maintainer group and a neutral home. The `rtcbench` GitHub org is taken by an unrelated
  party, so the project currently sits on a personal account.

---

## Near-term, concretely

The next few sessions' worth, inside M1–M3.

1. **Cantera oracle for the three reacting plants** (`ph_neutralization`, `van_de_vusse`,
   `unstable_cstr`). The highest-value item on the whole list: the only one whose downside is
   "everything measured so far is wrong".
2. ~~Static decoupler + PI reference class.~~ **Done and refuted** — see M2. It is worse
   than a tuned PI bank on all three multivariable plants. `shell_fractionator` stays
   provisional; un-provisionalling it needs a constrained MPC, not a gain-matrix inverse.
3. **Reference config out of the task file.** An hour, and it closes the exact hole that
   motivated the sandbox — an agent read the gains out of `tasks/four_tank_v1.yaml` and
   reported them as its own tuning.
4. **Sandbox wired into `experiments/commission.py`**, which still loads submissions
   in-process.
5. **Faster tuner for multi-loop plants.** Shell costs ~7000 ensemble evaluations because cost
   scales as loops x params x grid x rounds x seeds; the probe-seed trick already used for the
   coarse sweep applies to the descent too.

---

## Distribution — getting this in front of people

A reframe first, because it decides everything below: **RTCbench is not a heavy-compute
benchmark.** The whole suite runs on a laptop in minutes — a designed property, not an
accident. What costs money is *generating* a submission (API tokens); *verifying* one is a few
thousand cheap simulations.

That inverts the usual problem. MLPerf and SWE-bench need central evaluation because
verification is expensive; here anyone can verify anyone. The plan should exploit that rather
than copy a pipeline built for a different cost structure.

1. **A Zenodo DOI per release.** GitHub-integrated, one webhook, and what makes the project
   citable.
2. **An HF Dataset for run records.** Solves a problem the repo already has: records carry
   full `(t, y, r, u)` traces and are what make a score falsifiable, but they would bloat
   `git clone` badly. `leaderboard/` keeps the small summary JSON.
3. **An HF Space for the leaderboard.** Discoverability — free CPU is plenty, since it renders
   published results rather than computing anything.
4. **A container image pinned per task-pack version.** Nothing currently stops a numpy release
   changing a trajectory in the twelfth decimal and silently re-denominating every published
   score.
5. **Submission by pull request, scored in CI.**

**Do NOT run submissions inside a Space.** Executing arbitrary submitted Python on hosted
infrastructure is precisely the threat model `rtcbench.sandbox` does not cover.

PapersWithCode, the traditional channel, was sunset — confirm before planning around it.

---

## Open questions

- **Is the NMP cliff real?** `four_tank_nmp` has a -1.88 median: most of the field is worse
  than doing nothing. That may be genuine bimodality — a wrongly-paired controller is not
  slightly wrong, it is fighting itself — or an artefact worth correcting.
- **Is a task whose median model scores below zero still useful?** It ranks the top of the
  field fine, and the per-task floor already bounds the aggregate. Probably yes.
- **Is the blind tier testing what it is for?** On a four-tank a robustly tuned PI needs no
  model, so withholding one costs nothing. The tier only bites where system identification is
  genuinely required.
- **What does RTC expand to?** Assumed "real-time control"; never confirmed.
- **Should a repo-access track exist**, separately labelled? "What does an agent with a
  filesystem do" is a legitimate measurement — the answer so far is *it reads the answer key*
  — but it is a different benchmark and must never share a table with this one.

---

## Deliberately not doing

- **Rebuilding a thermodynamics or physical-properties engine.** No property database, no
  equation of state, no flash. Plants are small published ODEs with constants from their
  source papers. Anything needing real thermo belongs behind an adapter.
- **Vendoring a GPL or proprietary engine into the scored suite.** Not a licensing preference,
  a reproducibility one: a stranger with a laptop must be able to re-run any scored result,
  which rules out closed, COM-bound and GPL engines alike for the same reason.
- **A live agent-in-the-loop track as the main leaderboard.** Submissions are sealed artifacts
  so they can be re-scored on tasks written years later. A transcript cannot be.
- **Adding dependencies to core.** numpy and pyyaml. The plotting is hand-rolled SVG precisely
  so `pip install rtcbench` stays a two-package install.
