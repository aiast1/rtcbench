# RTCbench — design notes

**What it is:** a benchmark where LLMs compete to commission real closed-loop control
systems against dynamic process simulations, scored on closed-loop performance rather
than on text.

**What it is not:** a control-theory quiz. A text benchmark named *ControlBench*
(Kevian et al., 2024) already exists, built from textbook problems — hence the distinct
name here. This is the opposite end of the spectrum: nothing is graded by reading an
answer, everything is graded by running a plant.

---

## 0. The one-line thesis

> A model's control ability is what its controller does to a plant it has never seen,
> under a mismatch it was not told about, judged against a well-tuned human baseline.

Every decision below follows from that sentence.

---

## 1. The licensing constraint (decided)

RTCbench core is **MIT** and must be installable and fully reproducible by a stranger
with a laptop — no GPU, no Windows, no COM, no proprietary engine.

That rules out making any closed engine the default. It also rules out linking GPL engines
into the core.

| Layer | License | Ships where |
|---|---|---|
| `rtcbench` (protocol, harness, sandbox, metrics, scoring, CLI) | MIT | core repo |
| `rtcbench-plants` (pure-numpy reference plants) | MIT | core repo |
| `rtcbench-fmi` (any FMU as a plant, via FMPy) | MIT (FMPy is BSD-2) | core repo, extra |
| `rtcbench-opcua` (hardware / soft-PLC in the loop) | MIT | core repo, extra |
| `rtcbench-dwsim` (DWSIM / CAPE-OPEN) | **GPL-3.0** | **separate repo, out of tree** |
| `rtcbench-acaysiart` (AcaysiaRT as a plant backend) | proprietary | Acaysia's own repos, never vendored here |

**DWSIM must be out-of-tree.** DWSIM is GPLv3; an adapter that links it is a derivative
work and would drag the core into GPL if distributed together. It is also
Windows/.NET/COM-bound and steady-state-first, so it can never host a *scored* task — the
leaderboard requires deterministic, cross-platform, dynamic simulation. DWSIM belongs in a
"showcase" track, not the scored suite.

**FMI is the real escape hatch.** Speaking FMI 2.0/3.0 Co-Simulation lets OpenModelica,
Dymola, Simulink and friends host plants as `.fmu` binaries distributed separately from
this repo — interop with zero license entanglement, because we never link their source.

---

## 2. Engine independence

Core depends on exactly one thing: a protocol.

```python
class Plant(Protocol):
    spec: PlantSpec                                  # tags, units, engineering ranges,
                                                     # actuator limits, sample time T_s
    def reset(self, seed: int) -> Observation: ...
    def step(self, u: NDArray) -> Observation: ...   # advance exactly one control period
    def close(self) -> None: ...
```

`Observation` carries `(t, y_measured, quality_flags)` and **nothing else**. The plant's
internal state vector is never handed out. That single omission is what makes the
benchmark about control rather than about reading a state.

Everything else — AcaysiaRT, an FMU, DWSIM, an OPC UA endpoint, a physical rig — is an
adapter behind that protocol. Adapters are versioned separately and never gate the core
test suite.

### AcaysiaRT's role (decided)

**RT is a plant backend, exactly like a CAPE-OPEN modeler.** It sits in the adapter tier
alongside FMI, DWSIM and OPC UA — one more way to host a plant — and nothing about it is
privileged in the benchmark's design.

Two consequences worth stating explicitly, because they are what keeps this clean:

- **RT's MPPI is not a competitor and is never scored.** It has full access to the exact
  model it is controlling, so a leaderboard row for it would be an oracle beating a field
  of blind agents — a meaningless comparison that would read as the affiliated party
  winning its own benchmark. Where a strong upper reference is genuinely useful, it is
  labeled an *oracle anchor*, printed as context, and excluded from ranking.
- **RT-hosted tasks cannot carry the scored suite.** Same reason DWSIM cannot: a stranger
  can't reproduce them. Closed engine, GPL engine and COM-bound engine all land in the
  same bucket — *extended backends* — for the same reproducibility reason, not for
  licensing reasons. The scored suite stays on the MIT plant pack.

What RT is genuinely good for here: authoring and cross-checking plants. A model validated
against independent physics oracles is a strong reference for deciding whether an open
reimplementation is faithful, and that use costs the project no reproducibility at all.

---

## 3. What a competitor actually submits

**The submission is a sealed controller artifact, not a chat transcript.**

```
submission/
  controller.py        # implements Controller
  params.json          # tuned gains, identified models, whatever it learned
  NOTES.md             # the agent's own account of what it did
  manifest.json        # deps, hashes, budget consumed
```

```python
class Controller(Protocol):
    def __init__(self, brief: TaskBrief): ...          # the operating manual, not the ODEs
    def reset(self) -> None: ...
    def step(self, t: float, y: NDArray, r: NDArray) -> NDArray: ...   # -> u
```

Why an artifact rather than a live agent-in-the-loop:

- **Re-runnable forever.** Any submission can be re-scored on task packs written years
  later. The leaderboard becomes a matrix (submission x task), not a frozen run.
- **Deterministic and cheap.** Scoring 200 scenarios costs sim time, not tokens.
- **It matches the real job.** An engineer commissions a loop once; the loop then runs
  unattended for a year. Grading the artifact grades the right thing.

### The commissioning phase is where the agent gets to be an agent

```
rtcbench commission --task four_tank_v1 --agent <scaffold>  ->  submission/
```

During commissioning the agent gets a **training replica**: same topology and brief,
*different* seed set and *different* parameter draw than eval. It may bump-test,
step-test, run relay experiments, fit models, try controllers, look at trends. All of it
is metered:

- simulated seconds of plant time
- number of episodes
- wall-clock and tokens

Then eval runs sealed, against a plant it has never touched. That is system identification
plus control design under uncertainty — the actual skill — instead of PID-tuning trivia.

---

## 4. The difficulty ladder (the scientific core)

The interesting axis is **how much the controller is told**.

| Tier | The controller receives | Tests |
|---|---|---|
| **T0 Nominal** | the exact plant model | can you tune at all |
| **T1 Mismatch** | the *nominal* model; plant params drawn from a published distribution | robustness |
| **T2 Blind** | only the brief — tag list, units, ranges, actuator limits, `T_s`, safety envelope | sysID + design. **the flagship** |

Layered per scenario: measurement noise, sensor lag and deadtime, actuator saturation
*and* rate limits, valve stiction/hysteresis, quantization, sample dropout, exogenous
disturbance sequences, setpoint ramps and grade changes, and a hard safety envelope.

### Reference plant pack — all reimplementable from published equations

Chosen so each one breaks a *different* naive controller:

| Plant | The trap |
|---|---|
| Four-tank (Johansson 2000) | valve split slides it from minimum- to non-minimum-phase; RHP zero |
| Van de Vusse CSTR | input multiplicity, sign-flipping gain |
| Exothermic CSTR (Hicks–Ray) | open-loop unstable, ignition/extinction |
| pH neutralization (Hall & Seborg) | gain varies by orders of magnitude across the range |
| Column A (Skogestad, 41-tray LV) | ill-conditioned MIMO, large RGA, directionality |
| Shell heavy oil fractionator | the classic constrained-MPC problem |
| Boiler drum level | shrink-and-swell inverse response |
| FOPDT with variable deadtime | deadtime dominance |
| Tennessee Eastman (Downs & Vogel 1993) | plantwide, recycle, safety trips, economics |

A controller that scores well across all nine is not pattern-matching a PID recipe.

---

## 5. Scoring — where benchmarks usually die

**Never publish a raw IAE leaderboard.** Anchor every scenario between two references:

```
S_scenario = (J_hold - J_submission) / (J_hold - J_reference)
                 0.0 = do nothing        1.0 = maintainer's well-tuned controller
```

Scores above 1.0 are possible and are the whole point. Anchoring makes cross-plant
aggregation meaningful and makes a strawman baseline structurally impossible: since the unit
of measurement *is* a well-tuned controller, there is no way to look good by picking a weak
comparison. A strawman baseline does not merely weaken a result, it invalidates it, and
anchoring forecloses that structurally rather than leaving it to reviewer vigilance.

`J` combines tracking (IAE over the setpoint profile) and control effort (total variation
of `u`), both normalized, with published weights frozen per task version.

Three rules that matter more than the formula:

1. **Safety is a gate, not a term.** Any hard-constraint violation zeroes the scenario and
   is reported separately. Real plants do not trade overpressure against IAE.
2. **Headline metric is CVaR@10%, not the mean.** Mean-scoring rewards gamblers. Report
   the mean too, but rank on the bad tail.
3. **Cost goes on the leaderboard.** Score *and* commissioning budget (tokens,
   sim-seconds, wall-clock). It is a Pareto frontier, or it degenerates into whoever
   spent the most.

---

## 6. Integrity

- **Process isolation.** The controller runs in a subprocess with no network, no plant
  imports, and a per-step wall-clock budget. Enforced by the harness, not by honor.
- **Overruns are realistic, not fatal.** Blowing the step budget holds the last `u` and
  logs a scan overrun, exactly like a DCS. Too many overruns fails the run.
- **Public dev set, sealed test set.** Task YAML is public; the test-set seed salt is not.
  Maintainers run the sealed set. Rotate on a published schedule.
- **Canary tasks** whose known-optimal answer is a trap, to detect memorization.
- **Every run emits a replayable record**: engine version, task hash, seed, RNG scheme,
  full `(t, y, r, u)` trace. `rtcbench replay <record>` reproduces it bit-exactly.
- **Every leaderboard row links to a trend plot** (PV/SP/OP). Any control engineer can
  then eyeball in three seconds whether the "winner" is sane or a bang-bang chatterbox.
  Cheapest credibility feature in the whole project.
- **Human baseline.** Two or three real control engineers submitting under the same
  budget. A benchmark with a human reference line is worth ten times one without.

---

## 7. Governance, given the Acaysia affiliation

- **Not under the Acaysia org.** Currently `github.com/aiast1/rtcbench` — a personal
  account rather than the neutral org this section originally called for, because the
  `rtcbench` org name is already taken by an unrelated party. A personal account is the
  weaker arrangement of the two and is meant to be temporary: GitHub transfers a repo to an
  org without losing stars, issues or history, so this should move once a maintainer group
  exists and a name is settled. `NOTICE` carries the standing claim either way:
  *originated at Acaysia, governed independently.*
- **MIT** for code, **CC-BY-4.0** for tasks and results.
- **DCO sign-off, not a CLA.** CLAs deter contributors; DCO is sufficient.
- **Written conflict-of-interest rule, published on day one:** Acaysia-affiliated
  submissions are labeled on the leaderboard, and no maintainer scores their own
  submission on the sealed set without a second maintainer countersigning.
- **Zero Acaysia dependencies in core.** Not in `pyproject.toml`, not in CI, not in the
  test suite. The private stack stays entirely behind the adapter boundary.

Pre-committing to the COI rule before there is anything to be accused of is what makes the
affiliation a non-issue for the rest of the project's life.

---

## 8. Repo shape

```
rtcbench/
  README.md  LICENSE (MIT)  NOTICE
  GOVERNANCE.md  CONTRIBUTING.md  CODE_OF_CONDUCT.md
  RULES.md                    # the benchmark contract: what is allowed, what is scored
  src/rtcbench/
    plant.py                  # Plant protocol — the ONLY engine dependency
    controller.py             # Controller protocol
    task.py                   # Task/Scenario spec, YAML-loadable, content-hashed
    harness.py                # sealed loop: sensors -> controller subprocess -> actuators
    sandbox.py                # isolation, budgets, import guard
    metrics.py                # IAE, TV, violations, CVaR
    score.py                  # anchored normalization
    record.py                 # run record + replay
    baselines/                # hold, tuned PID, IMC, relay autotune, MPC oracle
    cli.py
  plants/                     # MIT reference plant pack
  adapters/                   # fmi/, opcua/   (dwsim lives in its own repo)
  tasks/                      # task packs: YAML + sealed-seed manifest
  leaderboard/                # committed results JSON + provenance
```

**Tasks are data, not code.** A task is YAML: engine ref, model id, parameter
distribution, sensor/actuator config, disturbance schedule, setpoint profile, constraints,
budgets, and the brief handed to the controller. Content-hashed and versioned
(`four_tank_v1`). A published task is **never mutated** — you cut `v2`.

---

## 9. The 60-second test

The project lives or dies on this working, on a laptop, with no GPU:

```bash
pip install rtcbench
rtcbench run --task four_tank_v1 --controller examples/pid.py
```

If that is ever slower than a minute or needs a binary blob, the benchmark is dead on
arrival regardless of how good the physics is.

---

## 10. Open questions

- **A neutral home.** The `rtcbench` GitHub org is taken by an unrelated party, so the
  project sits on a personal account for now (§7). Needs a name and a maintainer group
  before it can move.
- **What RTC expands to.** Assumed "real-time control"; never actually confirmed, and it
  belongs in the README once it is.
- Tennessee Eastman provenance — confirm a cleanly-licensed reimplementation path.
- Whether **T2 Blind** is the headline track from v0.1 or arrives at v0.2.
