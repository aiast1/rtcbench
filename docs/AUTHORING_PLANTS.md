# Adding a plant to RTCbench

A plant contribution is **three files you own outright** plus nothing else:

```
src/rtcbench/plants/<plant_id>.py     the physics
tasks/<plant_id>_v1.yaml              the task
tests/test_<plant_id>.py              its tests
```

Do **not** edit `plants/__init__.py`, the harness, the scoring, another plant, or another
plant's task. The registry discovers modules; nothing needs registering by hand. This is
what lets several plants be authored at the same time without colliding.

Read [`src/rtcbench/plants/four_tank.py`](../src/rtcbench/plants/four_tank.py) first — it is
the worked example for everything below.

---

## 1. The physics module

Two module-level names make it discoverable:

```python
PLANT_ID = "your_plant"

def build(cfg):            # cfg is the task file's `plant:` block, minus `kind`
    return YourPlant(**cfg)
```

The class implements `rtcbench.plant.Plant`:

```python
spec: PlantSpec                    # tags, units, ranges, T_s, constraints, initial_u
def reset(self, seed: int) -> Observation
def step(self, u) -> Observation   # advance exactly one control period
def audit(self) -> Audit           # TRUE state + violated constraint names
def set_params(self, overrides) -> None   # mid-run disturbance injection
def close(self) -> None
```

### Non-negotiables

1. **Deterministic.** Same `(seed, control sequence)` must give a bit-identical trajectory.
   Use `np.random.default_rng((seed, <a constant unique to your plant>))`. Never the global
   `np.random`, never wall-clock, never a set/dict iteration order that can vary.
2. **Fixed-step integration.** RK4 with a fixed substep count. An adaptive solver makes the
   trajectory depend on floating-point accidents in its error controller and silently breaks
   every reproducibility claim in the project.
3. **`Observation` carries measurements only.** Never the state. If a signal is constrained
   but not instrumented, that is a *feature* — say so in the description.
4. **Do not clip your way out of a violation.** If a level exceeds its limit, the state must
   exceed the limit so `audit()` can report it. Clip only what physics clips (a tank cannot
   hold negative volume).
5. **Cite the source.** Module docstring names the paper and states which parameter set you
   used. Reimplement from published equations only.

### Make it break something specific

Every plant in the pack exists to defeat a *different* naive controller. State in your
docstring which one yours is: inverse response, deadtime dominance, input multiplicity,
open-loop instability, ill-conditioning, gain that varies by orders of magnitude. A plant
that a textbook PI handles comfortably adds nothing to the suite.

---

## 2. The task file

Copy the structure of [`tasks/four_tank_v1.yaml`](../tasks/four_tank_v1.yaml). Required:

```yaml
task_id: your_plant_v1
tier: mismatch                # nominal | mismatch | blind
description: >
  What the unit is, what the actuators physically do, what is and is not instrumented.
  This is the operating manual a commissioning engineer would be handed. NOT the equations.

plant:
  kind: your_plant            # must equal PLANT_ID
  sample_time: <T_s>
  mismatch:                   # per-parameter relative log-normal spread
    some_param: 0.03

horizon: <seconds>
controlled: [0]               # indices into measurements that carry setpoints

setpoints:
  - { t: 0, values: [...] }   # must include t=0
  - { t: 300, values: [...], ramp: 60 }    # include at least one step AND one ramp

disturbances:
  - { t: 700, params: { some_param: <new value> } }

instruments:
  sensors:   { noise_std: ..., deadtime_samples: ..., quantization: ..., dropout_prob: ... }
  actuators: { rate_limit: ..., deadband: ... }

scoring:
  w_error: 1.0
  w_effort: 0.5
  cvar_alpha: 0.10
  max_total_variation: <~4x your tuned reference's tv>   # actuator duty gate

budget: { step_seconds: 0.05, max_overruns: 20 }

scenarios:
  seeds: [ ...20 seeds... ]
```

The reference anchor does NOT go in this file. It lives in
`tasks/references/<plant_id>_v1.yaml` and is merged in automatically at load time, because
the task file is what a competitor is handed and a submission with filesystem access will
read it. `apply_reference.py` writes to the right place for you.

### Sizing the mismatch

Size it against the **initial condition**, not by taste. Run your parameter draws and check
where the plant *starts*. If any draw begins near a constraint, the scenario is unwinnable
before the controller does anything, and an unwinnable scenario is noise in the ensemble.
This is a real defect that shipped in `four_tank_v1`'s first draft — see the comments in
that file.

---

## 3. The reference anchor — the part that matters most

The reference **defines 1.0**. An under-tuned one inflates every score on your task forever.

Tune it with the shared tuner rather than by hand:

```bash
python experiments/tune_reference.py --task tasks/your_plant_v1.yaml --apply
```

Run it in the **foreground**. It takes a few minutes. Do not background it and wait for a
notification: every author in the first round did exactly that, ended their turn waiting,
got re-invoked, waited again, and two tasks shipped with placeholder gains as a result.

Then apply the result and record its provenance:

```bash
python experiments/apply_reference.py --task tasks/your_plant_v1.yaml --pairing 0 --kp 1.8 --ti 20 --cost 0.0039 --flatness "+/-25% on kp costs 1-3%" --max-tv 0.017
```

Three lessons from the first round of plants, each of which cost a task:

* **Tune on ALL 20 seeds if your plant has multiple interacting loops or per-element
  deadtimes.** The default 6-seed tuning left the Shell fractionator gated on one of the
  other 14: a gain set that is safe on a sample is not necessarily safe on the ensemble.
  Single-loop plants are fine with the default.
* **A correct integral time can be in the thousands of seconds.** If your plant has an RHP
  zero or dominant deadtime, expect `ti` in the hundreds or thousands and do not "fix" it.
  The boiler drum's optimum is `ti = 2000 s`; an earlier grid that stopped at 200 s reported
  that no safe tuning existed at all, which reads as "impossible task" but means "the grid
  did not contain the answer".
* **If a naive PI beats your reference, your reference is wrong, not the naive PI.** Retune
  before shipping. This inflates every score on your task, permanently.

Set `max_total_variation` to roughly 4x the tuned reference's `tv` — high enough not to
punish necessary control action, low enough to catch a controller holding setpoint by
hammering the valve.

---

## 4. Acceptance — all three must pass

```bash
python -m rtcbench.cli validate --task tasks/your_plant_v1.yaml    # must print PASS
python -m pytest tests/test_<plant_id>.py -q                        # your tests
python -m pytest tests/ -q                                          # nothing else broken
```

`validate` checks that doing nothing does not already violate, that the reference is safe on
every draw, and that the anchors are well separated. **It failing is the task's fault, not
the harness's.**

### What to test

Mirror `tests/test_core.py`'s style — properties, not coverage:

- determinism: same seed and controls twice, `assert_array_equal`
- a steady state that is actually stationary
- the specific pathology your plant exists to exercise (measure it — e.g. assert the step
  response goes the *wrong way* first for an inverse-response plant)
- constraints fire when they should
- your published parameters reproduce the source paper's stated operating point

---

## 5. Do not

- run `git commit`, `git push`, or `pip install`
- edit any file outside your three
- read or modify `experiments/agents/`
- weaken a constraint, a disturbance, or the instrument config to make `validate` pass —
  fix the task instead, and say in a comment what you changed and why
