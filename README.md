# RTCbench

**A benchmark for LLMs that commission real closed-loop control systems.**

Not a control-theory quiz. Nothing here is graded by reading an answer — a submission is a
controller, it gets connected to a dynamic process simulation it has never seen, and it is
judged on what the plant actually does.

```bash
pip install -e .
rtcbench score --task tasks/four_tank_v1.yaml --controller examples/pi_controller.py
```

```
four_tank_v1 [mismatch]  20 scenarios  hash=67a65694b68586e5

  anchors   hold cost 0.10540   reference cost 0.00389
  submission          0.00479

  seed        score   IAE      TV       flags
  11         +0.993  0.00541  0.00270
  23         +0.987  0.00451  0.00260
  37         +0.990  0.00429  0.00262
  ...
  four_tank_v1: CVaR@10%=+0.986  mean=+0.991  worst=+0.985  gated=0/20
  (0.0 = actuators held, 1.0 = reference controller, higher is better)
```

No GPU. No Windows. No COM. Two dependencies. The whole run above takes about fifteen
seconds on a laptop, and that is a design constraint, not an accident — see
[DESIGN.md](DESIGN.md) §9.

---

## What makes a score here mean something

**Anchored scoring.** Every scenario is measured between two controllers you can inspect
and re-run: `0.0` is freezing the actuators, `1.0` is the maintainers' well-tuned reference,
published gains and all. Scores above 1.0 are the point. Because the *unit of measurement*
is a properly tuned baseline, there is no way to look good by beating a weak one.

The reference is held to that standard in the test suite. The first draft of
`four_tank_v1` shipped hand-picked gains that the naive example submission beat — which
silently inflates every score on the task — so the gains were retuned by coordinate
descent, the provenance was written into the task file, and
`test_the_reference_is_not_beaten_by_a_naive_pi` now fails the build if it regresses.

**Safety is a gate, not a penalty term.** Any hard-constraint violation zeroes the scenario.
Real plants do not trade an overflow against a bit of integral error.

**Ranked on the bad tail.** The headline number is CVaR@10% across the scenario ensemble,
not the mean. Ranking on the mean rewards a controller that is excellent on most parameter
draws and unsafe on a few — exactly the controller nobody wants commissioned.

**Tasks are validated before anyone is scored on them.**

```bash
rtcbench validate --task tasks/four_tank_v1.yaml
```

checks that doing nothing does not already violate, that the reference is safe on every
draw, and that the anchors are properly separated. A scenario no controller can pass is
noise in the ensemble, and this catches it. It has already caught two real defects in this
repo's own task file.

**Every run is falsifiable.** `rtcbench replay` recomputes a record's metrics from its own
stored trace, so a published score does not have to be taken on faith, and every run can be
rendered as a PV/SP/OP trend — the view where a controller that "wins" by chattering the
valve is obvious in three seconds.

---

## The task, and why it is hard

`four_tank_v1` is Johansson's quadruple-tank process. Two pumps, four tanks, each pump
reaching each measured tank by two paths with different time constants.

- **Only the two lower tanks are measured.** The upper tanks are unmeasured and still
  overflow at 20 cm. You must not overflow a tank you cannot see.
- **The loop gain falls with level**, because outflow is gravity through a fixed orifice.
- **The parameters are not the ones you are given.** On the `mismatch` tier the brief
  publishes the *nominal* model; the plant runs a draw from a published distribution.
- **The instruments are real**: measurement noise, a scan of transport delay,
  quantization, dropped samples, pump slew limits, and valve stiction.
- **The plant gets kicked.** A leak opens on tank 1 at t=700 s, after the loops have settled.

And one parameter — the splitter valve setting — moves a transmission zero across the
imaginary axis, which is what the non-minimum-phase sibling task will use to make the
diagonal pairing the *wrong* answer.

---

## Writing a submission

Expose a class named `Controller`. That is the entire interface.

```python
class Controller:
    def __init__(self, brief): ...          # tags, units, ranges, limits, T_s, safety envelope
    def reset(self): ...
    def step(self, t, y, r, quality):       # -> control vector
        ...
```

Start from [`examples/pi_controller.py`](examples/pi_controller.py), which is deliberately
competent-but-unremarkable: it scores 0.986, just under the reference, because it never
identifies the plant.

You get measurements and their quality flags. You never get the state vector — that
omission is what keeps this a benchmark about control rather than about reading a state.

```bash
rtcbench show  --task tasks/four_tank_v1.yaml     # the operating manual you'd be handed
rtcbench run   --task tasks/four_tank_v1.yaml --controller mine.py --out out/
rtcbench score --task tasks/four_tank_v1.yaml --controller mine.py --trends out/
```

---

## Status

v0.1. The harness, scoring, records, replay, trends and one validated task are working and
tested (`pytest` — 36 tests, 8 seconds).

Honest about what is not built yet:

- **No sandbox.** `rtcbench.submission` imports a controller with the harness's own
  privileges. Fine for running your own controllers; not fine for accepting submissions
  from strangers. Subprocess isolation is v0.2.
- **One plant.** Eight more are specified in [DESIGN.md](DESIGN.md) §4, each chosen to break
  a different naive controller.
- **No commissioning CLI yet.** The design calls for a metered phase where an agent
  bump-tests a training replica before submitting; today you write the controller directly.
- **No leaderboard, no sealed test set.** The salt-derived seed machinery exists; the
  governance around it does not.

---

## License and governance

MIT for code, CC-BY-4.0 intended for tasks and results. DCO sign-off, not a CLA.

RTCbench originated at [Acaysia](https://github.com/AcaysiaChem) and is intended to be
governed independently. Core has **zero** Acaysia dependencies — not in `pyproject.toml`,
not in CI, not in the tests — and it stays that way. Proprietary and GPL engines (AcaysiaRT,
DWSIM/CAPE-OPEN) plug in behind the same `Plant` protocol as everything else, out of tree,
and cannot host scored tasks for the same reason in both cases: a stranger cannot reproduce
them. See [DESIGN.md](DESIGN.md) §1–2 and §7.
