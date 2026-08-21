# Results

Each file is one scoring pass over the whole suite, kept verbatim. Never edited after the
fact: a leaderboard you can quietly revise is not evidence.

## matrix_2026-08-21.json — first full suite run

14 models, 8 tasks, 20 scenarios each, all commissioned through the identical Bedrock
harness (`experiments/commission.py`, 3 rounds, feedback on held-out seeds 1001-1005).

Headline: **claude-fable-5, suite +0.572** — double the next model, and the only entry with
no floored task anywhere.

What the run established about the tasks themselves, which matters more than the ranking:

* **Only 3 of 112 submissions beat their task's reference.** Haiku (+1.034) and Sonnet 5
  (+1.027) on Column A, Fable (+1.123) on deadtime. There is enormous headroom in this
  suite.
* **Column A is the one pathology models handle well** — nobody hit the floor on it.
  Ill-conditioning is apparently easier for a language model to reason about than
  non-minimum phase.
* **four_tank_nmp and van_de_vusse each floored 8 of 14 models**, and nobody beat the
  reference on the NMP task at all. Comparing each model's four_tank column against its
  four_tank_nmp column isolates the wrong-pairing trap on identical hardware: Mistral
  +0.920 -> -1.000, Sonnet +0.974 -> -1.000.

### Caveats recorded at the time

* **The two Claude models ran in a separate later batch.** Same tasks, harness, seeds,
  prompts and round count, but not the same wall-clock run — they were lost from the first
  batch by a `temperature` parameter bug and re-run after it was fixed.
* **Opus 5 is absent**, not scored zero: Bedrock returns AccessDeniedException for it on
  this account. It has no row rather than a misleading one.
* Scores are floored at -1.0 per scenario. Before flooring, the ranking was substantially an
  artifact of anchor width — Mistral ranked last on an unfloored scale and 7th on a floored
  one. See `src/rtcbench/score.py`.
