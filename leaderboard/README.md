# Results

**Current: `aggregate_2026-08-25.json`** (3 repeats; earlier `matrix_*.json` are single runs) — 20 models x 10 tasks, every task hash pinned and
verified current by `python experiments/check_stale.py`. The first results file in this repo
that can prove which task versions produced it.

Earlier files are kept, not deleted. `matrix_2026-08-21` and `matrix_2026-08-22` predate hash
recording and are marked UNPINNED by the staleness check: their numbers describe tasks of
unknown vintage, which is exactly why the pinning exists now.


**[report.html](report.html) is the readable view** of the run below — a ranked suite score
and a model x plant heatmap. Regenerate it after any new scoring pass with:

```bash
python experiments/report.py --matrix leaderboard/matrix_<date>.json --out leaderboard/report.html
```

Read the heatmap by column rather than by row: a plant where most of the field is red is a
plant that discriminates. That is more useful than the ranking.

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

### Superseded task

**The `shell_fractionator` column measured `shell_fractionator_v1`, which has since been
superseded by `v2`.** v1's anchor gap was only 32% of its hold cost -- control bought too
little for the 0-to-1 scale to rank with -- so eight of twenty models bottomed out at the
score clamp on a task that was under-specified rather than hard.

v1 remains on disk byte-for-byte as published, deprecated and failing validation on purpose.
It was NOT edited: a published task file is immutable, and rewriting one silently changes
what every historical score meant. Do not compare that column against any future run.

### Caveats recorded at the time

* **The two Claude models ran in a separate later batch.** Same tasks, harness, seeds,
  prompts and round count, but not the same wall-clock run — they were lost from the first
  batch by a `temperature` parameter bug and re-run after it was fixed.
* **Opus 5 is absent**, not scored zero: Bedrock returns AccessDeniedException for it on
  this account. It has no row rather than a misleading one.
* Scores are floored at -1.0 per scenario. Before flooring, the ranking was substantially an
  artifact of anchor width — Mistral ranked last on an unfloored scale and 7th on a floored
  one. See `src/rtcbench/score.py`.
