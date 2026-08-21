# CLAUDE.md

The guidance for coding agents in this repo lives in **[AGENTS.md](AGENTS.md)** — read it
before making changes. It is a single file rather than two so the two cannot drift apart and
start contradicting each other.

It covers the layout, the commands, the invariants that must not break (determinism, the
sealed observation boundary, task immutability, the reference anchor), and the specific
mistakes that have already cost this project time.

The short version, if you read nothing else:

- **Never edit a published task file.** Its content hash is in every run record. Cut a `v2`.
- **Never widen `Observation`.** A controller sees measurements, never state.
- **Core keeps two dependencies**, numpy and pyyaml.
- **Run `python -m pytest tests/ -q` and `rtcbench validate --task <changed task>`** before
  claiming anything works.
- **Commit with plain `git commit`** — the repo-local identity is already configured.
