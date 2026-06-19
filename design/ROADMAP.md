# Roadmap & Milestones

Phased build order with acceptance gates. Each phase produces something runnable
and testable. The emulator and exporter come first because every later phase
depends on them and on the preview loop. Validation uses `samples/short/*` for
fast iteration and `samples/long/*` for full‑track gates.

---

## Phase 0 — Project skeleton

- Repo layout from [ARCHITECTURE.md](ARCHITECTURE.md); `cli.py` stub with
  `convert`/`preview`/`validate`/`info` commands wired to no‑ops.
- Audio I/O: decode `wav/mp3/ogg/flac` to mono PCM; encode `wav/mp3` (ffmpeg).
- Config/profiles/seed plumbing.

**Gate:** `audio2ay info` and audio decode/encode round‑trip work on all samples.

---

## Phase 1 — AY emulator + exporter (the sacred core)

- `ay/registers.py`, `ay/emulator.py`, `ay/renderer.py`, `ay/timing.py`.
- All 16 envelope shapes, 17‑bit LFSR noise, logarithmic volume LUT, mixer,
  50 Hz latching with 44.1/48 kHz render.
- `export/` raw + PSG + YM; `preview/render.py`.

**Gate:** hand‑authored register streams (e.g. a scale, a noise drum, an envelope
buzz) render correctly; output cross‑checks against a reference YM/PSG player;
`AYState↔registers` and export↔import are lossless. **Preview mode works** end to
end on a hand‑made stream.

---

## Phase 2 — Analysis front‑end

- `analysis/*`: `FrameFeatures` at 50 Hz with overlap; loudness, chroma, onsets,
  centroid, salience, tempo/beat.

**Gate:** features are stable and visually sensible (plots) on `samples/short/*`;
pitch salience tracks the obvious melody in `01_arpeggio_mono.wav` and
`trumpet.ogg`.

---

## Phase 3 — Candidate generator + similarity

- `candidate/generator.py` + `heuristics.py`: features → plausible `AYState`.
- `similarity/loss.py` with mel/STFT + loudness + chroma + transient terms;
  profiles. Begin [LOSS.md](LOSS.md) — the loss is the project; document and
  tune it here.

**Gate:** generator output rendered through Phase‑1 emulator is *recognizable*
on simple samples (`01`, `02`); loss orders candidates sensibly (better‑sounding
state ⇒ lower loss) in unit tests.

---

## Phase 3.5 — Proposal diversity (multi‑start)

- Extend `candidate/generator.py` to emit **N diverse proposals** per window
  (e.g. melody‑led, bass‑led, arpeggio, noise‑heavy) rather than one warm start.
- `optimizer/multistart.py`: optimize from all proposals, keep the best basin.
- Diagnostics: measure which proposal "wins" per window.

**Gate:** on `samples/short/*`, the best‑of‑N warm start reaches lower loss than
any single fixed proposal; basin‑selection stats show real diversity (no single
proposal always wins). Finding a good basin is the goal here, not polish.

---

## Phase 4 — Optimizer (interfaces + coarse‑to‑fine)

- `optimizer/interfaces.py`: `MoveGenerator` / `Evaluator` / `SearchStrategy`
  ABCs. `moves.py`, `evaluator.py`, and `strategies/local_search.py`
  (+ `simulated_annealing.py`) as first concrete implementations.
- `optimizer/pyramid.py`: coarse‑to‑fine (160→80→40→20 ms) search.
- Render cache + incremental re‑render for speed.

**Gate:** optimizing from the candidate start measurably lowers perceptual loss
vs. the raw guess on `samples/short/*`; coarse‑to‑fine reaches equal/lower loss
than flat 20 ms search and escapes obvious local minima; preview is clearly
closer to source than Phase‑3.5 output.

---

## Phase 5 — Temporal optimization

- `optimizer/temporal.py`: windowed joint optimization + smoothness penalty.
- `strategies/beam_search.py` / `genetic.py` as alternative `SearchStrategy`
  plugins (no changes to moves/evaluator needed — proves the interface split).

**Gate:** audible reduction in register thrashing/jitter; smoother sustained
notes and stable channel assignments; loss not worse than Phase 4, continuity
metric improved.

---

## Phase 6 — Instrument behaviors (a prior, not hardware knowledge)

- `candidate/behaviors.py`: attack/decay/wobble/brightness/ornament models that
  **bias the proposals** and constrain temporal evolution. The optimizer never
  learns "what a piano is" — behaviors live entirely in the proposal stage.

**Gate:** lead/bass/percussion in `02_bass_and_lead.wav` and `03_drum_loop.wav`
get characteristic AY treatment (e.g. ornaments on lead, envelope bass, noise
drums) and score better than Phase‑5 baseline.

---

## Phase 7 — Full‑track scaling & validation report

- Streaming chunked processing for `samples/long/*`; parallel windows (`-j`).
- `report/validate.py`: original‑vs‑AY audio, per‑frame loss plot, overall score.
- `report/history.py`: **optimization history** (iteration vs loss per window) to
  reveal stagnation.
- `report/landscape.py`: **loss‑landscape sweeps** — perturb a register field by
  ±1 and plot loss, to see whether the search surface is smooth or jagged.
- Two‑pass (fast draft → annealing polish).

**Gate:** all `samples/long/*` convert within a reasonable time budget and
produce auditionable previews + reports; history plots show convergence (not
stagnation) on the short samples.

---

## Phase 8 — Learned proposals & move selection (optional)

- `ml/proposal_network.py` + `training.py`: bootstrap training data from Phase
  4–6 outputs and synthetic AY renders; network warm‑starts the optimizer.
- Learned **move selection**: a network scores which legal moves are promising,
  pruning the search — a much easier target than predicting registers directly.
  Implemented as another `SearchStrategy`, so nothing AY‑specific changes.

**Gate:** network‑warm‑started optimization reaches equal/lower loss in fewer
iterations than heuristic warm start; learned move‑selection cuts evaluations
per window without raising loss.

---

## Phase 9 — Perceptual polish & packaging

- Stage‑9 micro‑detune search; `--chip ym2149` support; compression.
- Docs, examples, packaged CLI.

**Gate:** final previews on the long samples are convincingly the source tracks
within AY limits; deterministic, documented, installable.

---

## Risk register

| Risk | Mitigation |
|---|---|
| Emulator inaccuracy poisons the whole loop | Build/validate it first vs. reference players (Phase 1 gate) |
| Search space too large / slow | Diverse warm starts, render cache, incremental moves, native backend, parallelism |
| Stuck in bad local minima | Multi‑start (best basin) + coarse‑to‑fine pyramid; landscape diagnostics |
| Perceptual loss doesn't match human hearing | Dedicated [LOSS.md](LOSS.md); tune weighted terms; A/B previews; psychoacoustic/masking term |
| Committing to one search algorithm | Move/Evaluator/Strategy interface split — strategies are interchangeable plugins |
| 3‑voice limit can't represent dense mixes | Stem separation + salience‑driven resource allocation; accept graceful degradation |
| Overfitting to short samples | Gate on long samples; regression tracking |

## Tracking

- Use `samples/short/*` as the CI/regression set (fast).
- Record per‑sample perceptual loss each phase to confirm monotone improvement.
- Keep a pure‑Python emulator reference alongside the native backend for tests.
