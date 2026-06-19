# audio2ay — Design Plan

**Convert instrumental audio into General Instrument AY‑3‑8910 register streams via analysis‑by‑synthesis.**

*The AY chip is the language.*

> Find the sequence of AY register states at 50 Hz that, when synthesized by a
> cycle‑accurate AY emulator, minimizes the perceptual difference from the
> original instrumental recording.

This document is the top‑level plan. Companion documents:

- [AY_REFERENCE.md](AY_REFERENCE.md) — hardware, registers, timing, output formats.
- [ARCHITECTURE.md](ARCHITECTURE.md) — modules, data contracts, CLI surface.
- [LOSS.md](LOSS.md) — the perceptual loss: terms, weights, profiles (the heart).
- [ROADMAP.md](ROADMAP.md) — milestones, deliverables, validation gates.

---

## 1. Goal

Build a command‑line tool that:

1. Takes an instrumental music file as input (`wav`, `mp3`, `ogg`, `flac`).
2. Produces an **AY register stream**: one 14‑register dump per frame, **50 frames
   per second** (PAL timing), suitable for a hardware/emulator replay routine.
3. Offers a **preview mode** that renders the produced register stream back to
   audio (`mp3`/`wav`) so the result can be auditioned and validated.

The core philosophy: **the AY chip is the language.** We do not transcribe to
notes/MIDI/tracker effects. We search directly over register values, using a
cycle‑accurate emulator inside the loop, and score candidates by how they
*sound* rather than how their parameters compare.

```
Original Audio → Features → Initial Register Guess → AY Emulator →
Synthesized Audio → Perceptual Similarity → Register Optimization → repeat →
Final Register Stream  (→ optional preview render to mp3)
```

---

## 2. Why analysis‑by‑synthesis

The conventional pipeline is `Audio → Notes → Tracker → Registers`. Every step
loses information and injects assumptions the AY never had. The AY does not play
notes; it plays *register states*. So the target representation of the search
**is** the output format. There is no lossy intermediate.

Consequences:

- The objective is perceptual, not symbolic — a slightly "wrong" pitch that
  *sounds* closer (e.g. because another channel masks the beating) is preferred.
- Hard hardware constraints (3 tone channels, 1 noise generator, 1 shared
  envelope, quantized periods, 16 volume levels) are satisfied *by construction*,
  because we only ever propose legal register states.
- Quality is bounded only by the optimizer and the loss, not by a transcription
  front‑end.

The risk: brute perceptual search over a 14‑register × 2500‑frame space is huge.
The plan mitigates this with diverse heuristic proposals (multi‑start to find a
good basin of attraction), coarse‑to‑fine (multi‑resolution) search, beam/
annealing search with locality, temporal coupling, and (later) a learned
proposal / move‑selection network.

---

## 3. The AY constraint budget (what we are searching within)

Per frame we must emit a legal AY‑3‑8910 state. Summary (full detail in
[AY_REFERENCE.md](AY_REFERENCE.md)):

| Resource | Count | Notes |
|---|---|---|
| Tone channels (A/B/C) | 3 | 12‑bit period each → quantized pitch |
| Noise generator | 1 | 5‑bit period, **shared** across channels |
| Envelope generator | 1 | 16‑bit period + 4‑bit shape, **shared** |
| Volume per channel | 4‑bit | 16 levels, or "use envelope" flag |
| Mixer | 6 bits | per‑channel tone enable + noise enable |

So at most **3 simultaneous pitched voices + 1 noise color + 1 shared envelope**.
The optimizer's job is to allocate these scarce resources to whatever in the
source is most perceptually salient at each moment.

Clock and frame model: default ZX‑Spectrum clock **1.7734 MHz**, frame rate
**50 Hz** → register updates land on 20 ms boundaries; the emulator renders
sub‑frame at full audio rate. Clock is configurable (`--clock`, `--fps`).

---

## 4. Pipeline stages

### Stage 1 — Audio analysis (perceptual description)

Frame the input at the AY update rate (20 ms, 50 Hz, with overlap for analysis).
For each frame compute a perceptual feature vector — *describe what the listener
hears*, not what notes exist. Organized in three tiers to keep the philosophy
consistent (raw percepts first, interpretations clearly marked):

- **Acoustic features** (what the ear receives): loudness (ITU‑R BS.1770), bass
  energy, brightness, spectral centroid, spectral flatness, harmonic vs. noise
  energy split, transient/onset strength.
- **Musical features** (interpretations derived from the acoustic layer): chroma
  (12‑bin), dominant‑pitch salience + confidence, beat position, tempo,
  harmonic stability.
- **Optional AI features** (later): learned embeddings / source‑aware cues.

Optional **stem separation** (Demucs/Spleeter) to analyze bass / lead / drums /
harmony independently — this strongly informs channel allocation.

### Stage 2 — Candidate generator (diverse plausible guesses)

Map `FrameFeatures → N diverse candidate AY states`. Outputs need not be correct,
only plausible — the hard part is not optimization but **finding a good basin of
attraction**, so we hand the optimizer several starting points and keep the best:

- e.g. a melody‑led, a bass‑led, an arpeggio, and a noise‑heavy proposal.
- Salient pitch(es) → tone periods on 1–3 channels (bass → one channel).
- Transient/percussive energy → noise enable + noise period from brightness.
- Loudness → volume levels; sustained swells → shared envelope.
- **Instrument behaviors act as a prior here** (attack/decay/wobble/ornament
  curves bias the proposals). They live entirely in the proposal stage — the
  optimizer never knows "what a piano is" (see Stage 7).

### Stage 3 — AY emulator (the sacred core)

A cycle‑accurate AY‑3‑8910 emulator renders register state(s) to PCM. Everything
downstream depends on its fidelity, so it is built and tested first. Must model:
tone counters, 17‑bit LFSR noise, envelope shapes, mixer, 16‑step (logarithmic)
DAC, and accurate sub‑frame timing. Validated against reference YM/PSG playback.

### Stage 4 — Similarity engine (compare sound, not registers)

Render candidate → extract the same perceptual features → compute a single loss
against the original frame's features. Weighted combination of:

- mel‑spectrogram / log‑magnitude STFT distance (primary)
- loudness, spectral centroid (timbre), chroma (harmony)
- transient/onset alignment (rhythm), harmonic similarity

`loss = 0` ⇒ perfect perceptual match. Weights are configurable presets
(e.g. `--profile melodic|percussive|balanced`).

### Stage 5 — Optimizer (replaces "scheduling")

Search the legal state space with discrete moves on the **semantic `AYState`**
(never raw registers): change pitch, change volume, change/replace envelope, swap
channel assignment, mute, enable/disable noise, change noise pitch, change
ornament. Each move → render → compare → keep best. The optimizer is built from
three decoupled interfaces — `MoveGenerator` (what is legal), `Evaluator`
(render + loss), `SearchStrategy` (which candidates to explore) — so search
algorithms (local / beam / annealing / genetic / MCTS …) are interchangeable
plugins (see [ARCHITECTURE.md](ARCHITECTURE.md)). Two robustness multipliers:

- **Multi‑start:** optimize each window from all N Stage‑2 proposals and keep the
  best basin — a good basin matters more than polishing one guess.
- **Coarse‑to‑fine:** optimize at 160 ms first, then refine 80→40→20 ms
  (image‑pyramid style) to escape bad local minima before committing detail.

The emulator is in the loop, so caching and incremental rendering matter for
speed.

### Stage 6 — Temporal optimization (continuity)

Optimize *windows* of frames jointly (e.g. 125–129), not isolated frames, so
channel assignments, pitches, volumes and envelopes evolve smoothly. Add a
temporal‑smoothness penalty to the loss (discourage thrashing channel swaps and
register jitter) — analogous to motion coherence in video compression.

### Stage 7 — Instrument behaviors (a prior on proposals)

Define instruments as **behaviors**, not waveforms (the AY has no waveforms):
attack, decay, pitch wobble, brightness trajectory, envelope use, noise use,
typical volume curve, typical ornament. Behaviors are a **prior** that shapes the
Stage‑2 proposals and constrains temporal evolution — they are *not* a stage the
optimizer reasons through, and the optimizer holds no concept of "piano".

### Stage 8 — Learning (optional, later)

Introduce ML only for **proposal generation** and **move selection**, not
transcription: a network maps `audio window → initial proposals`, and/or scores
which legal moves are promising to prune the search (much easier than predicting
registers). The optimizer still refines (diffusion‑style: AI proposes,
optimization corrects). Learned move‑selection is just another `SearchStrategy`.
Training data is bootstrapped from the optimizer's own high‑quality outputs and
from synthetic AY renders.

### Stage 9 — Perceptual search refinement

Final micro‑search over near‑optimal states by *listening* — accept small period
detunes if they reduce perceptual loss (masking, beating), which symbolic methods
cannot capture.

### Stage 10 — Register stream export

Emit per‑frame `R0…R13` at 50 Hz in the replay format(s): raw register dump,
**PSG**, and **YM5/YM6** (see [AY_REFERENCE.md](AY_REFERENCE.md)). Optional
delta/RLE compression for size.

---

## 5. Preview / validation mode

`audio2ay preview` (or `--preview out.mp3` on convert) takes a register stream
(or runs convert then previews) and renders it through the **same** emulator to
`wav`/`mp3`. This closes the loop for the user exactly as the optimizer's loss
closes it internally — what you hear in preview is what the loss optimized.

Validation artifacts: side‑by‑side original vs. AY render, per‑frame loss plot,
and an overall perceptual score, written to a report for each sample.

---

## 6. Success criteria

- **Correctness:** every emitted frame is a legal AY state; replay format parses
  in standard players (e.g. AYM/YM tools) and on emulator.
- **Fidelity:** preview render is recognizably the source; perceptual loss
  trends down across optimizer iterations on the provided samples.
- **Usability:** single command converts a file; preview produces an mp3.
- **Determinism:** same input + seed ⇒ same output.

Validation set: `samples/short/*` for fast iteration, `samples/long/*` for
full‑track runs and performance/quality gates.

---

## 7. Scope and non‑goals

In scope: instrumental input, AY‑3‑8910 (and YM2149 quirks via clock/volume
table), 50 Hz mono register stream, preview render, CLI.

Out of scope (initial): vocals/lyrics handling, real‑time conversion, multi‑chip
(TurboSound) output, GUI. These are possible future extensions.

See [ROADMAP.md](ROADMAP.md) for the phased build order and acceptance gates.
