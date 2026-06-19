# Architecture, Modules & CLI

How the pipeline in [PLAN.md](PLAN.md) decomposes into code, the data contracts
between stages, and the command‑line surface.

---

## 1. Language & key dependencies

- **Python 3.11+** for the pipeline (fast to iterate, rich audio/ML ecosystem).
- Audio I/O & DSP: `numpy`, `scipy`, `librosa`, `soundfile`, `ffmpeg` (decode
  mp3/ogg, encode mp3 preview).
- Optional stems: `demucs`. Optional ML (later): `torch`.
- The **AY emulator** is the hot loop: prototype in Python/NumPy (vectorized),
  then port to a C/Cython/Rust extension once the interface is stable. Keep a
  pure‑Python reference for correctness tests.

> The emulator interface is frozen early so the optimizer never cares whether the
> backend is Python or native.

---

## 2. Module layout

```
audio2ay/
├── analysis/
│   ├── feature_extractor.py   # FrameFeatures per 20 ms
│   ├── onset_detector.py      # transients / beat grid
│   ├── chroma.py              # 12-bin chroma + salience
│   ├── loudness.py            # BS.1770 loudness, bass/brightness
│   ├── salience.py            # dominant-pitch estimation + confidence
│   └── stem_separator.py      # optional Demucs wrapper
│
├── ay/
│   ├── registers.py           # AYState dataclass, encode/decode R0..R13
│   ├── emulator.py            # cycle-accurate state -> PCM
│   ├── renderer.py            # frame/window rendering, caching
│   └── timing.py              # clock, fps, sample-rate conversions
│
├── candidate/
│   ├── generator.py           # FrameFeatures -> N diverse AYState proposals
│   ├── behaviors.py           # instrument behavior PRIOR (biases proposals)
│   └── heuristics.py          # pitch->TP, brightness->NP, loudness->vol
│
├── optimizer/
│   ├── interfaces.py          # MoveGenerator / Evaluator / SearchStrategy ABCs
│   ├── moves.py               # legal discrete moves over AYState (MoveGenerator)
│   ├── evaluator.py           # wraps render+loss (Evaluator), caches
│   ├── strategies/
│   │   ├── local_search.py
│   │   ├── beam_search.py
│   │   ├── simulated_annealing.py
│   │   └── genetic.py
│   ├── multistart.py          # run strategy from N proposals, keep best basin
│   ├── pyramid.py             # coarse-to-fine (multi-resolution) scheduler
│   └── temporal.py            # windowed/joint optimization + smoothness
│
├── similarity/
│   ├── mel.py                 # mel / STFT distance
│   ├── chroma.py              # harmonic similarity
│   ├── psychoacoustic.py      # loudness/masking-aware terms
│   └── loss.py                # weighted combined perceptual loss
│
├── ml/                        # optional, later
│   ├── proposal_network.py
│   └── training.py
│
├── export/
│   ├── register_stream.py     # raw dump
│   ├── psg.py                 # PSG format
│   ├── ym.py                  # YM5/YM6 format
│   └── compression.py         # delta/RLE
│
├── preview/
│   └── render.py              # stream -> emulator -> wav/mp3
│
├── report/
│   ├── validate.py            # loss plots, original-vs-AY comparison
│   ├── history.py             # optimization history (iteration vs loss)
│   └── landscape.py           # loss-landscape sweeps (±1 register diagnostics)
│
├── cli.py                     # argument parsing, command dispatch
└── config.py                  # profiles, defaults, seeds
```

---

## 3. Data contracts

```python
@dataclass
class FrameFeatures:
    loudness: float
    bass_energy: float
    brightness: float
    spectral_centroid: float
    harmonic_energy: float
    transient_energy: float
    chroma: np.ndarray          # shape (12,)
    onset_strength: float
    beat_position: float
    tempo: float
    pitch_hz: float
    pitch_confidence: float
    harmonic_stability: float

@dataclass
class AYState:                  # one legal frame — SEMANTIC, not registers
    tone_period:  tuple[int,int,int]   # A,B,C  (1..4095)
    noise_period: int                  # 1..31
    mixer: int                         # R7 bits (active low)
    volume: tuple[int,int,int]         # 0..15 or USE_ENV flag
    env_period: int                    # 0..65535
    env_shape: int                     # 0..15
    def to_registers(self) -> bytes:   # 14 bytes — ONLY the exporter calls this
    @classmethod
    def from_registers(cls, b: bytes) -> "AYState":

RegisterStream = list[AYState]   # one entry per 50 Hz frame
```

> **Strict boundary rule.** Nothing upstream of `export/` ever sees `R0…R13`.
> The candidate generator, moves, evaluator, search strategies and temporal
> logic operate exclusively on the semantic `AYState` (channels / noise /
> envelope). `to_registers()` / `from_registers()` are the *only* crossing
> points, owned by the exporter and the raw‑stream importer. This keeps moves
> readable ("detune channel A", "swap A↔B") and makes a future second chip a
> matter of a new state type + exporter, not an optimizer rewrite.

Stage interfaces:

```
analysis:   audio (wav/pcm)        -> list[FrameFeatures]
candidate:  FrameFeatures          -> list[AYState]      # N diverse proposals
emulator:   AYState | window       -> PCM
similarity: (orig_pcm, ay_pcm)     -> float loss
optimizer:  (features, proposals)  -> RegisterStream
export:     RegisterStream         -> file (raw/psg/ym)
preview:    RegisterStream         -> wav/mp3
```

---

## 4. Optimizer: three decoupled interfaces

The optimizer never knows about registers or about *which* search algorithm runs.
It is built from three plug‑in interfaces so search strategies can be swapped or
invented without touching AY‑specific logic:

```python
class MoveGenerator(ABC):                 # "what changes are legal"
    def legal(self, state: AYState) -> Iterable[Move]: ...

class Evaluator(ABC):                     # "how good is this state"
    def loss(self, window, state: AYState) -> float: ...   # render + perceptual loss (+cache)

class SearchStrategy(ABC):                # "which candidates to explore"
    def optimize(self, init: list[AYState], moves: MoveGenerator,
                 evaluator: Evaluator) -> AYState: ...
```

Concrete `SearchStrategy` implementations (`local_search`, `beam_search`,
`simulated_annealing`, `genetic`, and later MCTS / learned‑move‑selection) live
under `optimizer/strategies/` and are interchangeable. `local_search` ships first
so the loss can be validated against a working strategy.

## 5. Optimization loop (multi‑start, coarse‑to‑fine, per window)

```
for window in sliding_windows(features, size=W, hop=H):
    proposals = candidate.generator(window)      # N DIVERSE warm starts
                                                 # e.g. melody / bass / arp / noise-heavy
    state = None
    for scale in pyramid([160ms, 80ms, 40ms, 20ms]):   # coarse-to-fine
        seeds = proposals if state is None else [state]
        state = multistart(
            seeds, scale,
            strategy   = SearchStrategy(...),    # local/beam/annealing/genetic
            moves      = MoveGenerator(),
            evaluator  = Evaluator(loss, render_cache),
            temporal   = temporal_penalty,       # smoothness across neighbors
        )                                        # keeps the best basin
    commit(window, state)
```

- **Multi‑start (`multistart.py`):** every window is optimized from *all* N
  proposals; the best final basin wins. Robustness against bad local minima is
  the goal — finding a good basin matters more than polishing one.
- **Coarse‑to‑fine (`pyramid.py`):** optimize at 160 ms blocks first, then refine
  at 80/40/20 ms, carrying the winner down each level (image‑pyramid style). This
  escapes the worst local minima before fine detail is committed.
- `render` (inside `Evaluator`) caches per‑`AYState` PCM and supports incremental
  re‑render when only one field changed.
- `temporal_penalty` discourages register thrashing and abrupt channel swaps.

---

## 6. CLI surface

```
audio2ay convert <input> [options]      # audio -> register stream
audio2ay preview <stream> [options]     # register stream -> mp3/wav
audio2ay validate <input> [options]     # convert + compare + report
audio2ay info <stream>                  # inspect a stream
```

### `convert`

| Option | Default | Description |
|---|---|---|
| `-o, --output` | `<input>.ay` | Output stream file |
| `--format` | `raw` | `raw` \| `psg` \| `ym` |
| `--fps` | `50` | Frames per second (50 PAL / 60 NTSC) |
| `--clock` | `1773400` | AY clock in Hz |
| `--chip` | `ay` | `ay` \| `ym2149` (volume table / 5‑bit vol) |
| `--profile` | `balanced` | Loss preset: `melodic`\|`percussive`\|`balanced` |
| `--optimizer` | `annealing` | `local`\|`beam`\|`annealing`\|`genetic` |
| `--proposals` | `4` | Diverse warm starts per window (multi‑start) |
| `--pyramid` | `160,80,40,20` | Coarse‑to‑fine scales in ms (`off` to disable) |
| `--window` | `5` | Temporal window (frames) |
| `--iters` | `200` | Max optimizer iterations per window |
| `--stems` | `off` | `off`\|`demucs` source separation |
| `--seed` | `0` | Determinism |
| `--preview` | — | Also render result to this mp3/wav |
| `--report` | — | Write validation report (plots + audio) |
| `-j, --jobs` | `cpu` | Parallel windows |

### `preview`

| Option | Default | Description |
|---|---|---|
| `-o, --output` | `<stream>.mp3` | Rendered audio |
| `--format` | auto | Input stream format |
| `--samplerate` | `44100` | Render sample rate |
| `--clock`/`--chip` | match stream | Emulation parameters |

---

## 7. Performance strategy

- Emulator is the bottleneck → native backend + render cache + incremental moves.
- Windows are independent → parallelize across cores (`-j`).
- Two‑pass quality: fast `local` pass for a draft, optional `annealing` polish.
- Long tracks (`samples/long/*`) run in streaming chunks to bound memory.

---

## 8. Testing

- **Emulator**: golden‑file tests vs. a reference YM/PSG player; LFSR and
  envelope unit tests; volume‑LUT checks.
- **Round‑trip**: `AYState → registers → AYState` is identity; export→import for
  raw/psg/ym is lossless.
- **Loss sanity**: rendering the *original's* best‑known AY (hand‑made) scores
  lower than random states.
- **Determinism**: same seed ⇒ identical stream hash.
- **Regression**: track perceptual loss per sample over time; CI gate on the
  short samples.
