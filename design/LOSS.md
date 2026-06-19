# LOSS — The Perceptual Objective

> The entire project depends on `loss(audio_original, audio_ay)`.
> If the loss is wrong, every optimizer faithfully reproduces the wrong answer.

This document is the contract for the single most important function in the
system. It starts small and is expected to grow into the most detailed document
in `design/`. See [PLAN.md](PLAN.md) Stage 4 and [ARCHITECTURE.md](ARCHITECTURE.md)
§4 (the `Evaluator` interface) for context.

---

## 1. Principle

Compare **sound**, not registers. The loss takes two audio signals — the original
frame/window and the AY render of a candidate `AYState` — and returns a single
non‑negative scalar where `0` means *perceptually identical*. It must correlate
with human judgement, not with parameter distance: a detuned tone period that
*sounds* closer (because another channel masks the beating) must score lower than
the "numerically correct" one.

```
original audio ─┐
                ├─ same feature extraction ─→ difference ─→ weighted sum ─→ loss
ay render ──────┘
```

---

## 2. Terms (initial set)

Each term is normalized to roughly comparable scale, then combined with weights.

| Term | Captures | Typical metric |
|---|---|---|
| `mel` / `stft` | overall timbre & spectral shape (**primary**) | L1/L2 on log‑mel or log‑magnitude STFT |
| `loudness` | perceived level / dynamics | BS.1770 loudness difference |
| `centroid` | brightness / timbre tilt | spectral‑centroid difference |
| `chroma` | harmony / pitch class | cosine distance on 12‑bin chroma |
| `harmonic` | tonal similarity | harmonic‑energy correlation |
| `transient` | rhythm / onset alignment | onset‑envelope distance / cross‑corr |
| `temporal` | continuity (state‑to‑state) | smoothness penalty on `AYState` deltas |

`temporal` is a regularizer on the candidate sequence (not an audio comparison);
it lives in the same scalar so the optimizer trades fidelity against continuity.

```
loss = w_mel*mel + w_loud*loudness + w_cent*centroid
     + w_chroma*chroma + w_harm*harmonic + w_trans*transient
     + w_temp*temporal
```

---

## 3. Profiles (weight presets)

Exposed via `--profile`. Starting points, to be tuned against the samples:

| Profile | Emphasis | Use |
|---|---|---|
| `balanced` | even weights | general instrumental (default) |
| `melodic` | chroma + harmonic + mel | tonal/lead material (`trumpet.ogg`, `01`) |
| `percussive` | transient + mel | drums/rhythm (`03_drum_loop.wav`) |

Profiles are data (a weight table), not code paths.

---

## 4. Requirements

- **Range:** `loss ≥ 0`, `0` ⇔ perceptually identical; monotone with perceived
  difference.
- **Determinism:** same inputs ⇒ same scalar (no randomness, fixed FFT params).
- **Speed:** evaluated millions of times — vectorized, cache feature extraction
  of the (fixed) original, reuse FFT plans. The `Evaluator` caches per‑`AYState`
  renders and may cache features.
- **Window‑aware:** operates on single frames and on temporal windows (for
  coarse‑to‑fine and temporal optimization).
- **Scale‑aware:** usable at each pyramid resolution (160/80/40/20 ms).

---

## 5. Validation

- **Ordering test:** a hand‑authored "good" AY rendition of a sample scores lower
  than random/degraded states.
- **Monotonicity test:** progressively detuning/muting a known‑good state raises
  the loss.
- **Landscape probe:** sweep one field (e.g. tone period ±N) and plot loss — the
  surface should be informative (not flat, not pure noise). See
  `report/landscape.py`.
- **Human A/B:** spot‑check that lower loss ⇒ subjectively closer preview.

---

## 6. Open questions (to expand here over time)

- Multi‑resolution STFT vs. mel: which best matches perception for square/noise
  timbres specific to the AY?
- Masking model: incorporate simultaneous/temporal masking so inaudible errors
  are not penalized.
- Differentiable surrogate: a smooth approximation of the loss to enable
  gradient‑guided proposals (the emulator itself is non‑differentiable).
- Phase sensitivity: how much, if any, phase information to include.
- Normalization across terms so weights are interpretable and transferable.
