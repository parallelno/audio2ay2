# AY‑3‑8910 Hardware Reference

Everything the optimizer, emulator and exporter must respect. The search space
*is* this register set, so these constraints define what is legal to propose.

---

## 1. Register map (R0–R13)

| Reg | Bits used | Meaning |
|----|----|----|
| R0 | 8 | Channel A tone period — fine |
| R1 | 4 | Channel A tone period — coarse |
| R2 | 8 | Channel B tone period — fine |
| R3 | 4 | Channel B tone period — coarse |
| R4 | 8 | Channel C tone period — fine |
| R5 | 4 | Channel C tone period — coarse |
| R6 | 5 | Noise period |
| R7 | 6 (+2 I/O) | Mixer: tone/noise enables (active **low**) |
| R8 | 5 | Channel A amplitude (bit4 = use envelope) |
| R9 | 5 | Channel B amplitude (bit4 = use envelope) |
| R10 | 5 | Channel C amplitude (bit4 = use envelope) |
| R11 | 8 | Envelope period — fine |
| R12 | 8 | Envelope period — coarse |
| R13 | 4 | Envelope shape/cycle |

A **frame** is the full set R0–R13 (14 bytes). The stream is one frame every
20 ms (50 Hz). R7 bits 6–7 are I/O port direction — set to inputs (0) for audio;
we never drive the I/O ports.

---

## 2. Tone period and pitch

Tone period `TP` is a 12‑bit value: `TP = (coarse << 8) | fine`, range 1–4095
(0 is treated as 1).

```
f_tone = clock / (16 * TP)
TP     = round( clock / (16 * f_target) )
```

- Pitch is **quantized** by TP being integer — resolution coarsens at high pitch
  and is fine at low pitch. The optimizer must search neighboring TP values, and
  may accept a detuned TP if it sounds closer (Stage 9).
- With clock = 1.7734 MHz: `f = 1773400 / (16*TP)`. TP=1 → ~110.8 kHz (inaudible),
  TP=4095 → ~27 Hz. Musical range roughly TP ≈ 30–1000.

---

## 3. Noise generator

- One 5‑bit period `NP` (R6), range 1–31, shared by all channels routed to noise.
  `f_noise = clock / (16 * NP)`.
- Implemented as a 17‑bit LFSR (taps at bits 0 and 3 on the AY). The emulator must
  reproduce the exact LFSR for correct spectral color.
- Only **one** noise color exists at a time — percussion/hats/snares all share it.
  The optimizer picks the NP that best matches dominant transient brightness.

---

## 4. Mixer (R7, active low)

| Bit | 0 = enabled | Function |
|----|----|----|
| 0 | Tone A on | Channel A tone |
| 1 | Tone B on | Channel B tone |
| 2 | Tone C on | Channel C tone |
| 3 | Noise A on | Noise into channel A |
| 4 | Noise B on | Noise into channel B |
| 5 | Noise C on | Noise into channel C |
| 6 | I/O A dir | keep 0 |
| 7 | I/O B dir | keep 0 |

A channel can mix tone AND noise simultaneously. "Silent" channel = tone+noise
disabled or amplitude 0.

---

## 5. Amplitude / volume (R8–R10)

- Bits 0–3: 4‑bit fixed level, **16 steps**, logarithmic DAC (~3 dB/step on AY,
  non‑linear). Use the correct volume table per chip (AY vs YM2149 differ).
- Bit 4 = 1: ignore fixed level, **use envelope** generator for this channel's
  amplitude (envelope is shared).
- Emulator must use a measured/standard volume LUT, not a linear ramp.

---

## 6. Envelope generator (R11–R13)

- 16‑bit period `EP = (R12 << 8) | R11`; `f_env = clock / (256 * EP)`.
- R13 selects one of the shape/cycle patterns (Continue/Attack/Alternate/Hold
  bits). The 8 audible shapes: `\___`, `/___`, `\\\\`, `\~~~`(\then up alt), etc.
  Standard shapes 8–15 (0x08,0x0A,0x0C,0x0E most used). The emulator implements
  all 16 R13 values.
- **One** envelope shared by all channels using it. Fast envelopes (small EP) can
  act as a buzzer/bass timbre; slow envelopes do volume swells. The optimizer
  treats the envelope as a single shared resource and assigns it where it helps
  most.

---

## 7. Clock variants

| Platform | Chip | Clock (MHz) |
|----|----|----|
| ZX Spectrum 128 | AY‑3‑8912 | 1.7734 |
| Atari ST | YM2149 | 2.0 |
| Amstrad CPC | AY‑3‑8912 | 1.0 |
| MSX | AY‑3‑8910 | 1.7897725 |

Clock is a CLI option (`--clock`, default 1.7734e6) because it changes the
TP→pitch mapping and the available pitch resolution. YM2149 also has a 5‑bit
volume mode and slightly different volume table — selectable via `--chip`.

---

## 8. Frame timing model

- Output rate fixed at 50 Hz (PAL). `--fps` allows 60 Hz (NTSC) experiments.
- Registers are latched at frame boundaries; between boundaries the generators
  free‑run. The emulator renders continuously at the audio sample rate and only
  applies new register values on the 20 ms boundary.
- Internal audio render rate (e.g. 44.1/48 kHz) is independent of the 50 Hz
  control rate.

---

## 9. Output / replay formats

The exporter targets these (raw is canonical; others for tooling/players):

- **Raw register dump** — `frames × 14 bytes`, plus a small header
  (clock, fps, frame count). Simplest replay routine.
- **PSG** — `PSG\x1A` header then a stream of `reg,value` pairs with `0xFF`
  end‑of‑frame and `0xFE` multi‑frame‑wait markers.
- **YM5/YM6** — interleaved register format used by YM players/StSound; supports
  metadata (title, author, clock, frame rate). Good for cross‑checking fidelity
  in existing players.
- Optional **delta/RLE compression** for size (most registers change rarely).

All formats encode exactly the same 50 Hz `R0…R13` content; choice is `--format`.

---

## 10. Emulator correctness checklist

- [ ] Tone counters and 50 % duty square output per channel.
- [ ] 17‑bit LFSR noise with correct taps and period scaling.
- [ ] All 16 envelope shapes (R13) cycle‑accurate.
- [ ] Logarithmic 16‑step volume LUT (chip‑specific).
- [ ] Mixer combines tone/noise per channel correctly (active‑low R7).
- [ ] Sub‑frame register latching at 50 Hz, audio render at 44.1/48 kHz.
- [ ] Cross‑validated against a reference player on known YM/PSG files.
