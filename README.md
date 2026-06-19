# audio2ay

> **The AY chip is the language.**

Convert instrumental audio into General Instrument **AY-3-8910** register streams
via **analysis-by-synthesis**: search the legal AY register space with a
cycle-accurate emulator in the loop, scoring candidates by *perceptual*
similarity to the source rather than by parameter distance.

See [design/PLAN.md](design/PLAN.md), [design/ARCHITECTURE.md](design/ARCHITECTURE.md),
[design/AY_REFERENCE.md](design/AY_REFERENCE.md), [design/LOSS.md](design/LOSS.md)
and [design/ROADMAP.md](design/ROADMAP.md).

## Install (Windows / PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

## Usage

```powershell
# activate the venv first (once per terminal session)
.\.venv\Scripts\Activate.ps1
```

```powershell
# audio -> 50 Hz register stream (+ optional mp3 preview)
audio2ay2 convert samples\short\01_arpeggio_mono.wav -o out.ay --preview out.mp3

# render a stream back to audio
audio2ay2 preview out.ay -o out.mp3

# convert + compare + validation report (audio + loss plot + score)
audio2ay2 validate samples\short\01_arpeggio_mono.wav -o reports\arp

# inspect a stream
audio2ay2 info out.ay
```

Key options for `convert`: `--format raw|psg|ym`, `--profile balanced|melodic|percussive`,
`--optimizer local|beam|annealing|genetic`, `--proposals N`, `--pyramid 160,80,40,20`,
`--clock`, `--fps`, `--chip ay|ym2149`, `--seed`.

## Pipeline

```
audio -> features -> diverse proposals -> [coarse-to-fine multi-start optimizer
         (emulator + perceptual loss in the loop)] -> 50 Hz register stream
         -> export (raw / PSG / YM) and/or preview (wav / mp3)
```

The optimizer is built from three decoupled interfaces — `MoveGenerator`,
`Evaluator`, `SearchStrategy` — and operates only on the semantic `AYState`;
raw registers `R0..R13` exist solely at the export boundary.
