"""audio2ay2 command-line interface.

Commands: convert, preview, validate, info. See design/ARCHITECTURE.md s.6.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .ay.timing import Timing, DEFAULT_CLOCK_HZ, DEFAULT_FPS, DEFAULT_SAMPLE_RATE
from .config import Config, PROFILES, seed_everything


def _parse_pyramid(value: str) -> tuple[int, ...]:
    if value.strip().lower() == "off":
        return ()
    return tuple(int(x) for x in value.split(",") if x.strip())


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--clock", type=int, default=DEFAULT_CLOCK_HZ, help="AY clock (Hz)")
    p.add_argument("--fps", type=int, default=DEFAULT_FPS, help="frames per second")
    p.add_argument("--chip", choices=["ay", "ym2149"], default="ay")
    p.add_argument("--samplerate", type=int, default=DEFAULT_SAMPLE_RATE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="audio2ay2", description=__doc__)
    parser.add_argument("--version", action="version", version=f"audio2ay2 {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("convert", help="audio -> register stream")
    c.add_argument("input")
    c.add_argument("-o", "--output", default=None)
    c.add_argument("--format", choices=["raw", "psg", "ym"], default="raw")
    c.add_argument("--profile", choices=list(PROFILES), default="balanced")
    c.add_argument("--optimizer", choices=["local", "beam", "annealing", "genetic"],
                   default="annealing")
    c.add_argument("--proposals", type=int, default=4)
    c.add_argument("--pyramid", type=str, default="160,80,40,20")
    c.add_argument("--window", type=int, default=5)
    c.add_argument("--iters", type=int, default=200)
    c.add_argument("--stabilize", type=int, default=3,
                   help="post median-smoothing width (frames); 0/1 disables")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--preview", default=None, help="also render result to this mp3/wav")
    c.add_argument("--report", default=None, help="write validation report to this dir")
    _add_common(c)

    pv = sub.add_parser("preview", help="register stream -> mp3/wav")
    pv.add_argument("stream")
    pv.add_argument("-o", "--output", default=None)
    _add_common(pv)

    v = sub.add_parser("validate", help="convert + compare + report")
    v.add_argument("input")
    v.add_argument("-o", "--output", default="reports", help="report directory")
    v.add_argument("--profile", choices=list(PROFILES), default="balanced")
    v.add_argument("--optimizer", choices=["local", "beam", "annealing", "genetic"],
                   default="annealing")
    v.add_argument("--proposals", type=int, default=4)
    v.add_argument("--pyramid", type=str, default="160,80,40,20")
    v.add_argument("--iters", type=int, default=200)
    v.add_argument("--seed", type=int, default=0)
    _add_common(v)

    i = sub.add_parser("info", help="inspect a register stream")
    i.add_argument("stream")

    return parser


def _config_from_args(args) -> Config:
    return Config(
        profile=args.profile,
        optimizer=args.optimizer,
        proposals=args.proposals,
        pyramid_ms=_parse_pyramid(args.pyramid),
        window=getattr(args, "window", 5),
        iters=args.iters,
        stabilize=getattr(args, "stabilize", 3),
        seed=args.seed,
        chip=args.chip,
    )


def cmd_convert(args) -> int:
    from .audioio import load_audio
    from .pipeline import convert_pcm
    from .export import write_stream

    seed_everything(args.seed)
    timing = Timing(clock_hz=args.clock, fps=args.fps, sample_rate=args.samplerate)
    config = _config_from_args(args)

    print(f"[convert] loading {args.input}")
    pcm = load_audio(args.input, timing.sample_rate)
    print(f"[convert] {pcm.shape[0] / timing.sample_rate:.1f}s @ {timing.sample_rate} Hz; "
          f"optimizer={config.optimizer} proposals={config.proposals} "
          f"pyramid={config.pyramid_ms or 'off'}")

    result = convert_pcm(pcm, timing, config)
    out = args.output or (str(Path(args.input).with_suffix("." + args.format)))
    write_stream(out, result.states, args.format, timing.clock_hz, timing.fps)
    mean_loss = (sum(result.finest_losses) / len(result.finest_losses)
                 if result.finest_losses else 0.0)
    print(f"[convert] wrote {out} ({len(result.states)} frames, "
          f"mean block loss {mean_loss:.4f})")

    if args.preview:
        from .preview.render import write_preview
        write_preview(args.preview, result.states, timing, config.chip)
        print(f"[convert] preview -> {args.preview}")

    if args.report:
        from .similarity.loss import PerceptualLoss
        from .report.validate import write_report
        loss = PerceptualLoss(timing.sample_rate, config.resolved_weights())
        summary = write_report(args.report, pcm, result.states, timing, loss, config.chip)
        print(f"[convert] report -> {args.report} ({summary})")
    return 0


def cmd_preview(args) -> int:
    from .export import read_stream
    from .preview.render import write_preview

    states, clock, fps = read_stream(args.stream)
    timing = Timing(clock_hz=clock, fps=fps, sample_rate=args.samplerate)
    out = args.output or (str(Path(args.stream).with_suffix(".mp3")))
    write_preview(out, states, timing, args.chip)
    print(f"[preview] {len(states)} frames -> {out}")
    return 0


def cmd_validate(args) -> int:
    from .audioio import load_audio
    from .pipeline import convert_pcm
    from .similarity.loss import PerceptualLoss
    from .report.validate import write_report

    seed_everything(args.seed)
    timing = Timing(clock_hz=args.clock, fps=args.fps, sample_rate=args.samplerate)
    config = Config(profile=args.profile, optimizer=args.optimizer,
                    proposals=args.proposals, pyramid_ms=_parse_pyramid(args.pyramid),
                    iters=args.iters, seed=args.seed, chip=args.chip)
    pcm = load_audio(args.input, timing.sample_rate)
    result = convert_pcm(pcm, timing, config)
    loss = PerceptualLoss(timing.sample_rate, config.resolved_weights())
    summary = write_report(args.output, pcm, result.states, timing, loss, config.chip)
    print(f"[validate] report -> {args.output}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


def cmd_info(args) -> int:
    from .export import read_stream
    from .export.compression import rle_ratio

    states, clock, fps = read_stream(args.stream)
    dur = len(states) / fps if fps else 0.0
    print(f"stream: {args.stream}")
    print(f"  frames: {len(states)}  fps: {fps}  clock: {clock} Hz  duration: {dur:.1f}s")
    print(f"  rle ratio: {rle_ratio(states):.3f}")
    if states:
        s = states[len(states) // 2]
        print(f"  mid-frame: tone={s.tone_period} noise={s.noise_period} "
              f"vol={s.volume} mixer={s.mixer:#08b} env=({s.env_period},{s.env_shape:#x})")
    return 0


_DISPATCH = {
    "convert": cmd_convert,
    "preview": cmd_preview,
    "validate": cmd_validate,
    "info": cmd_info,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return _DISPATCH[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
