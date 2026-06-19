"""Tests for export/import round-trips (raw, PSG) and compression."""

from audio2ay.ay.registers import AYState, USE_ENV
from audio2ay.export import write_raw, read_raw, write_psg, read_psg, write_ym
from audio2ay.export.compression import rle_encode, rle_ratio


def _sample_stream():
    return [
        AYState(tone_period=(284, 300, 1), mixer=0b111100, volume=(12, 8, 0)),
        AYState(tone_period=(284, 300, 1), mixer=0b111100, volume=(12, 8, 0)),
        AYState(tone_period=(200, 300, 1), noise_period=5, mixer=0b110100,
                volume=(USE_ENV, 8, 0), env_period=2000, env_shape=0x0A),
    ]


def test_raw_roundtrip(tmp_path):
    states = _sample_stream()
    p = tmp_path / "s.ay"
    write_raw(p, states, clock_hz=1773400, fps=50)
    back, clock, fps = read_raw(p)
    assert clock == 1773400 and fps == 50
    assert len(back) == len(states)
    for a, b in zip(states, back):
        assert a.canonical() == b.canonical()


def test_psg_roundtrip(tmp_path):
    states = _sample_stream()
    p = tmp_path / "s.psg"
    write_psg(p, states)
    back = read_psg(p)
    assert len(back) == len(states)
    for a, b in zip(states, back):
        assert a.canonical() == b.canonical()


def test_ym_writes_valid_header(tmp_path):
    p = tmp_path / "s.ym"
    write_ym(p, _sample_stream(), clock_hz=2000000, fps=50)
    data = p.read_bytes()
    assert data[:4] == b"YM6!"
    assert data[4:12] == b"LeOnArD!"
    assert data[-4:] == b"End!"


def test_rle_is_compact_for_static_stream():
    static = [AYState(tone_period=(284, 284, 284), volume=(10, 10, 10))] * 50
    assert rle_ratio(static) < 0.2
    assert len(rle_encode(static)) == 14  # one run per register lane
