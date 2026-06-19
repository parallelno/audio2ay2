"""Tests for the AYState <-> registers boundary and the emulator."""

import numpy as np
import pytest

from audio2ay2.ay.registers import AYState, USE_ENV
from audio2ay2.ay.emulator import AYEmulator
from audio2ay2.ay.timing import Timing


def test_register_roundtrip_identity():
    s = AYState(tone_period=(284, 1000, 4095), noise_period=17, mixer=0b101010,
                volume=(12, USE_ENV, 3), env_period=4321, env_shape=0x0A)
    assert AYState.from_registers(s.to_registers()).canonical() == s.canonical()


def test_register_dump_is_14_bytes():
    assert len(AYState().to_registers()) == 14


def test_canonical_clamps_fields():
    s = AYState(tone_period=(0, 99999, -5), noise_period=99, volume=(99, -1, 8),
                env_shape=99).canonical()
    assert all(1 <= t <= 4095 for t in s.tone_period)
    assert 1 <= s.noise_period <= 31
    assert all(0 <= v <= 15 or v == USE_ENV for v in s.volume)
    assert 0 <= s.env_shape <= 15


def test_use_env_flag_survives_roundtrip():
    s = AYState(volume=(USE_ENV, 0, USE_ENV))
    back = AYState.from_registers(s.to_registers())
    assert back.volume[0] == USE_ENV and back.volume[2] == USE_ENV


def test_emulator_tone_frequency_is_correct():
    # f = clock / (16 * TP). Pick TP for ~440 Hz at default clock.
    timing = Timing()
    tp = round(timing.clock_hz / (16 * 440))
    s = AYState(tone_period=(tp, tp, tp), mixer=0b111110, volume=(15, 0, 0))
    emu = AYEmulator(timing)
    pcm = emu.render([s] * 50)  # 1 second
    spec = np.abs(np.fft.rfft(pcm * np.hanning(len(pcm))))
    freqs = np.fft.rfftfreq(len(pcm), 1.0 / timing.sample_rate)
    peak = freqs[int(np.argmax(spec))]
    assert abs(peak - 440) < 15  # square-wave fundamental near target


def test_emulator_silence_when_all_disabled():
    s = AYState(mixer=0b111111, volume=(0, 0, 0))
    emu = AYEmulator(Timing())
    pcm = emu.render([s] * 10)
    assert float(np.sqrt(np.mean(pcm ** 2))) < 1e-3


def test_emulator_noise_is_broadband():
    s = AYState(noise_period=1, mixer=0b110111, volume=(15, 0, 0))
    emu = AYEmulator(Timing())
    pcm = emu.render([s] * 25)
    assert float(np.sqrt(np.mean(pcm ** 2))) > 1e-3


def test_determinism_same_states_same_output():
    s = AYState(tone_period=(300, 400, 500), mixer=0b101010, volume=(10, 8, 6))
    a = AYEmulator(Timing()).render([s] * 20)
    b = AYEmulator(Timing()).render([s] * 20)
    assert np.array_equal(a, b)
