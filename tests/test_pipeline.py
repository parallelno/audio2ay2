"""Tests for the perceptual loss, moves and a tiny end-to-end conversion."""

import numpy as np

from audio2ay2.ay.registers import AYState
from audio2ay2.ay.timing import Timing
from audio2ay2.ay.emulator import AYEmulator
from audio2ay2.config import Config
from audio2ay2.optimizer.moves import DefaultMoves
from audio2ay2.optimizer.temporal import semantic_distance
from audio2ay2.similarity.loss import PerceptualLoss


def test_loss_zero_for_identical_audio():
    timing = Timing()
    s = AYState(tone_period=(300, 300, 300), mixer=0b111110, volume=(12, 0, 0))
    pcm = AYEmulator(timing).render([s] * 10)
    loss = PerceptualLoss(timing.sample_rate, Config().resolved_weights())
    target = loss.features(pcm)
    assert loss.compare(target, pcm) < 1e-6


def test_loss_orders_candidates_sensibly():
    # A tone closer to the target should score lower than a far-off one.
    timing = Timing()
    target_state = AYState(tone_period=(300, 1, 1), mixer=0b111110, volume=(12, 0, 0))
    near = AYState(tone_period=(305, 1, 1), mixer=0b111110, volume=(12, 0, 0))
    far = AYState(tone_period=(900, 1, 1), mixer=0b111110, volume=(12, 0, 0))
    pcm = AYEmulator(timing).render([target_state] * 10)
    loss = PerceptualLoss(timing.sample_rate, Config(profile="melodic").resolved_weights())
    target = loss.features(pcm)
    near_pcm = AYEmulator(timing).render([near] * 10)
    far_pcm = AYEmulator(timing).render([far] * 10)
    assert loss.compare(target, near_pcm) < loss.compare(target, far_pcm)


def test_moves_are_all_legal():
    moves = DefaultMoves()
    s = AYState(tone_period=(300, 400, 500), volume=(10, 8, 6))
    for m in moves.legal(s):
        result = m.apply(s)
        assert all(1 <= t <= 4095 for t in result.tone_period)
        assert 1 <= result.noise_period <= 31


def test_semantic_distance_is_zero_for_same_state():
    s = AYState(tone_period=(300, 400, 500), volume=(10, 8, 6))
    assert semantic_distance(s, s) == 0.0


def test_semantic_distance_grows_with_pitch_change():
    # Channels must be audible for pitch to count (silent frames are identical).
    a = AYState(tone_period=(300, 300, 300), mixer=0b111000, volume=(10, 10, 10))
    b = AYState(tone_period=(301, 300, 300), mixer=0b111000, volume=(10, 10, 10))
    c = AYState(tone_period=(600, 300, 300), mixer=0b111000, volume=(10, 10, 10))
    assert semantic_distance(a, b) < semantic_distance(a, c)


def test_end_to_end_tiny_conversion():
    from audio2ay2.pipeline import convert_pcm

    timing = Timing()
    # Synthesize a simple 0.4s AY tone as the "source" to convert.
    src_state = AYState(tone_period=(300, 1, 1), mixer=0b111110, volume=(13, 0, 0))
    pcm = AYEmulator(timing).render([src_state] * 20)
    config = Config(optimizer="local", proposals=2, pyramid_ms=(80, 20), iters=20)
    result = convert_pcm(pcm, timing, config)
    assert len(result.states) > 0
    assert all(isinstance(s, AYState) for s in result.states)
