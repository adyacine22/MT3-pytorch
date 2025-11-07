from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from data.symbolic import vocabulary


def test_pitch_ids():
    min_id = vocabulary.pitch_id(21)
    max_id = vocabulary.pitch_id(108)
    print(f"pitch_id(21) -> {min_id}, expected {vocabulary.EVENT_RANGES[1].min_id}")
    print(f"pitch_id(108) -> {max_id}, expected {vocabulary.EVENT_RANGES[1].max_id}")
    assert min_id == vocabulary.EVENT_RANGES[1].min_id
    assert max_id == vocabulary.EVENT_RANGES[1].max_id
    with pytest.raises(ValueError):
        vocabulary.pitch_id(20)


def test_shift_ids():
    print(
        f"Using config: steps_per_second={vocabulary.STEPS_PER_SECOND}, "
        f"max_shift_ms={vocabulary.MAX_SHIFT_MS}"
    )
    max_shift = (vocabulary.STEPS_PER_SECOND * vocabulary.MAX_SHIFT_MS) // 1000
    print(f"Calculated max_shift_steps={max_shift}")
    assert vocabulary.shift_id(1) == vocabulary.EVENT_RANGES[0].min_id
    assert vocabulary.shift_id(max_shift) == vocabulary.EVENT_RANGES[0].max_id
    with pytest.raises(ValueError):
        vocabulary.shift_id(0)


def test_decode_roundtrip():
    token = vocabulary.pitch_id(60)
    name, value = vocabulary.decode_event(token)
    print(f"decode_event({token}) -> (name={name}, value={value})")
    assert name == "pitch"
    assert value == 60 - 21


if __name__ == "__main__":
    test_pitch_ids()
    test_shift_ids()
    test_decode_roundtrip()
    
