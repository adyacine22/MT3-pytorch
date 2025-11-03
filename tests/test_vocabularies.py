import unittest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import vocabularies


class TestVocabularies(unittest.TestCase):

    def test_encode_decode(self):
        codec = vocabularies.build_codec(vocabularies.VocabularyConfig())
        vocab = vocabularies.GenericTokenVocabulary(codec.num_classes)

        events = [
            vocabularies.Event(type="pitch", value=60),
            vocabularies.Event(type="shift", value=5),
            vocabularies.Event(type="pitch", value=62),
        ]

        encoded_events = [codec.encode_event(e) for e in events]
        encoded_tokens = vocab.encode(encoded_events)

        decoded_tokens = vocab.decode(encoded_tokens)
        decoded_events = [codec.decode_event_index(t) for t in decoded_tokens]

        self.assertEqual(events, decoded_events)

    def test_velocity_quantization(self):
        self.assertEqual(0, vocabularies.velocity_to_bin(0, num_velocity_bins=1))
        self.assertEqual(0, vocabularies.velocity_to_bin(0, num_velocity_bins=127))
        self.assertEqual(0, vocabularies.bin_to_velocity(0, num_velocity_bins=1))
        self.assertEqual(0, vocabularies.bin_to_velocity(0, num_velocity_bins=127))

        self.assertEqual(
            1,
            vocabularies.velocity_to_bin(
                vocabularies.bin_to_velocity(1, num_velocity_bins=1),
                num_velocity_bins=1,
            ),
        )

        for velocity_bin in range(1, 128):
            self.assertEqual(
                velocity_bin,
                vocabularies.velocity_to_bin(
                    vocabularies.bin_to_velocity(velocity_bin, num_velocity_bins=127),
                    num_velocity_bins=127,
                ),
            )


if __name__ == "__main__":
    unittest.main()
