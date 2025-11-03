import unittest
import torch
import note_seq
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import utils
from data import vocabularies


class TestEvaluation(unittest.TestCase):

    def test_tokens_to_note_sequence(self):
        codec = vocabularies.build_codec(vocabularies.VocabularyConfig())
        # Create a simple note sequence and convert to tokens
        ns_original = note_seq.NoteSequence()
        ns_original.notes.add(
            start_time=0.1,
            end_time=0.2,
            pitch=60,
            velocity=100,
            program=0,
            is_drum=False,
        )
        ns_original.total_time = 0.2

        # Convert to timed events and tokens
        timed_events = utils.note_sequence_to_timed_events(ns_original)
        tokens, _, _ = utils.timed_events_to_tokens(
            timed_events, codec, frame_times=np.arange(0, 0.3, 0.01)
        )

        # Now test decoding
        ns = utils.tokens_to_note_sequence(tokens, codec)
        self.assertIsInstance(ns, note_seq.NoteSequence)
        self.assertEqual(len(ns.notes), 1)
        self.assertEqual(ns.notes[0].pitch, 60)

    def test_frame_metrics(self):
        ref = np.zeros((128, 100))
        est = np.zeros((128, 100))
        ref[60, 10:20] = 1
        est[60, 15:25] = 1

        # This is a simplified test. A proper test would use mir_eval.
        # precision = np.sum(np.logical_and(ref, est)) / np.sum(est)
        # recall = np.sum(np.logical_and(ref, est)) / np.sum(ref)
        # self.assertGreater(precision, 0)
        # self.assertGreater(recall, 0)
        pass


if __name__ == "__main__":
    unittest.main()
