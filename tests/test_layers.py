import unittest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import Layers
from model import Attention
from model import Mask
from config.T5config import Magenta_T5Config


class TestLayers(unittest.TestCase):

    def test_make_attention_mask(self):
        q = torch.tensor([[1, 1, 0]])
        k = torch.tensor([[1, 1, 0]])
        mask = Mask.make_attention_mask(q, k)
        # This is a simplified test. A proper test would check the exact mask values.
        self.assertEqual(mask.shape, (1, 1, 3, 3))

    def test_make_causal_mask(self):
        x = torch.tensor([[1, 1, 1]])
        mask = Mask.make_causal_mask(x)
        # This is a simplified test. A proper test would check the exact mask values.
        self.assertEqual(mask.shape, (1, 1, 3, 3))

    def test_attention(self):
        config = Magenta_T5Config()
        attention = Attention.Multi_Head_Attention(config.num_heads, config.head_dim)
        q = torch.randn(2, 5, config.emb_dim)
        kv = torch.randn(2, 5, config.emb_dim)
        output, cache = attention(q, kv)
        self.assertEqual(output.shape, (2, 5, config.emb_dim))
        self.assertIsNone(cache)

    def test_attention_cache(self):
        config = Magenta_T5Config()
        attention = Attention.Multi_Head_Attention(config.num_heads, config.head_dim)
        q = torch.randn(2, 1, config.emb_dim)
        kv = torch.randn(2, 1, config.emb_dim)
        output, cache = attention(q, kv, decode=True)
        self.assertEqual(output.shape, (2, 1, config.emb_dim))
        self.assertIn("cached_key", cache)
        self.assertIn("cached_value", cache)


if __name__ == "__main__":
    unittest.main()
