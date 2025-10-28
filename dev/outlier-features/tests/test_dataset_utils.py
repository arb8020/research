import sys
import unittest
from pathlib import Path

from datasets import disable_caching
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from dataset_utils import _effective_sequence_length


disable_caching()


class EffectiveSequenceLengthTests(unittest.TestCase):
    def test_clamps_to_tokenizer_max_length(self):
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", trust_remote_code=False)
        self.assertEqual(
            _effective_sequence_length(2048, tokenizer),
            tokenizer.model_max_length,
        )

    def test_leaves_length_when_max_is_large(self):
        class DummyTokenizer:
            model_max_length = 10**12

        dummy = DummyTokenizer()
        self.assertEqual(_effective_sequence_length(2048, dummy), 2048)


if __name__ == "__main__":
    unittest.main()
