import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from nnsight import LanguageModel

from extract_activations import extract_activations_optimized


class ActivationExtractionTest(unittest.TestCase):
    MODELS = (
        ("openai-community/gpt2", None),
        ("facebook/opt-125m", [0]),
    )

    def test_activation_shapes_cpu(self):
        texts = ["Hello world! This is a tiny test input."]
        for model_name, layer_selection in self.MODELS:
            with self.subTest(model=model_name):
                with TemporaryDirectory() as tmpdir:
                    llm = LanguageModel(
                        model_name,
                        device_map="cpu",
                        torch_dtype=torch.float32,
                    )

                    try:
                        run_dir, metadata = extract_activations_optimized(
                            llm=llm,
                            texts=texts,
                            layers=layer_selection,
                            save_dir=tmpdir,
                            chunk_size=None,
                        )
                    finally:
                        del llm
                        torch.cuda.empty_cache()

                    run_path = Path(run_dir)
                    activation_files = sorted(run_path.glob("layer_*_activations.pt"))
                    self.assertTrue(activation_files, "expected activation tensors to be written")

                    for activation_file in activation_files:
                        tensor = torch.load(activation_file)
                        self.assertEqual(tensor.dim(), 3, f"Expected 3D tensor, got {tensor.shape}")
                        self.assertEqual(tensor.shape[0], len(texts), "batch dimension mismatch")

                    self.assertIn("layers_extracted", metadata)
                    if layer_selection is not None:
                        self.assertEqual(metadata["layers_extracted"], layer_selection)
                    else:
                        self.assertGreaterEqual(len(metadata["layers_extracted"]), 1)


if __name__ == "__main__":
    unittest.main()
