#!/usr/bin/env python3
"""Reusable test suite template for functional model implementations.

This template provides a standardized test suite that works with any
HuggingFace model and its functional implementation.

Usage:
    from test_template import run_test_suite

    results = run_test_suite(
        model_name="Qwen/Qwen2.5-0.5B",
        functional_forward=my_forward,
        weights=weights,
        device="cuda:0",
    )

Or subclass FunctionalModelTestSuite for custom tests.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class TestResult:
    """Result of a single test."""

    name: str
    passed: bool
    max_diff: float
    details: dict | None = None

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"{self.name}: max_diff={self.max_diff:.2e} [{status}]"


@dataclass
class TestSuiteResult:
    """Result of running the full test suite."""

    total: int
    passed: int
    failed: int
    results: list[TestResult]

    @property
    def all_passed(self) -> bool:
        return self.failed == 0

    def __str__(self) -> str:
        return f"{self.passed}/{self.total} tests passed"


class FunctionalModelTestSuite:
    """Test suite for functional model implementations.

    Subclass this and override get_functional_forward() to test your model.
    """

    def __init__(
        self,
        model_name: str,
        functional_forward: Callable,
        weights: dict[str, Tensor],
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> None:
        self.model_name = model_name
        self.functional_forward = functional_forward
        self.weights = weights
        self.device = device
        self.dtype = dtype
        self.rtol = rtol
        self.atol = atol

        self.hf_model = None
        self.results: list[TestResult] = []

    def setup(self) -> None:
        """Load HF model. Called once before tests."""
        from transformers import AutoModelForCausalLM

        print(f"Loading {self.model_name}...")
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
        )
        self.hf_model.eval()

    def teardown(self) -> None:
        """Cleanup. Called after tests."""
        del self.hf_model
        self.hf_model = None
        torch.cuda.empty_cache()

    def run_test(
        self,
        name: str,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        custom_rtol: float | None = None,
        custom_atol: float | None = None,
    ) -> TestResult:
        """Run a single comparison test."""
        rtol = custom_rtol or self.rtol
        atol = custom_atol or self.atol

        with torch.no_grad():
            kwargs = {}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask

            hf_logits = self.hf_model(input_ids, **kwargs).logits
            func_logits = self.functional_forward(input_ids, self.weights, **kwargs)

        matches = torch.allclose(hf_logits, func_logits, rtol=rtol, atol=atol)
        max_diff = (hf_logits - func_logits).abs().max().item()

        result = TestResult(
            name=name,
            passed=matches,
            max_diff=max_diff,
            details={
                "shape": tuple(hf_logits.shape),
                "rtol": rtol,
                "atol": atol,
            },
        )
        self.results.append(result)
        return result

    def run_mask_test(
        self,
        name: str,
        input_ids: Tensor,
        attention_mask: Tensor,
        diff_threshold: float = 1e-4,
    ) -> TestResult:
        """Run test with attention mask, checking only non-padded positions."""
        with torch.no_grad():
            hf_logits = self.hf_model(input_ids, attention_mask=attention_mask).logits
            func_logits = self.functional_forward(
                input_ids, self.weights, attention_mask=attention_mask
            )

        # Calculate diff only at non-padded positions
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hf_logits)
        diff = (hf_logits - func_logits).abs()
        diff_masked = torch.where(mask_expanded == 1, diff, torch.zeros_like(diff))
        max_diff = diff_masked.max().item()

        passed = max_diff < diff_threshold

        result = TestResult(
            name=name,
            passed=passed,
            max_diff=max_diff,
            details={
                "shape": tuple(hf_logits.shape),
                "threshold": diff_threshold,
            },
        )
        self.results.append(result)
        return result

    # === Standard Test Categories ===

    def test_sequence_lengths(self, seq_lengths: list[int] | None = None) -> None:
        """Test various sequence lengths."""
        if seq_lengths is None:
            seq_lengths = [1, 4, 16, 32, 64, 128]

        print("\n### Sequence Length Tests ###")
        for seq_len in seq_lengths:
            input_ids = torch.randint(1, 1000, (1, seq_len), device=self.device)
            result = self.run_test(f"seq_len={seq_len:3d}", input_ids)
            print(f"  {result}")

    def test_batch_sizes(self, batch_sizes: list[int] | None = None, seq_len: int = 16) -> None:
        """Test various batch sizes."""
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]

        print("\n### Batch Size Tests ###")
        for batch in batch_sizes:
            input_ids = torch.randint(1, 1000, (batch, seq_len), device=self.device)
            result = self.run_test(f"batch={batch}, seq={seq_len}", input_ids)
            print(f"  {result}")

    def test_edge_cases(self) -> None:
        """Test edge cases: single token, repeated tokens, etc."""
        print("\n### Edge Cases ###")

        # Single token
        input_ids = torch.tensor([[42]], device=self.device)
        result = self.run_test("single_token", input_ids)
        print(f"  {result}")

        # Repeated tokens
        input_ids = torch.full((1, 32), 100, device=self.device)
        result = self.run_test("repeated_token", input_ids)
        print(f"  {result}")

        # Sequential tokens
        input_ids = torch.arange(1, 65, device=self.device).unsqueeze(0)
        result = self.run_test("sequential_tokens", input_ids)
        print(f"  {result}")

        # High token IDs (near vocab boundary)
        vocab_size = self.weights.get(
            "model.embed_tokens.weight", self.weights.get("transformer.wte.weight")
        ).shape[0]
        high_start = max(1, vocab_size - 1000)
        input_ids = torch.randint(high_start, vocab_size, (1, 16), device=self.device)
        result = self.run_test("high_token_ids", input_ids)
        print(f"  {result}")

    def test_attention_masks(self) -> None:
        """Test with various attention mask patterns."""
        print("\n### Attention Mask Tests ###")

        # All ones (no padding)
        input_ids = torch.randint(1, 1000, (1, 16), device=self.device)
        attention_mask = torch.ones_like(input_ids)
        result = self.run_mask_test("all_ones", input_ids, attention_mask)
        print(f"  {result}")

        # Left padding
        input_ids = torch.randint(1, 1000, (2, 16), device=self.device)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[0, :4] = 0
        attention_mask[1, :2] = 0
        result = self.run_mask_test("left_padding", input_ids, attention_mask)
        print(f"  {result}")

        # Variable length
        input_ids = torch.randint(1, 1000, (4, 32), device=self.device)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[0, :8] = 0
        attention_mask[1, :16] = 0
        attention_mask[2, :24] = 0
        attention_mask[3, :4] = 0
        result = self.run_mask_test("variable_length", input_ids, attention_mask)
        print(f"  {result}")

    def test_dtypes(self, dtypes: list[tuple[str, torch.dtype]] | None = None) -> None:
        """Test different dtypes by reloading model."""
        if dtypes is None:
            dtypes = [
                ("bf16", torch.bfloat16),
                ("fp16", torch.float16),
                ("fp32", torch.float32),
            ]

        print("\n### Dtype Tests ###")

        from transformers import AutoModelForCausalLM

        for dtype_name, dtype in dtypes:
            # Load model with specified dtype
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=self.device,
            )
            model.eval()
            weights = dict(model.state_dict())

            input_ids = torch.randint(1, 1000, (1, 16), device=self.device)

            with torch.no_grad():
                hf_logits = model(input_ids).logits
                func_logits = self.functional_forward(input_ids, weights)

            # Looser tolerance for fp16/fp32
            rtol = atol = (
                1e-3 if dtype == torch.float16 else (1e-4 if dtype == torch.float32 else 1e-5)
            )

            matches = torch.allclose(hf_logits, func_logits, rtol=rtol, atol=atol)
            max_diff = (hf_logits - func_logits).abs().max().item()

            result = TestResult(name=dtype_name, passed=matches, max_diff=max_diff)
            self.results.append(result)
            print(f"  {result}")

            del model, weights
            torch.cuda.empty_cache()

    def run_all(self) -> TestSuiteResult:
        """Run all standard tests."""
        self.setup()

        try:
            self.test_sequence_lengths()
            self.test_batch_sizes()
            self.test_edge_cases()
            self.test_dtypes()
            self.test_attention_masks()
        finally:
            self.teardown()

        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        return TestSuiteResult(
            total=len(self.results),
            passed=passed,
            failed=failed,
            results=self.results,
        )

    def print_summary(self, result: TestSuiteResult) -> None:
        """Print test summary."""
        print("\n" + "=" * 60)
        if result.all_passed:
            print(f"ALL {result.total} TESTS PASSED!")
            print("Functional implementation is numerically identical to HF model.")
        else:
            print(f"SOME TESTS FAILED: {result.passed}/{result.total} passed")
            print("\nFailed tests:")
            for r in result.results:
                if not r.passed:
                    print(f"  {r}")


def run_test_suite(
    model_name: str,
    functional_forward: Callable,
    weights: dict[str, Tensor],
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> TestSuiteResult:
    """Convenience function to run the full test suite.

    Args:
        model_name: HuggingFace model name
        functional_forward: Your functional forward function
        weights: Model weights dict
        device: Device to run on
        dtype: Default dtype

    Returns:
        TestSuiteResult with all test results
    """
    print("=" * 60)
    print(f"{model_name} Functional Implementation Test")
    print("=" * 60)

    suite = FunctionalModelTestSuite(
        model_name=model_name,
        functional_forward=functional_forward,
        weights=weights,
        device=device,
        dtype=dtype,
    )

    result = suite.run_all()
    suite.print_summary(result)

    return result


# CLI support
if __name__ == "__main__":
    print("This is a template module. Import and use run_test_suite() or FunctionalModelTestSuite.")
    print("\nExample:")
    print("  from test_template import run_test_suite")
    print("  from my_model import my_forward")
    print("  ")
    print("  weights = dict(hf_model.state_dict())")
    print("  results = run_test_suite('model/name', my_forward, weights)")
