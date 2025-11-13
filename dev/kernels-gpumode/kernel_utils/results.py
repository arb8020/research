"""Test result data structures for NVFP4 kernel testing.

Provides structured results for correctness and performance tests,
enabling JSON export and result aggregation.
"""
from dataclasses import dataclass, asdict
from typing import Optional
import json
from pathlib import Path


@dataclass
class CorrectnessResult:
    """Result of a single correctness test."""

    test_name: str
    backend_name: str
    is_correct: bool
    test_params: str  # Serialized test parameters (m, k, l, seed)
    max_abs_error: Optional[float] = None
    max_rel_error: Optional[float] = None
    error_msg: Optional[str] = None
    error_type: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class PerformanceResult:
    """Result of a single performance test."""

    test_name: str
    backend_name: str
    successfully_ran: bool
    test_params: str  # Serialized test parameters
    avg_time_ms: float
    speedup: Optional[float] = None  # vs reference backend
    reference_time_ms: Optional[float] = None
    error_msg: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class BackendResults:
    """Aggregated results for a single backend."""

    backend_name: str
    correctness_tests: list[CorrectnessResult]
    performance_tests: list[PerformanceResult]

    @property
    def correctness_score(self) -> float:
        """Fraction of correctness tests passed (0.0 to 1.0)."""
        if not self.correctness_tests:
            return 0.0
        passed = sum(1 for r in self.correctness_tests if r.is_correct)
        return passed / len(self.correctness_tests)

    @property
    def avg_speedup(self) -> float:
        """Average speedup across performance tests."""
        if not self.performance_tests:
            return 0.0
        speedups = [r.speedup for r in self.performance_tests
                    if r.successfully_ran and r.speedup is not None]
        if not speedups:
            return 0.0
        return sum(speedups) / len(speedups)

    @property
    def geomean_speedup(self) -> float:
        """Geometric mean speedup (better for multiplicative metrics)."""
        if not self.performance_tests:
            return 0.0
        speedups = [r.speedup for r in self.performance_tests
                    if r.successfully_ran and r.speedup is not None and r.speedup > 0]
        if not speedups:
            return 0.0

        # Geometric mean: exp(mean(log(speedups)))
        import math
        log_sum = sum(math.log(s) for s in speedups)
        return math.exp(log_sum / len(speedups))

    @property
    def all_correct(self) -> bool:
        """True if all correctness tests passed."""
        return all(r.is_correct for r in self.correctness_tests)

    def summary(self) -> str:
        """Human-readable summary."""
        total_correct = len(self.correctness_tests)
        passed_correct = sum(1 for r in self.correctness_tests if r.is_correct)

        lines = [
            f"Backend: {self.backend_name}",
            f"  Correctness: {passed_correct}/{total_correct} passed ({self.correctness_score:.1%})",
            f"  Performance: {len(self.performance_tests)} tests",
        ]

        if self.performance_tests:
            lines.append(f"    Avg speedup: {self.avg_speedup:.2f}x")
            lines.append(f"    Geomean speedup: {self.geomean_speedup:.2f}x")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "backend_name": self.backend_name,
            "correctness_score": self.correctness_score,
            "all_correct": self.all_correct,
            "avg_speedup": self.avg_speedup,
            "geomean_speedup": self.geomean_speedup,
            "correctness_tests": [r.to_dict() for r in self.correctness_tests],
            "performance_tests": [r.to_dict() for r in self.performance_tests],
        }


@dataclass
class TestSuiteResults:
    """Complete results for all backends in a test suite."""

    suite_name: str
    backends: list[BackendResults]

    def get_backend(self, name: str) -> Optional[BackendResults]:
        """Get results for a specific backend."""
        for backend in self.backends:
            if backend.backend_name == name:
                return backend
        return None

    def summary_table(self) -> str:
        """Generate comparison table across backends."""
        if not self.backends:
            return "No results"

        lines = [
            f"Test Suite: {self.suite_name}",
            "=" * 80,
            f"{'Backend':<20} {'Correctness':<15} {'Avg Speedup':<15} {'Geomean Speedup':<15}",
            "-" * 80,
        ]

        for backend in self.backends:
            status = "✅" if backend.all_correct else "❌"
            lines.append(
                f"{backend.backend_name:<20} "
                f"{status} {backend.correctness_score:>5.1%}       "
                f"{backend.avg_speedup:>6.2f}x         "
                f"{backend.geomean_speedup:>6.2f}x"
            )

        lines.append("=" * 80)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "suite_name": self.suite_name,
            "backends": [b.to_dict() for b in self.backends],
        }

    def to_json(self, filepath: str | Path) -> None:
        """Save results to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str | Path) -> "TestSuiteResults":
        """Load results from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            TestSuiteResults instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        backends = []
        for b_data in data["backends"]:
            correctness = [CorrectnessResult(**r) for r in b_data["correctness_tests"]]
            performance = [PerformanceResult(**r) for r in b_data["performance_tests"]]
            backends.append(BackendResults(
                backend_name=b_data["backend_name"],
                correctness_tests=correctness,
                performance_tests=performance,
            ))

        return cls(
            suite_name=data["suite_name"],
            backends=backends,
        )
