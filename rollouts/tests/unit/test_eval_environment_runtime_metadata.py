from dataclasses import replace

import pytest

from rollouts.agents import Actor, AgentState
from rollouts.core import EvalConfig, Metric, Score
from rollouts.dtypes import (
    ChatCompletion,
    Choice,
    LLMCallEnd,
    Logprob,
    Logprobs,
    Message,
    StopReason,
    StreamChunk,
    ToolCallError,
    Trajectory,
    Usage,
)
from rollouts.eval.native import (
    EvalRuntime,
    _AgentRunResult,
    _build_semantic_trace,
    compute_summary_metrics,
    evaluate_sample,
)
from rollouts.training.scoring import FunctionScorer
from rollouts.training.types import DatasetRow, RowAttempt, ScoringContext, Status


class _FakeEnvironment:
    def __init__(
        self,
        *,
        best_speedup: float,
        has_correct_kernel: bool,
        finalize_best_speedup: float | None = None,
        finalize_has_correct_kernel: bool | None = None,
    ) -> None:
        self.best_speedup = best_speedup
        self.has_correct_kernel = has_correct_kernel
        self.finalize_calls = 0
        self.finalize_best_speedup = finalize_best_speedup
        self.finalize_has_correct_kernel = finalize_has_correct_kernel

    def get_tools(self) -> list[object]:
        return []

    async def serialize(self) -> dict:
        return {
            "best_speedup": self.best_speedup,
            "has_correct_kernel": self.has_correct_kernel,
        }

    def get_runtime_metadata(self) -> dict:
        return {
            "best_speedup": self.best_speedup,
            "has_correct_kernel": self.has_correct_kernel,
            "turn_history": [],
        }

    async def finalize_attempt(self, **_: object) -> None:
        self.finalize_calls += 1
        if self.finalize_best_speedup is not None:
            self.best_speedup = self.finalize_best_speedup
        if self.finalize_has_correct_kernel is not None:
            self.has_correct_kernel = self.finalize_has_correct_kernel


class _ContextualScorer:
    async def score(self, result: RowAttempt, context: ScoringContext) -> Score:
        del result
        metadata = context.environment.get_runtime_metadata()
        correct = 1.0 if metadata["has_correct_kernel"] else 0.0
        speedup = float(metadata["best_speedup"])
        return Score(
            metrics=(
                Metric("reward", speedup, weight=1.0),
                Metric("correct", correct, weight=0.0),
                Metric("speedup", speedup, weight=0.0),
            )
        )


@pytest.mark.trio
async def test_evaluate_sample_scores_against_final_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_env = _FakeEnvironment(best_speedup=0.0, has_correct_kernel=False)
    final_env = _FakeEnvironment(best_speedup=1.25, has_correct_kernel=True)

    stale_trajectory = Trajectory(
        messages=[Message(role="assistant", content="done")],
        metadata={"best_speedup": 0.0, "has_correct_kernel": False},
    )
    final_state = AgentState(
        actor=Actor(trajectory=stale_trajectory, endpoint=None, tools=[]),
        environment=final_env,
    )

    async def _fake_run_agent_with_error_handling(
        initial_state: AgentState,
        run_config: object,
        sample_id: str,
    ) -> _AgentRunResult:
        del run_config, sample_id
        return _AgentRunResult(
            states=[initial_state, replace(final_state, turn_idx=1)],
            final_trajectory=stale_trajectory,
        )

    monkeypatch.setattr(
        "rollouts.eval.native._run_agent_with_error_handling",
        _fake_run_agent_with_error_handling,
    )

    config = EvalConfig(
        endpoint=None,
        prepare_messages=lambda _: [Message(role="user", content="hi")],
        scorer=_ContextualScorer(),
        verbose=False,
        show_progress=False,
    )
    runtime = EvalRuntime(config=config)

    result = await evaluate_sample(
        sample_data={"name": "sample"},
        sample_id="sample_0000",
        runtime=runtime,
        environment=initial_env,
    )

    assert result.score is not None
    assert result.reward == 1.25
    assert result.metadata["has_correct_kernel"] is True
    assert result.metadata["best_speedup"] == 1.25
    assert result.environment_state is not None
    assert result.environment_state["has_correct_kernel"] is True


@pytest.mark.trio
async def test_evaluate_sample_finalizes_environment_before_serialization_and_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_env = _FakeEnvironment(best_speedup=0.0, has_correct_kernel=False)
    final_env = _FakeEnvironment(
        best_speedup=0.0,
        has_correct_kernel=False,
        finalize_best_speedup=2.0,
        finalize_has_correct_kernel=True,
    )

    stale_trajectory = Trajectory(
        messages=[Message(role="assistant", content="done")],
        metadata={},
    )
    final_state = AgentState(
        actor=Actor(trajectory=stale_trajectory, endpoint=None, tools=[]),
        environment=final_env,
    )

    async def _fake_run_agent_with_error_handling(
        initial_state: AgentState,
        run_config: object,
        sample_id: str,
    ) -> _AgentRunResult:
        del run_config, sample_id
        return _AgentRunResult(
            states=[initial_state, replace(final_state, turn_idx=1)],
            final_trajectory=stale_trajectory,
        )

    monkeypatch.setattr(
        "rollouts.eval.native._run_agent_with_error_handling",
        _fake_run_agent_with_error_handling,
    )

    config = EvalConfig(
        endpoint=None,
        prepare_messages=lambda _: [Message(role="user", content="hi")],
        scorer=_ContextualScorer(),
        verbose=False,
        show_progress=False,
    )
    runtime = EvalRuntime(config=config)

    result = await evaluate_sample(
        sample_data={"name": "sample"},
        sample_id="sample_0000",
        runtime=runtime,
        environment=initial_env,
    )

    assert final_env.finalize_calls == 1
    assert result.reward == 2.0
    assert result.metadata["has_correct_kernel"] is True
    assert result.metadata["best_speedup"] == 2.0
    assert result.environment_state == {
        "best_speedup": 2.0,
        "has_correct_kernel": True,
    }


@pytest.mark.trio
async def test_evaluate_sample_marks_aborted_runs_honestly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _FakeEnvironment(best_speedup=0.0, has_correct_kernel=False)
    aborted_trajectory = Trajectory(
        messages=[Message(role="assistant", content="partial")],
        metadata={},
    )
    aborted_state = AgentState(
        actor=Actor(trajectory=aborted_trajectory, endpoint=None, tools=[]),
        environment=environment,
        stop=StopReason.ABORTED,
    )

    async def _fake_run_agent_with_error_handling(
        initial_state: AgentState,
        run_config: object,
        sample_id: str,
    ) -> _AgentRunResult:
        del run_config, sample_id
        return _AgentRunResult(
            states=[initial_state, replace(aborted_state, turn_idx=0)],
            final_trajectory=aborted_trajectory,
        )

    async def _fail_if_scored(result: RowAttempt, context: ScoringContext) -> Score:
        del result, context
        raise AssertionError("aborted samples should not be scored")

    monkeypatch.setattr(
        "rollouts.eval.native._run_agent_with_error_handling",
        _fake_run_agent_with_error_handling,
    )

    config = EvalConfig(
        endpoint=None,
        prepare_messages=lambda _: [Message(role="user", content="hi")],
        scorer=_fail_if_scored,
        verbose=False,
        show_progress=False,
    )
    runtime = EvalRuntime(config=config)

    result = await evaluate_sample(
        sample_data={"name": "sample"},
        sample_id="sample_0000",
        runtime=runtime,
        environment=environment,
    )

    assert result.metadata["status"] == "aborted"
    assert result.status == Status.ABORTED
    assert result.score is None
    assert result.reward == 0.0


def test_build_semantic_trace_from_trajectory_completion() -> None:
    trajectory = Trajectory(
        completions=[
            ChatCompletion(
                id="cmpl-1",
                object="chat.completion",
                created=123,
                model="stub-model",
                usage=Usage(input_tokens=4, output_tokens=3),
                prompt_token_ids=(1, 2, 3, 4),
                choices=[
                    Choice(
                        0,
                        Message(role="assistant", content="<answer>stub</answer>"),
                        "stop",
                        logprobs=Logprobs(
                            content=[
                                Logprob(
                                    token="<answer>",
                                    logprob=-0.01,
                                    token_id=11,
                                    top_candidates=[
                                        {
                                            "token": "<answer>",
                                            "logprob": -0.01,
                                            "token_id": 11,
                                            "bytes": [],
                                        },
                                        {
                                            "token": "<think>",
                                            "logprob": -4.2,
                                            "token_id": 12,
                                            "bytes": [],
                                        },
                                    ],
                                ),
                                Logprob(token="stub", logprob=-0.02, token_id=13),
                            ]
                        ),
                        token_ids=(11, 13),
                    )
                ],
            )
        ]
    )

    semantic_trace = _build_semantic_trace(trajectory)

    assert semantic_trace is not None
    assert semantic_trace["completion_count"] == 1
    call = semantic_trace["llm_calls"][0]
    assert call["prompt_token_ids"] == [1, 2, 3, 4]
    assert call["output_token_ids"] == [11, 13]
    assert call["output_tokens"] == ["<answer>", "stub"]
    assert call["output_token_logprobs"] == [-0.01, -0.02]
    assert call["output_top_logprobs"][0][0]["token"] == "<answer>"
    assert call["finish_reason"] == "stop"


def test_compute_summary_metrics_excludes_aborted_from_completion_and_success() -> None:
    aborted = RowAttempt(
        attempt_id="aborted",
        status=Status.ABORTED,
        metadata={"status": "aborted", "turns_used": 0, "total_tokens": 10},
    )
    failed = RowAttempt(
        attempt_id="failed",
        status=Status.COMPLETED,
        metadata={
            "status": "failed",
            "error": "ValueError: boom",
            "turns_used": 1,
            "total_tokens": 20,
        },
    )
    success = RowAttempt(
        attempt_id="success",
        status=Status.COMPLETED,
        metadata={"status": "success", "turns_used": 2, "total_tokens": 30},
    )
    success.score = Score(metrics=(Metric("reward", 1.0, weight=1.0),))

    summary = compute_summary_metrics([aborted, failed, success])

    assert summary["aborted_samples"] == 1
    assert summary["failed_samples"] == 1
    assert summary["successful_samples"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["completion_rate"] == 2 / 3


@pytest.mark.trio
async def test_evaluate_sample_preserves_llm_runtime_metrics() -> None:
    async def _attempt_executor(
        sample_data: dict[str, str],
        sample_id: str,
        environment: object,
        run_config: object,
    ) -> RowAttempt:
        del environment
        await run_config.on_chunk(  # type: ignore[attr-defined]
            LLMCallEnd(
                duration_ms=123.4,
                ttft_ms=45.6,
                provider="openai",
                model="test-model",
                tokens_in=12,
                tokens_out=7,
                status="success",
            )
        )
        return RowAttempt(
            attempt_id=sample_id,
            problem=DatasetRow(problem_id=sample_id, payload=sample_data),
            trajectory=Trajectory(messages=[Message(role="assistant", content="ok")]),
            metadata={"status": "success"},
        )

    config = EvalConfig(
        endpoint=None,
        prepare_messages=lambda _: [],
        attempt_executor=_attempt_executor,
        scorer=FunctionScorer(lambda _sample, _context: Score(metrics=())),
        verbose=False,
        show_progress=False,
    )
    runtime = EvalRuntime(config=config)

    result = await evaluate_sample(
        sample_data={"prompt": "hello"},
        sample_id="sample_0000",
        runtime=runtime,
    )

    assert result.metadata["llm_call_count"] == 1
    assert result.metadata["llm_tokens_in_total"] == 12
    assert result.metadata["llm_tokens_out_total"] == 7
    assert result.metadata["llm_duration_ms_mean"] == 123.4
    assert result.metadata["llm_ttft_ms_mean"] == 45.6
    llm_metrics = result.metadata["llm_call_metrics"]
    assert isinstance(llm_metrics, list)
    assert llm_metrics[0]["model"] == "test-model"


@pytest.mark.trio
async def test_evaluate_sample_tracks_tool_call_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted_events: list[tuple[str, dict[str, object]]] = []

    def _capture_eval_event(event: str, **data: object) -> None:
        emitted_events.append((event, data))

    monkeypatch.setattr("rollouts.eval.native._emit_eval_event", _capture_eval_event)

    final_trajectory = Trajectory(messages=[Message(role="assistant", content="done")])
    final_state = AgentState(
        actor=Actor(trajectory=final_trajectory, endpoint=None, tools=[]),
        environment=None,
    )

    async def _fake_run_agent_with_error_handling(
        initial_state: AgentState,
        run_config: object,
        sample_id: str,
    ) -> _AgentRunResult:
        del sample_id
        await run_config.on_chunk(  # type: ignore[attr-defined]
            StreamChunk("turn_start", {"turn": 0})
        )
        await run_config.on_chunk(  # type: ignore[attr-defined]
            ToolCallError(
                content_index=0,
                tool_call_id="tc-1",
                tool_name="write_file",
                error="Invalid JSON arguments: missing }",
                raw_arguments='{"path": "/tmp/x"',
            )
        )
        await run_config.on_chunk(  # type: ignore[attr-defined]
            StreamChunk(
                "tool_call_dispatch",
                {
                    "turn": 0,
                    "tool_call_id": "tc-1",
                    "tool_name": "write_file",
                    "action": "parse_error",
                    "error": "Invalid JSON arguments: missing }",
                },
            )
        )
        await run_config.on_chunk(  # type: ignore[attr-defined]
            StreamChunk(
                "tool_call_dispatch",
                {
                    "turn": 0,
                    "tool_call_id": "tc-2",
                    "tool_name": "write_file",
                    "action": "schema_error",
                    "error": "Arguments failed schema validation: 'content' is a required property",
                },
            )
        )
        return _AgentRunResult(
            states=[initial_state, replace(final_state, turn_idx=1)],
            final_trajectory=final_trajectory,
        )

    monkeypatch.setattr(
        "rollouts.eval.native._run_agent_with_error_handling",
        _fake_run_agent_with_error_handling,
    )

    config = EvalConfig(
        endpoint=None,
        prepare_messages=lambda _: [Message(role="user", content="hi")],
        scorer=FunctionScorer(lambda _sample, _context: Score(metrics=())),
        verbose=False,
        show_progress=False,
    )
    runtime = EvalRuntime(config=config)

    result = await evaluate_sample(
        sample_data={"prompt": "hello"},
        sample_id="sample_0000",
        runtime=runtime,
    )

    assert result.metadata["tool_call_error_count"] == 1
    assert result.metadata["tool_dispatch_count"] == 2
    assert result.metadata["tool_call_parse_error_count"] == 1
    assert result.metadata["tool_call_schema_error_count"] == 1
    assert result.metadata["tool_call_execute_count"] == 0
    assert result.metadata["tool_call_error_metrics"][0]["tool_name"] == "write_file"
    assert result.metadata["tool_dispatch_metrics"][1]["action"] == "schema_error"
    assert any(event == "tool_call_error" for event, _ in emitted_events)
    assert any(
        event == "tool_call_dispatch" and data.get("action") == "schema_error"
        for event, data in emitted_events
    )


def test_compute_summary_metrics_includes_llm_runtime_telemetry() -> None:
    first = RowAttempt(
        attempt_id="sample-1",
        status=Status.COMPLETED,
        metadata={
            "status": "success",
            "turns_used": 1,
            "total_tokens": 10,
            "duration_seconds": 2.0,
            "llm_call_metrics": [
                {
                    "duration_ms": 100.0,
                    "ttft_ms": 40.0,
                    "tokens_in": 8,
                    "tokens_out": 4,
                    "status": "success",
                }
            ],
            "tool_call_error_metrics": [],
            "tool_dispatch_metrics": [],
            "tool_execution_metrics": [],
        },
    )
    second = RowAttempt(
        attempt_id="sample-2",
        status=Status.COMPLETED,
        metadata={
            "status": "failed",
            "error": "ValueError: boom",
            "turns_used": 1,
            "total_tokens": 12,
            "duration_seconds": 4.0,
            "llm_call_metrics": [
                {
                    "duration_ms": 200.0,
                    "ttft_ms": 80.0,
                    "tokens_in": 16,
                    "tokens_out": 6,
                    "status": "error",
                }
            ],
            "tool_call_error_metrics": [
                {
                    "tool_call_id": "tc-1",
                    "tool_name": "write_file",
                    "error": "Invalid JSON arguments",
                }
            ],
            "tool_dispatch_metrics": [
                {
                    "tool_call_id": "tc-1",
                    "tool_name": "write_file",
                    "action": "parse_error",
                },
                {
                    "tool_call_id": "tc-2",
                    "tool_name": "write_file",
                    "action": "schema_error",
                },
            ],
            "tool_execution_metrics": [
                {
                    "duration_ms": 55.0,
                    "status": "success",
                }
            ],
        },
    )

    summary = compute_summary_metrics([first, second])

    assert summary["llm_call_count_total"] == 2
    assert summary["llm_call_error_count"] == 1
    assert summary["llm_tokens_in_total"] == 24
    assert summary["llm_tokens_out_total"] == 10
    assert summary["llm_duration_ms_mean"] == 150.0
    assert summary["llm_ttft_ms_p50"] == 60.0
    assert summary["sample_duration_seconds_mean"] == 3.0
    assert summary["tool_execution_count_total"] == 1
    assert summary["tool_call_error_count_total"] == 1
    assert summary["tool_call_dispatch_total"] == 2
    assert summary["tool_call_parse_error_total"] == 1
    assert summary["tool_call_schema_error_total"] == 1
    assert summary["llm_tpot_ms_p50"] == 22.0
    assert summary["llm_itl_ms_p50"] == 22.0


def test_compute_summary_metrics_includes_output_tokens_per_min_per_gpu() -> None:
    first = RowAttempt(
        attempt_id="sample-1",
        status=Status.COMPLETED,
        metadata={
            "status": "success",
            "turns_used": 1,
            "total_tokens": 10,
            "llm_call_metrics": [
                {
                    "duration_ms": 100.0,
                    "ttft_ms": 40.0,
                    "tokens_in": 8,
                    "tokens_out": 40,
                    "status": "success",
                }
            ],
            "tool_execution_metrics": [],
        },
    )
    second = RowAttempt(
        attempt_id="sample-2",
        status=Status.COMPLETED,
        metadata={
            "status": "success",
            "turns_used": 1,
            "total_tokens": 12,
            "llm_call_metrics": [
                {
                    "duration_ms": 200.0,
                    "ttft_ms": 80.0,
                    "tokens_in": 16,
                    "tokens_out": 20,
                    "status": "success",
                }
            ],
            "tool_execution_metrics": [],
        },
    )

    summary = compute_summary_metrics(
        [first, second],
        wall_time_seconds=30.0,
        gpu_count=2,
    )

    assert summary["total_output_tokens_per_sec"] == 2.0
    assert summary["output_tokens_per_min_per_gpu"] == 60.0


def test_compute_summary_metrics_respects_distribution_percentiles() -> None:
    result = RowAttempt(
        attempt_id="sample-1",
        status=Status.COMPLETED,
        metadata={
            "status": "success",
            "turns_used": 1,
            "total_tokens": 10,
            "llm_call_metrics": [
                {
                    "duration_ms": 100.0,
                    "ttft_ms": 10.0,
                    "tokens_in": 8,
                    "tokens_out": 10,
                    "status": "success",
                },
                {
                    "duration_ms": 200.0,
                    "ttft_ms": 20.0,
                    "tokens_in": 8,
                    "tokens_out": 10,
                    "status": "success",
                },
            ],
            "tool_execution_metrics": [],
        },
    )

    summary = compute_summary_metrics(
        [result],
        distribution_percentiles={
            "llm_ttft_ms": (90, 99),
            "llm_tpot_ms": (50, 99),
            "llm_itl_ms": (90, 95, 99),
        },
    )

    assert "llm_ttft_ms_p50" not in summary
    assert summary["llm_ttft_ms_p90"] == 19.0
    assert summary["llm_ttft_ms_p99"] == 19.9
    assert summary["llm_tpot_ms_p50"] == 15.0
    assert summary["llm_tpot_ms_p99"] == 19.9
    assert summary["llm_itl_ms_p90"] == 19.0
    assert summary["llm_itl_ms_p95"] == 19.5
    assert summary["llm_itl_ms_p99"] == 19.9
