#!/usr/bin/env python3
"""Rollout dataclass for corpus-proximity inference and evaluation."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping, Iterator, Protocol, runtime_checkable, Callable, Awaitable
import dacite

# ────────────────────────────── Core Message Types ──────────────────────────────

@dataclass(frozen=True)
class Message:
    """A single message in a conversation (user, assistant, system)."""
    role: str
    content: str


# ────────────────────────────── Completion Types ──────────────────────────────

@dataclass(frozen=True)
class Usage:
    """Token usage information from API response."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[Any] = None


@dataclass(frozen=True)
class Logprob:
    """Single token logprob information."""
    token: str
    logprob: float
    bytes: List[int]
    top_logprobs: List[float]


@dataclass(frozen=True)
class Logprobs:
    """Collection of logprobs for a completion."""
    content: List[Logprob] = field(default_factory=list)


@dataclass(frozen=True)
class Choice:
    """A single completion choice from the API."""
    index: int
    message: Message
    finish_reason: str
    logprobs: Optional[Logprobs] = None
    stop_reason: Optional[Any] = None


@dataclass(frozen=True)
class ChatCompletion:
    """A complete API response from the model."""
    id: str
    object: str
    created: int
    model: str
    usage: Usage
    kv_transfer_params: Optional[Any] = None
    choices: List[Choice] = field(default_factory=list)
    prompt_logprobs: Optional[List[Any]] = None


# ────────────────────────────── Main Rollout Type ──────────────────────────────

@dataclass
class Rollout:
    """Batch of inference interactions: messages (input), completions (output), metadata."""
    messages: List[Message] = field(default_factory=list)
    completions: List[ChatCompletion] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ────────────────────── Serialization ──────────────────────

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(json_str: str) -> "Rollout":
        """Deserialize from JSON string using dacite."""
        data = json.loads(json_str)
        return dacite.from_dict(data_class=Rollout, data=data)

    # ────────────────────── JSONL Batch Operations ──────────────────────

    @staticmethod
    def to_jsonl(rollouts: List["Rollout"]) -> str:
        """Convert list of rollouts to JSONL string."""
        return "\n".join(r.to_json() for r in rollouts)

    @staticmethod
    def from_jsonl(jsonl_str: str) -> List["Rollout"]:
        """Parse JSONL string into list of rollouts."""
        return [Rollout.from_json(line) for line in jsonl_str.strip().splitlines() if line]

    @staticmethod
    def save_jsonl(rollouts: List["Rollout"], filepath: str | Path) -> None:
        """Save rollouts to JSONL file."""
        Path(filepath).write_text(Rollout.to_jsonl(rollouts), encoding="utf-8")

    @staticmethod
    def load_jsonl(filepath: str | Path) -> List["Rollout"]:
        """Load rollouts from JSONL file."""
        return Rollout.from_jsonl(Path(filepath).read_text(encoding="utf-8"))

    @staticmethod
    def load_jsonl_streaming(filepath: str | Path) -> Iterator["Rollout"]:
        """Stream rollouts from JSONL file (memory-efficient for large files)."""
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield Rollout.from_json(line)

    # ────────────────────── Convenience Helpers ──────────────────────

    def get_completion_tokens(self) -> int:
        """Total completion tokens across all completions."""
        return sum(c.usage.completion_tokens for c in self.completions)

    def get_total_tokens(self) -> int:
        """Total tokens (prompt + completion) for the last completion."""
        if not self.completions:
            return 0
        return self.completions[-1].usage.total_tokens

    def get_last_message_content(self) -> Optional[str]:
        """Get content from the last completion message."""
        if not self.completions or not self.completions[-1].choices:
            return None
        return self.completions[-1].choices[0].message.content

    def hash(self) -> str:
        """Generate a unique hash for this rollout."""
        import hashlib
        rollout_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(rollout_str.encode()).hexdigest()[:16]


# ────────────────────── Sample Protocol ──────────────────────

@runtime_checkable
class Sample(Protocol):
    """Protocol for dataset samples (training, eval, etc)."""

    def to_rollout(self) -> Rollout:
        """Convert sample to Rollout with initial messages and metadata."""
        ...


# ────────────────────── Dataset Sample Implementations ──────────────────────

@dataclass
class GSM8KSample:
    """GSM8K dataset sample."""
    question: str
    answer: str
    sample_id: str = ""

    def __post_init__(self):
        """Assert invariants (Tiger Style)."""
        assert self.question.strip(), "question cannot be empty"
        assert self.answer.strip(), "answer cannot be empty"

    def to_rollout(self) -> Rollout:
        """Convert to Rollout with user message and metadata."""
        return Rollout(
            messages=[Message(role="user", content=self.question)],
            metadata={
                "dataset": "gsm8k",
                "ground_truth": self.answer,
                "sample_id": self.sample_id
            }
        )

    @staticmethod
    def from_hf_dict(item: dict, idx: int = 0) -> "GSM8KSample":
        """Create from HuggingFace dict (works for both loaded and streaming)."""
        answer = item["answer"].split("####")[-1].strip() if "####" in item["answer"] else item["answer"]
        return GSM8KSample(
            question=item["question"],
            answer=answer,
            sample_id=f"gsm8k_{idx:04d}"
        )


# ────────────────────── Endpoint & Inference ──────────────────────

@dataclass(frozen=True)
class Endpoint:
    """OpenAI/vLLM API endpoint configuration."""
    model: str
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    temperature: float = 1.0
    max_tokens: int = 2048


async def generate(
    endpoint: Endpoint,
    rollout: Rollout,
    on_chunk: Optional[Callable[[str], Awaitable[None]]] = None
) -> Rollout:
    """Generate completion and return updated rollout.

    Args:
        endpoint: API endpoint configuration (OpenAI or vLLM)
        rollout: Input rollout with messages
        on_chunk: Optional callback for streaming tokens

    Returns:
        Updated rollout with new completion appended
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=endpoint.api_key, base_url=endpoint.api_base)

    # Convert messages to OpenAI format
    messages = [{"role": m.role, "content": m.content} for m in rollout.messages]

    # Determine if we should stream
    stream = on_chunk is not None

    if stream:
        # Streaming mode: accumulate response and call on_chunk
        response = await client.chat.completions.create(
            model=endpoint.model,
            messages=messages,
            temperature=endpoint.temperature,
            max_tokens=endpoint.max_tokens,
            stream=True
        )

        accumulated_content = ""
        finish_reason = None
        response_id = None
        created = None

        async for chunk in response:
            if response_id is None:
                response_id = chunk.id
                created = chunk.created

            delta = chunk.choices[0].delta
            if delta.content:
                accumulated_content += delta.content
                await on_chunk(delta.content)

            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        # Build completion from accumulated stream
        completion = ChatCompletion(
            id=response_id or "unknown",
            object="chat.completion",
            created=created or 0,
            model=endpoint.model,
            usage=Usage(0, 0, 0),  # Stream doesn't provide usage
            choices=[Choice(
                index=0,
                message=Message(role="assistant", content=accumulated_content),
                finish_reason=finish_reason or "stop"
            )]
        )
    else:
        # Non-streaming mode: get full response
        response = await client.chat.completions.create(
            model=endpoint.model,
            messages=messages,
            temperature=endpoint.temperature,
            max_tokens=endpoint.max_tokens,
            stream=False
        )

        # Parse response into our ChatCompletion format
        completion = ChatCompletion(
            id=response.id,
            object=response.object,
            created=response.created,
            model=response.model,
            usage=Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            ),
            choices=[Choice(
                index=choice.index,
                message=Message(role=choice.message.role, content=choice.message.content or ""),
                finish_reason=choice.finish_reason
            ) for choice in response.choices]
        )

    # Return updated rollout
    return Rollout(
        messages=rollout.messages + [completion.choices[0].message],
        completions=rollout.completions + [completion],
        metadata=rollout.metadata
    )


# ────────────────────── CLI ──────────────────────

async def main():
    """CLI for querying OpenAI/vLLM endpoints."""
    import argparse
    import asyncio
    import logging
    import os
    import sys

    parser = argparse.ArgumentParser(description="Query OpenAI/vLLM endpoints")
    parser.add_argument("query", help="User query/prompt")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name (default: gpt-4o-mini)")
    parser.add_argument("--api-base", default="https://api.openai.com/v1", help="API base URL")
    parser.add_argument("--api-key", help="API key (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--system", help="System message")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature (default: 1.0)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens (default: 2048)")
    parser.add_argument("--stream", action="store_true", help="Stream output to terminal")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key and "api.openai.com" in args.api_base:
        logger.error("Error: OPENAI_API_KEY not set. Either:")
        logger.error("  1. Export OPENAI_API_KEY=your-key")
        logger.error("  2. Pass --api-key your-key")
        return 1

    # Create endpoint
    endpoint = Endpoint(
        model=args.model,
        api_base=args.api_base,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

    # Create initial rollout with messages
    messages = []
    if args.system:
        messages.append(Message(role="system", content=args.system))
    messages.append(Message(role="user", content=args.query))

    rollout = Rollout(messages=messages)

    logger.info(f"Querying {args.model} at {args.api_base}")
    if args.system:
        logger.debug(f"System: {args.system}")
    logger.debug(f"User: {args.query}")

    # Define streaming callback
    async def print_token(token: str):
        print(token, end="", flush=True)

    try:
        # Generate response
        if args.stream:
            logger.info("Streaming response...\n")
            updated_rollout = await generate(endpoint, rollout, on_chunk=print_token)
            print()  # Newline after stream
        else:
            updated_rollout = await generate(endpoint, rollout)
            response = updated_rollout.get_last_message_content()
            logger.info("Response:")
            print(response)

        # Show token usage
        if updated_rollout.completions:
            usage = updated_rollout.completions[-1].usage
            logger.info(f"\nTokens - Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    import asyncio
    sys.exit(asyncio.run(main()))
