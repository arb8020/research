"""
AskUserQuestionEnvironment - Environment with a single tool for asking the user questions.

This environment provides the ask_user_question tool, which allows the agent to
ask the user multiple-choice questions during execution. It's designed to be
composed with other environments (like LocalFilesystemEnvironment) to add
interactive question-asking capability.

Example usage:
    from rollouts.environments import compose, LocalFilesystemEnvironment
    from rollouts.environments.ask_user import AskUserQuestionEnvironment

    env = compose(
        LocalFilesystemEnvironment(working_dir=Path.cwd()),
        AskUserQuestionEnvironment(),
    )
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import trio

from ..dtypes import (
    AgentState,
    Message,
    RunConfig,
    Tool,
    ToolCall,
    ToolFunction,
    ToolFunctionParameter,
    ToolResult,
)

if TYPE_CHECKING:
    from rollouts.frontends.tui.theme import Theme


# Type for the question handler callback
# Takes questions list, returns answers dict mapping question text -> selected answer(s)
QuestionHandler = Callable[
    [list[dict[str, Any]]],  # questions
    Awaitable[dict[str, str]],  # answers: {question_text: selected_label(s)}
]


async def default_question_handler(questions: list[dict[str, Any]]) -> dict[str, str]:
    """Default handler that prompts via stdin.

    For each question, displays the options and asks user to select.
    """
    answers: dict[str, str] = {}

    for q in questions:
        question_text = q["question"]
        header = q.get("header", "")
        options = q["options"]
        multi_select = q.get("multiSelect", False)

        print(f"\n[{header}] {question_text}")
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt['label']}: {opt['description']}")
        print(f"  {len(options) + 1}. Other (free text)")

        if multi_select:
            print("Enter numbers separated by commas (e.g., 1,3):")
        else:
            print("Enter number:")

        # Use trio's to_thread for blocking input
        response = await trio.to_thread.run_sync(input, "Choice: ")
        response = response.strip()

        if multi_select:
            # Parse comma-separated numbers
            selected_labels = []
            for part in response.split(","):
                part = part.strip()
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(options):
                        selected_labels.append(options[idx]["label"])
                    elif idx == len(options):
                        # "Other" selected
                        other_text = await trio.to_thread.run_sync(input, "Enter your answer: ")
                        selected_labels.append(other_text.strip())
            answers[question_text] = ", ".join(selected_labels)
        else:
            if response.isdigit():
                idx = int(response) - 1
                if 0 <= idx < len(options):
                    answers[question_text] = options[idx]["label"]
                elif idx == len(options):
                    # "Other" selected
                    other_text = await trio.to_thread.run_sync(input, "Enter your answer: ")
                    answers[question_text] = other_text.strip()
                else:
                    answers[question_text] = options[0]["label"]  # Default to first
            else:
                # Treat as free text
                answers[question_text] = response

    return answers


def format_ask_user_question(
    tool_name: str,
    args: dict,
    result: dict | None,
    expanded: bool,
    theme: Theme | None = None,
) -> str:
    """Format ask_user_question tool for TUI display."""
    questions = args.get("questions") or []
    num_questions = len(questions) if isinstance(questions, list) else 0

    if num_questions == 1:
        q = questions[0]
        header = q.get("header", "Question") if isinstance(q, dict) else "Question"
        question_text = q.get("question", "...") if isinstance(q, dict) else "..."
        text = f"ask_user_question([{header}] {question_text[:50]})"
    elif num_questions > 1:
        text = f"ask_user_question({num_questions} questions)"
    else:
        text = "ask_user_question(invalid: no questions)"

    if result:
        # Check for error first
        is_error = result.get("isError", False) or result.get("is_error", False)
        error = result.get("error", "")

        if is_error and error:
            text += f"\n⎿ Error: {error[:100]}"
        else:
            # Show the answers
            content = result.get("content", "")
            if isinstance(content, str) and content:
                try:
                    answers = json.loads(content)
                    if isinstance(answers, dict):
                        text += "\n⎿ User answered:"
                        for q_text, answer in answers.items():
                            # Truncate long question text
                            short_q = q_text[:40] + "..." if len(q_text) > 40 else q_text
                            text += f"\n  {short_q}: {answer}"
                except json.JSONDecodeError:
                    text += f"\n⎿ {content[:100]}"

    return text


@dataclass
class AskUserQuestionEnvironment:
    """Environment with ask_user_question tool for interactive user queries.

    This environment provides a single tool that allows the agent to ask the user
    multiple-choice questions. It's designed to be composed with other environments.

    Args:
        question_handler: Async callback that handles presenting questions to the user
                         and returning their answers. Defaults to stdin prompts.
    """

    question_handler: QuestionHandler = field(default_factory=lambda: default_question_handler)

    def get_name(self) -> str:
        return "ask_user"

    def get_tools(self) -> list[Tool]:
        """Return the ask_user_question tool."""
        return [
            Tool(
                type="function",
                function=ToolFunction(
                    name="ask_user_question",
                    description=(
                        "Ask the user one or more multiple-choice questions. Use this to gather "
                        "preferences, clarify requirements, or get decisions on implementation "
                        "choices. Each question can have 2-4 options, and users can always "
                        "provide a custom 'Other' response."
                    ),
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "questions": {
                                "type": "array",
                                "description": "Questions to ask the user (1-4 questions)",
                                "minItems": 1,
                                "maxItems": 4,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "question": {
                                            "type": "string",
                                            "description": (
                                                "The complete question to ask. Should be clear, "
                                                "specific, and end with a question mark."
                                            ),
                                        },
                                        "header": {
                                            "type": "string",
                                            "description": (
                                                "Short label displayed as a tag (max 12 chars). "
                                                "Examples: 'Auth method', 'Library', 'Approach'"
                                            ),
                                            "maxLength": 12,
                                        },
                                        "options": {
                                            "type": "array",
                                            "description": "Available choices (2-4 options)",
                                            "minItems": 2,
                                            "maxItems": 4,
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "label": {
                                                        "type": "string",
                                                        "description": (
                                                            "Concise choice text (1-5 words)"
                                                        ),
                                                    },
                                                    "description": {
                                                        "type": "string",
                                                        "description": (
                                                            "Explanation of what this option "
                                                            "means or its trade-offs"
                                                        ),
                                                    },
                                                },
                                                "required": ["label", "description"],
                                            },
                                        },
                                        "multiSelect": {
                                            "type": "boolean",
                                            "description": (
                                                "Allow selecting multiple options. Use when "
                                                "choices are not mutually exclusive."
                                            ),
                                            "default": False,
                                        },
                                    },
                                    "required": ["question", "header", "options", "multiSelect"],
                                },
                            },
                        },
                    ),
                    required=["questions"],
                ),
            ),
        ]

    async def serialize(self) -> dict:
        """Serialize environment state."""
        return {"env_kind": "ask_user"}

    @staticmethod
    async def deserialize(data: dict) -> AskUserQuestionEnvironment:
        """Deserialize environment state.

        Note: The question_handler callback cannot be serialized, so deserialization
        uses the default handler. Override after deserialization if needed.
        """
        return AskUserQuestionEnvironment()

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        """ask_user_question doesn't need confirmation - it IS user interaction."""
        return False

    def get_tool_formatter(
        self, tool_name: str
    ) -> Callable[[str, dict, dict | None, bool, Theme | None], str] | None:
        """Return formatter for TUI display."""
        if tool_name == "ask_user_question":
            return format_ask_user_question
        return None

    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState:
        """No special message handling needed."""
        return state

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        """Execute the ask_user_question tool."""
        if tool_call.name != "ask_user_question":
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Unknown tool: {tool_call.name}",
            )

        questions = tool_call.args.get("questions")

        # Validate questions
        if not questions:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=(
                    "No questions provided. You must pass a 'questions' array with 1-4 question objects. "
                    "Each question needs: question (string), header (string, max 12 chars), "
                    "options (array of 2-4 {label, description} objects), and multiSelect (boolean)."
                ),
            )

        if len(questions) > 4:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error="Maximum 4 questions allowed",
            )

        for i, q in enumerate(questions):
            if not q.get("question"):
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content="",
                    error=f"Question {i + 1} is missing 'question' field",
                )
            options = q.get("options", [])
            if len(options) < 2:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content="",
                    error=f"Question {i + 1} must have at least 2 options",
                )
            if len(options) > 4:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content="",
                    error=f"Question {i + 1} has more than 4 options",
                )

        try:
            # Call the question handler to get user answers
            answers = await self.question_handler(questions)

            # Return answers as JSON
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=False,
                content=json.dumps(answers),
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Failed to get user response: {e}",
            )
