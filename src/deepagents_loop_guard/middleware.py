"""Middleware that stops repeated, non-progressing tool calls.

The middleware instance deliberately stores configuration only. Runtime decisions are
derived from the graph's message history so one instance is safe to reuse across
threads and concurrent invocations.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolCallRequest,
    hook_config,
)
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.types import Command

_SOFT_BLOCK_MARKER = "[LOOP_GUARD_BLOCKED"
_HARD_STOP_MARKER = "[LOOP_GUARD_HARD_STOP"


def _json_default(value: Any) -> Any:
    """Convert common non-JSON values into deterministic representations."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if isinstance(value, bytes):
        return {"__bytes_sha256__": hashlib.sha256(value).hexdigest()}
    return {"__type__": type(value).__qualname__, "__repr__": repr(value)}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _drop_ignored_keys(value: Any, ignored_keys: frozenset[str]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _drop_ignored_keys(child, ignored_keys)
            for key, child in value.items()
            if str(key) not in ignored_keys
        }
    if isinstance(value, (list, tuple)):
        return [_drop_ignored_keys(child, ignored_keys) for child in value]
    return value


def canonical_tool_signature(
    tool_name: str,
    arguments: Mapping[str, Any] | None,
    *,
    ignored_argument_keys: Collection[str] = (),
) -> str:
    """Return a stable SHA-256 signature for a tool name and its arguments."""
    ignored = frozenset(ignored_argument_keys)
    normalized_args = _drop_ignored_keys(arguments or {}, ignored)
    payload = _canonical_json({"name": tool_name, "args": normalized_args})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _output_digest(message: ToolMessage) -> str:
    payload = {
        "content": message.content,
        "status": getattr(message, "status", None),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class _CompletedCall:
    tool_call_id: str
    tool_name: str
    signature: str
    output_digest: str
    soft_blocked: bool


def _tool_calls(message: AIMessage) -> Sequence[Mapping[str, Any]]:
    return message.tool_calls or ()


class RepeatedToolCallMiddleware(AgentMiddleware):
    """Block an exact tool call when prior identical calls made no progress.

    ``repeat_threshold=3`` means:

    * execute attempts one and two normally;
    * if both produced the same output, return an artificial error ToolMessage for
      attempt three without invoking the tool;
    * if the model immediately ignores that message and asks for the same call again,
      terminate the current agent run gracefully.

    Different arguments, a different output, or any intervening completed tool call
    resets the consecutive-repeat sequence.
    """

    def __init__(
        self,
        *,
        repeat_threshold: int = 3,
        hard_stop_after_blocks: int = 1,
        per_tool_thresholds: Mapping[str, int] | None = None,
        excluded_tools: Collection[str] = (),
        ignored_argument_keys: Mapping[str, Collection[str]] | None = None,
        message_scan_limit: int = 200,
    ) -> None:
        super().__init__()
        if repeat_threshold < 2:
            raise ValueError("repeat_threshold must be at least 2")
        if hard_stop_after_blocks < 1:
            raise ValueError("hard_stop_after_blocks must be at least 1")
        if message_scan_limit < 1:
            raise ValueError("message_scan_limit must be positive")

        thresholds = dict(per_tool_thresholds or {})
        invalid = {name: value for name, value in thresholds.items() if value < 2}
        if invalid:
            raise ValueError(f"per-tool thresholds must be at least 2: {invalid}")

        self.repeat_threshold = repeat_threshold
        self.hard_stop_after_blocks = hard_stop_after_blocks
        self.per_tool_thresholds = thresholds
        self.excluded_tools = frozenset(excluded_tools)
        self.ignored_argument_keys = {
            name: frozenset(keys)
            for name, keys in (ignored_argument_keys or {}).items()
        }
        self.message_scan_limit = message_scan_limit

    def _threshold_for(self, tool_name: str) -> int:
        return self.per_tool_thresholds.get(tool_name, self.repeat_threshold)

    def _signature(self, tool_call: Mapping[str, Any]) -> str:
        name = str(tool_call.get("name", ""))
        arguments = tool_call.get("args")
        if not isinstance(arguments, Mapping):
            arguments = {"__raw_args__": arguments}
        return canonical_tool_signature(
            name,
            arguments,
            ignored_argument_keys=self.ignored_argument_keys.get(name, ()),
        )

    def _completed_calls(self, messages: Sequence[BaseMessage]) -> list[_CompletedCall]:
        pending: dict[str, tuple[str, str]] = {}
        completed: list[_CompletedCall] = []

        for message in messages[-self.message_scan_limit :]:
            if isinstance(message, AIMessage):
                for call in _tool_calls(message):
                    call_id = str(call.get("id", ""))
                    if call_id:
                        pending[call_id] = (
                            str(call.get("name", "")),
                            self._signature(call),
                        )
                continue

            if not isinstance(message, ToolMessage):
                continue

            call_id = str(message.tool_call_id)
            pending_call = pending.get(call_id)
            if pending_call is None:
                tool_name = str(getattr(message, "name", "") or "")
                if not tool_name:
                    continue
                signature = ""
            else:
                tool_name, signature = pending_call

            content = message.content if isinstance(message.content, str) else ""
            completed.append(
                _CompletedCall(
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    signature=signature,
                    output_digest=_output_digest(message),
                    soft_blocked=content.startswith(_SOFT_BLOCK_MARKER),
                )
            )

        return completed

    @staticmethod
    def _consecutive_identical_outcomes(
        completed: Sequence[_CompletedCall],
        signature: str,
    ) -> int:
        matches: list[_CompletedCall] = []
        for call in reversed(completed):
            if call.signature != signature or call.soft_blocked:
                break
            matches.append(call)

        if not matches:
            return 0
        first_output = matches[0].output_digest
        if any(call.output_digest != first_output for call in matches[1:]):
            return 0
        return len(matches)

    @staticmethod
    def _consecutive_soft_blocks(
        completed: Sequence[_CompletedCall],
        signature: str,
    ) -> int:
        count = 0
        for call in reversed(completed):
            if call.signature != signature or not call.soft_blocked:
                break
            count += 1
        return count

    def _soft_block_message(
        self,
        request: ToolCallRequest,
        *,
        signature: str,
        prior_identical_outcomes: int,
        prior_soft_blocks: int,
    ) -> ToolMessage:
        tool_call = request.tool_call
        attempt = prior_identical_outcomes + prior_soft_blocks + 1
        threshold = self._threshold_for(str(tool_call["name"]))
        return ToolMessage(
            content=(
                f"{_SOFT_BLOCK_MARKER} signature={signature[:16]}] "
                f"Blocked repeated call #{attempt} to '{tool_call['name']}'. "
                f"The previous {threshold - 1} calls used the same arguments and "
                "made no observable progress. Do not call it again with the same "
                "arguments. Reuse the previous result, change strategy or arguments, "
                "or finish with the information already available."
            ),
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
            status="error",
        )

    def _should_soft_block(
        self,
        request: ToolCallRequest,
    ) -> tuple[bool, str, int, int]:
        tool_call = request.tool_call
        tool_name = str(tool_call["name"])
        signature = self._signature(tool_call)
        completed = self._completed_calls(request.state.get("messages", []))
        prior_outcomes = self._consecutive_identical_outcomes(completed, signature)
        prior_blocks = self._consecutive_soft_blocks(completed, signature)
        threshold = self._threshold_for(tool_name)
        should_block = prior_blocks > 0 or prior_outcomes >= threshold - 1
        return should_block, signature, prior_outcomes, prior_blocks

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        tool_name = str(request.tool_call["name"])
        if tool_name in self.excluded_tools:
            return handler(request)

        should_block, signature, prior_outcomes, prior_blocks = self._should_soft_block(
            request
        )
        if should_block:
            return self._soft_block_message(
                request,
                signature=signature,
                prior_identical_outcomes=prior_outcomes,
                prior_soft_blocks=prior_blocks,
            )
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Awaitable[ToolMessage | Command[Any]]
        ],
    ) -> ToolMessage | Command[Any]:
        tool_name = str(request.tool_call["name"])
        if tool_name in self.excluded_tools:
            return await handler(request)

        should_block, signature, prior_outcomes, prior_blocks = self._should_soft_block(
            request
        )
        if should_block:
            return self._soft_block_message(
                request,
                signature=signature,
                prior_identical_outcomes=prior_outcomes,
                prior_soft_blocks=prior_blocks,
            )
        return await handler(request)

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: Mapping[str, Any], runtime: Any) -> dict[str, Any] | None:
        """End the run if the model repeats a call after one or more soft blocks."""
        del runtime
        messages = state.get("messages", [])
        if not messages:
            return None

        last_ai = next(
            (message for message in reversed(messages) if isinstance(message, AIMessage)),
            None,
        )
        if last_ai is None or not last_ai.tool_calls:
            return None

        completed = self._completed_calls(messages)
        offending_call: Mapping[str, Any] | None = None
        offending_signature = ""
        prior_blocks = 0

        for tool_call in last_ai.tool_calls:
            tool_name = str(tool_call.get("name", ""))
            if tool_name in self.excluded_tools:
                continue
            signature = self._signature(tool_call)
            block_count = self._consecutive_soft_blocks(completed, signature)
            if block_count >= self.hard_stop_after_blocks:
                offending_call = tool_call
                offending_signature = signature
                prior_blocks = block_count
                break

        if offending_call is None:
            return None

        offending_id = str(offending_call.get("id", ""))
        artificial_messages: list[BaseMessage] = []
        for tool_call in last_ai.tool_calls:
            call_id = str(tool_call.get("id", ""))
            call_name = str(tool_call.get("name", ""))
            if call_id == offending_id:
                content = (
                    f"{_HARD_STOP_MARKER} signature={offending_signature[:16]}] "
                    f"Execution stopped because '{call_name}' was requested again "
                    f"after {prior_blocks} loop-guard block(s)."
                )
            else:
                content = (
                    "Execution stopped before this tool call could run because another "
                    "call in the same batch triggered the loop guard."
                )
            artificial_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=call_id,
                    name=call_name,
                    status="error",
                )
            )

        artificial_messages.append(
            AIMessage(
                content=(
                    "Repeated tool-call loop detected. This run was stopped without "
                    "executing the repeated call. Start a new request with changed "
                    "inputs or review the preceding tool results."
                )
            )
        )
        return {"messages": artificial_messages, "jump_to": "end"}

    async def aafter_model(
        self,
        state: Mapping[str, Any],
        runtime: Any,
    ) -> dict[str, Any] | None:
        return self.after_model(state, runtime)


def build_agent_guardrails(
    *,
    repeat_threshold: int = 3,
    hard_stop_after_blocks: int = 1,
    tool_run_limit: int | None = 30,
    model_run_limit: int | None = 20,
    per_tool_thresholds: Mapping[str, int] | None = None,
    excluded_tools: Collection[str] = (),
    ignored_argument_keys: Mapping[str, Collection[str]] | None = None,
) -> list[AgentMiddleware]:
    """Build the repeat guard plus optional total call circuit breakers.

    Set ``tool_run_limit`` or ``model_run_limit`` to ``None`` to omit the
    corresponding limiter. A value of ``0`` remains a real zero-call limit.
    """
    guardrails: list[AgentMiddleware] = [
        RepeatedToolCallMiddleware(
            repeat_threshold=repeat_threshold,
            hard_stop_after_blocks=hard_stop_after_blocks,
            per_tool_thresholds=per_tool_thresholds,
            excluded_tools=excluded_tools,
            ignored_argument_keys=ignored_argument_keys,
        )
    ]
    if tool_run_limit is not None:
        guardrails.append(
            ToolCallLimitMiddleware(run_limit=tool_run_limit, exit_behavior="end")
        )
    if model_run_limit is not None:
        guardrails.append(
            ModelCallLimitMiddleware(run_limit=model_run_limit, exit_behavior="end")
        )
    return guardrails
