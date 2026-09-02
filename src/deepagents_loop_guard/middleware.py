"""진전 없이 반복되는 도구 호출을 중단하는 미들웨어입니다.

미들웨어 인스턴스에는 의도적으로 설정값만 저장합니다. 실행 중 판단은 그래프의
메시지 이력에서 계산하므로 하나의 인스턴스를 여러 스레드와 동시 호출에서
안전하게 재사용할 수 있습니다.
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
    """자주 사용되는 비 JSON 값을 항상 동일한 형태로 변환합니다."""
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
    """도구 이름과 인자로부터 일관된 SHA-256 시그니처를 생성합니다."""
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


@dataclass(frozen=True, slots=True)
class _CompletedBatch:
    """하나의 모델 응답에서 함께 생성된 도구 호출 묶음입니다."""

    calls: tuple[_CompletedCall, ...]


def _tool_calls(message: AIMessage) -> Sequence[Mapping[str, Any]]:
    return message.tool_calls or ()


class RepeatedToolCallMiddleware(AgentMiddleware):
    """이전의 동일한 호출에서 진전이 없으면 같은 도구 호출을 차단합니다.

    ``repeat_threshold=3``의 동작은 다음과 같습니다.

    * 첫 번째와 두 번째 호출은 정상적으로 실행합니다.
    * 두 호출의 출력까지 같으면 세 번째 호출은 실제 도구를 실행하지 않고
      오류 상태의 인위적인 ``ToolMessage``를 반환합니다.
    * 모델이 이 메시지를 무시하고 즉시 같은 호출을 다시 요청하면 현재 agent
      run을 안전하게 종료합니다.

    하나의 모델 응답에서 함께 생성된 호출은 한 batch로 추적합니다. 예를 들어
    ``A + B`` 병렬 batch가 반복되면 A와 B의 반복 횟수가 각각 증가하며, 함께
    실행된 B의 결과가 A의 연속성을 끊지 않습니다. 다음 모델 turn에서 A가
    호출되지 않거나 A의 인자 또는 출력이 바뀌면 A의 연속 반복 횟수를
    초기화합니다.
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

    def _completed_batches(self, messages: Sequence[BaseMessage]) -> list[_CompletedBatch]:
        batch_calls: list[list[tuple[str, str, str]]] = []
        batch_results: list[dict[str, ToolMessage]] = []
        pending_batch_by_call_id: dict[str, int] = {}

        for message in messages[-self.message_scan_limit :]:
            if isinstance(message, AIMessage):
                definitions: list[tuple[str, str, str]] = []
                for call in _tool_calls(message):
                    call_id = str(call.get("id", ""))
                    if call_id:
                        definitions.append(
                            (
                                call_id,
                                str(call.get("name", "")),
                                self._signature(call),
                            )
                        )
                if definitions:
                    batch_index = len(batch_calls)
                    batch_calls.append(definitions)
                    batch_results.append({})
                    for call_id, _, _ in definitions:
                        pending_batch_by_call_id[call_id] = batch_index
                continue

            if not isinstance(message, ToolMessage):
                continue

            call_id = str(message.tool_call_id)
            batch_index = pending_batch_by_call_id.get(call_id)
            if batch_index is not None:
                batch_results[batch_index][call_id] = message

        completed: list[_CompletedBatch] = []
        for definitions, results in zip(batch_calls, batch_results, strict=True):
            if not definitions or any(call_id not in results for call_id, _, _ in definitions):
                continue

            calls: list[_CompletedCall] = []
            for call_id, tool_name, signature in definitions:
                message = results[call_id]
                content = message.content if isinstance(message.content, str) else ""
                calls.append(
                    _CompletedCall(
                        tool_call_id=call_id,
                        tool_name=tool_name,
                        signature=signature,
                        output_digest=_output_digest(message),
                        soft_blocked=content.startswith(_SOFT_BLOCK_MARKER),
                    )
                )
            completed.append(_CompletedBatch(calls=tuple(calls)))

        return completed

    @staticmethod
    def _consecutive_identical_outcomes(
        completed: Sequence[_CompletedBatch],
        signature: str,
    ) -> int:
        expected_outputs: tuple[str, ...] | None = None
        count = 0
        for batch in reversed(completed):
            matching_calls = [call for call in batch.calls if call.signature == signature]
            if not matching_calls or any(call.soft_blocked for call in matching_calls):
                break
            outputs = tuple(sorted(call.output_digest for call in matching_calls))
            if expected_outputs is None:
                expected_outputs = outputs
            elif outputs != expected_outputs:
                break
            count += 1
        return count

    @staticmethod
    def _consecutive_soft_blocks(
        completed: Sequence[_CompletedBatch],
        signature: str,
    ) -> int:
        count = 0
        for batch in reversed(completed):
            matching_calls = [call for call in batch.calls if call.signature == signature]
            if not matching_calls or not all(call.soft_blocked for call in matching_calls):
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
        threshold = self._threshold_for(str(tool_call["name"]))
        attempt = (
            threshold + prior_soft_blocks
            if prior_soft_blocks
            else prior_identical_outcomes + 1
        )
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
        completed = self._completed_batches(request.state.get("messages", []))
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
        """모델이 1차 차단 이후에도 같은 호출을 반복하면 현재 실행을 종료합니다."""
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

        completed = self._completed_batches(messages)
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
    """반복 호출 보호 로직과 선택적인 전체 호출 제한 미들웨어를 구성합니다.

    ``tool_run_limit`` 또는 ``model_run_limit``을 ``None``으로 지정하면 해당
    제한 미들웨어를 추가하지 않습니다. ``0``은 비활성화가 아니라 실제 허용 횟수
    0회를 의미합니다.
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
