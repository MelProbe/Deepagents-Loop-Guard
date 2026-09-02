from __future__ import annotations

import asyncio

import pytest
from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest

from deepagents_loop_guard import (
    RepeatedToolCallMiddleware,
    build_agent_guardrails,
    canonical_tool_signature,
)
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware


def call(call_id: str, *, name: str = "read_file", args=None) -> dict:
    return {
        "id": call_id,
        "name": name,
        "args": args or {"path": "src/Foo.java"},
        "type": "tool_call",
    }


def completed(call_id: str, output: str, *, name: str = "read_file", args=None):
    tool_call = call(call_id, name=name, args=args)
    return [
        AIMessage(content="", tool_calls=[tool_call]),
        ToolMessage(
            content=output,
            tool_call_id=call_id,
            name=name,
            status="success",
        ),
    ]


def request(
    tool_call: dict,
    messages: list,
    *,
    current_batch: list[dict] | None = None,
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call=tool_call,
        tool=None,
        state={
            "messages": [
                *messages,
                AIMessage(content="", tool_calls=current_batch or [tool_call]),
            ]
        },
        runtime=None,
    )


def completed_parallel_batch(
    batch_number: int,
    *,
    a_output: str = "same-a",
    b_output: str = "same-b",
):
    a = call(
        f"a-{batch_number}",
        name="read_file",
        args={"path": "src/A.java"},
    )
    b = call(
        f"b-{batch_number}",
        name="git_log",
        args={"path": "src/B.java"},
    )
    return [
        AIMessage(content="", tool_calls=[a, b]),
        ToolMessage(
            content=a_output,
            tool_call_id=a["id"],
            name=a["name"],
            status="success",
        ),
        ToolMessage(
            content=b_output,
            tool_call_id=b["id"],
            name=b["name"],
            status="success",
        ),
    ]


def test_signature_is_stable_across_dictionary_order():
    left = canonical_tool_signature("search", {"query": "x", "limit": 10})
    right = canonical_tool_signature("search", {"limit": 10, "query": "x"})
    assert left == right


def test_signature_can_ignore_explicit_volatile_keys():
    left = canonical_tool_signature(
        "read_file", {"path": "A.java", "trace_id": "one"}, ignored_argument_keys={"trace_id"}
    )
    right = canonical_tool_signature(
        "read_file", {"path": "A.java", "trace_id": "two"}, ignored_argument_keys={"trace_id"}
    )
    assert left == right


def test_third_identical_outcome_is_blocked_without_executing_tool():
    middleware = RepeatedToolCallMiddleware(repeat_threshold=3)
    messages = [*completed("1", "same"), *completed("2", "same")]
    req = request(call("3"), messages)
    executed = False

    def handler(_request):
        nonlocal executed
        executed = True
        return ToolMessage(content="ran", tool_call_id="3")

    result = middleware.wrap_tool_call(req, handler)

    assert executed is False
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.content.startswith("[LOOP_GUARD_BLOCKED")


def test_third_repeated_parallel_batch_blocks_each_tool_lane():
    middleware = RepeatedToolCallMiddleware(repeat_threshold=3)
    messages = [*completed_parallel_batch(1), *completed_parallel_batch(2)]
    current_batch = [
        call("a-3", name="read_file", args={"path": "src/A.java"}),
        call("b-3", name="git_log", args={"path": "src/B.java"}),
    ]
    executed: list[str] = []

    def handler(tool_request):
        executed.append(tool_request.tool_call["name"])
        return ToolMessage(
            content="ran",
            tool_call_id=tool_request.tool_call["id"],
        )

    results = [
        middleware.wrap_tool_call(
            request(tool_call, messages, current_batch=current_batch),
            handler,
        )
        for tool_call in current_batch
    ]

    assert executed == []
    assert all(isinstance(result, ToolMessage) for result in results)
    assert all(result.content.startswith("[LOOP_GUARD_BLOCKED") for result in results)


def test_parallel_sibling_does_not_hide_repeated_tool_with_unchanged_output():
    middleware = RepeatedToolCallMiddleware(repeat_threshold=3)
    messages = [
        *completed_parallel_batch(1, b_output="b-v1"),
        *completed_parallel_batch(2, b_output="b-v2"),
    ]
    a = call("a-3", name="read_file", args={"path": "src/A.java"})
    b = call("b-3", name="git_log", args={"path": "src/B.java"})

    a_result = middleware.wrap_tool_call(
        request(a, messages, current_batch=[b, a]),
        lambda _request: pytest.fail("unchanged A must be blocked"),
    )
    b_result = middleware.wrap_tool_call(
        request(b, messages, current_batch=[b, a]),
        lambda _request: ToolMessage(content="b-v3", tool_call_id=b["id"]),
    )

    assert a_result.status == "error"
    assert b_result.content == "b-v3"


def test_changed_output_is_treated_as_progress():
    middleware = RepeatedToolCallMiddleware(repeat_threshold=3)
    messages = [*completed("1", "v1"), *completed("2", "v2")]
    req = request(call("3"), messages)

    result = middleware.wrap_tool_call(
        req,
        lambda _request: ToolMessage(content="v3", tool_call_id="3"),
    )

    assert isinstance(result, ToolMessage)
    assert result.content == "v3"


def test_intervening_different_call_resets_consecutive_sequence():
    middleware = RepeatedToolCallMiddleware(repeat_threshold=3)
    messages = [
        *completed("1", "same"),
        *completed("2", "same"),
        *completed("p", "changed", name="edit_file", args={"path": "src/Foo.java"}),
    ]
    req = request(call("3"), messages)

    result = middleware.wrap_tool_call(
        req,
        lambda _request: ToolMessage(content="same", tool_call_id="3"),
    )

    assert result.content == "same"


def test_excluded_tool_is_never_blocked():
    middleware = RepeatedToolCallMiddleware(excluded_tools={"poll_job_status"})
    args = {"job_id": "42"}
    messages = [
        *completed("1", "pending", name="poll_job_status", args=args),
        *completed("2", "pending", name="poll_job_status", args=args),
    ]
    req = request(call("3", name="poll_job_status", args=args), messages)

    result = middleware.wrap_tool_call(
        req,
        lambda _request: ToolMessage(content="done", tool_call_id="3"),
    )

    assert result.content == "done"


def test_tool_specific_threshold_can_block_second_attempt():
    middleware = RepeatedToolCallMiddleware(per_tool_thresholds={"edit_file": 2})
    args = {"path": "A.java", "old": "x", "new": "y"}
    messages = completed("1", "updated", name="edit_file", args=args)
    req = request(call("2", name="edit_file", args=args), messages)

    result = middleware.wrap_tool_call(req, lambda _request: pytest.fail("must not execute"))

    assert result.status == "error"


def test_repeating_after_soft_block_ends_run_and_covers_parallel_calls():
    middleware = RepeatedToolCallMiddleware()
    args = {"path": "src/Foo.java"}
    repeated = call("4", args=args)
    parallel = call("5", name="git_log", args={"limit": 10})
    signature = canonical_tool_signature("read_file", args)
    messages = [
        *completed("1", "same", args=args),
        *completed("2", "same", args=args),
        AIMessage(content="", tool_calls=[call("3", args=args)]),
        ToolMessage(
            content=f"[LOOP_GUARD_BLOCKED signature={signature[:16]}] blocked",
            tool_call_id="3",
            name="read_file",
            status="error",
        ),
        AIMessage(content="", tool_calls=[repeated, parallel]),
    ]

    result = middleware.after_model({"messages": messages}, runtime=None)

    assert result is not None
    assert result["jump_to"] == "end"
    assert len(result["messages"]) == 3
    assert all(
        isinstance(message, ToolMessage) for message in result["messages"][:2]
    )
    assert "Repeated tool-call loop detected" in result["messages"][-1].content


def test_async_path_also_short_circuits():
    middleware = RepeatedToolCallMiddleware(repeat_threshold=3)
    messages = [*completed("1", "same"), *completed("2", "same")]
    req = request(call("3"), messages)
    executed = False

    async def handler(_request):
        nonlocal executed
        executed = True
        return ToolMessage(content="ran", tool_call_id="3")

    result = asyncio.run(middleware.awrap_tool_call(req, handler))

    assert executed is False
    assert result.status == "error"


def test_create_deep_agent_integration_stops_repeated_tool_loop():
    executions: list[str] = []

    @tool
    def inspect_java(path: str) -> str:
        """테스트 생성을 위해 Java 소스 파일을 읽습니다."""
        executions.append(path)
        return "unchanged source"

    class ToolAwareFakeModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            del tools, kwargs
            return self

    repeated_calls = [
        AIMessage(
            content="",
            tool_calls=[
                call(str(index), name="inspect_java", args={"path": "src/Foo.java"})
            ],
        )
        for index in range(1, 5)
    ]
    model = ToolAwareFakeModel(responses=repeated_calls)
    agent = create_deep_agent(
        model=model,
        tools=[inspect_java],
        middleware=[RepeatedToolCallMiddleware(repeat_threshold=3)],
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Inspect Foo repeatedly"}]}
    )

    assert executions == ["src/Foo.java", "src/Foo.java"]
    assert "Repeated tool-call loop detected" in result["messages"][-1].content


def test_create_deep_agent_integration_stops_parallel_tool_loop():
    executions: list[tuple[str, str]] = []

    @tool
    def inspect_java(path: str) -> str:
        """테스트 생성을 위해 Java 소스 파일을 읽습니다."""
        executions.append(("inspect_java", path))
        return "unchanged source"

    @tool
    def inspect_git(path: str) -> str:
        """Java 소스 파일의 git 이력을 읽습니다."""
        executions.append(("inspect_git", path))
        return "unchanged history"

    class ToolAwareFakeModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            del tools, kwargs
            return self

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                call(
                    f"a-{index}",
                    name="inspect_java",
                    args={"path": "src/Foo.java"},
                ),
                call(
                    f"b-{index}",
                    name="inspect_git",
                    args={"path": "src/Foo.java"},
                ),
            ],
        )
        for index in range(1, 5)
    ]
    agent = create_deep_agent(
        model=ToolAwareFakeModel(responses=responses),
        tools=[inspect_java, inspect_git],
        middleware=[RepeatedToolCallMiddleware(repeat_threshold=3)],
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Inspect source and git repeatedly"}]}
    )

    assert executions.count(("inspect_java", "src/Foo.java")) == 2
    assert executions.count(("inspect_git", "src/Foo.java")) == 2
    assert "Repeated tool-call loop detected" in result["messages"][-1].content


def test_none_disables_both_total_call_limiters():
    guardrails = build_agent_guardrails(
        tool_run_limit=None,
        model_run_limit=None,
    )

    assert len(guardrails) == 1
    assert isinstance(guardrails[0], RepeatedToolCallMiddleware)


def test_none_disables_only_the_selected_limiter():
    tool_only = build_agent_guardrails(
        tool_run_limit=30,
        model_run_limit=None,
    )
    model_only = build_agent_guardrails(
        tool_run_limit=None,
        model_run_limit=20,
    )

    assert any(isinstance(item, ToolCallLimitMiddleware) for item in tool_only)
    assert not any(isinstance(item, ModelCallLimitMiddleware) for item in tool_only)
    assert any(isinstance(item, ModelCallLimitMiddleware) for item in model_only)
    assert not any(isinstance(item, ToolCallLimitMiddleware) for item in model_only)


def test_zero_keeps_limiters_enabled():
    guardrails = build_agent_guardrails(tool_run_limit=0, model_run_limit=0)

    assert any(isinstance(item, ToolCallLimitMiddleware) for item in guardrails)
    assert any(isinstance(item, ModelCallLimitMiddleware) for item in guardrails)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"repeat_threshold": 1}, "repeat_threshold"),
        ({"hard_stop_after_blocks": 0}, "hard_stop_after_blocks"),
        ({"message_scan_limit": 0}, "message_scan_limit"),
        ({"per_tool_thresholds": {"read_file": 1}}, "per-tool thresholds"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RepeatedToolCallMiddleware(**kwargs)
