"""Example of copying the middleware into an existing Deep Agents project."""

from deepagents import create_deep_agent

# Replace this import with the location where middleware.py was copied in the
# existing application. This example does not require installing a separate package.
from your_project.agent.middleware.repeated_tool_call import build_agent_guardrails

# Replace this with the company's OpenAI-compatible local model instance.
local_model = ...

guardrails = build_agent_guardrails(
    repeat_threshold=3,
    hard_stop_after_blocks=1,
    tool_run_limit=30,
    model_run_limit=20,
    # Set either limit to None to omit that limiter entirely.
    # Polling tools should use their own timeout/backoff policy.
    excluded_tools={"poll_job_status"},
    # Ignore volatile request metadata only when it does not change tool semantics.
    ignored_argument_keys={"read_file": {"trace_id"}},
)

agent = create_deep_agent(
    model=local_model,
    tools=[],  # patch-note, Java source, git-history and test-execution tools
    middleware=guardrails,
)

# Declarative subagents have their own middleware stack. Attach a fresh guardrail
# list to every subagent if the application enables delegation.
# subagents=[
#     {
#         "name": "java-test-writer",
#         "description": "Generates and verifies Java tests.",
#         "system_prompt": "...",
#         "tools": [...],
#         "middleware": build_agent_guardrails(),
#     }
# ]
