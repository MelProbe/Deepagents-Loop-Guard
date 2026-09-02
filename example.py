"""미들웨어를 기존 Deep Agents 프로젝트에 복사해서 적용하는 예제입니다."""

from deepagents import create_deep_agent

# 아래 경로를 기존 애플리케이션에서 middleware.py를 복사한 실제 위치로 바꿉니다.
# 이 예제를 적용하기 위해 별도 패키지를 설치할 필요는 없습니다.
from your_project.agent.middleware.repeated_tool_call import build_agent_guardrails

# 회사에서 사용하는 OpenAI 호환 로컬 모델 인스턴스로 교체합니다.
local_model = ...

guardrails = build_agent_guardrails(
    repeat_threshold=3,
    hard_stop_after_blocks=1,
    tool_run_limit=30,
    model_run_limit=20,
    # 제한값을 None으로 지정하면 해당 호출 제한 미들웨어를 사용하지 않습니다.
    # 주기적 상태 조회 도구에는 자체 시간 제한과 재시도 간격 정책을 적용합니다.
    excluded_tools={"poll_job_status"},
    # 도구의 의미를 바꾸지 않는 일회성 요청 메타데이터만 비교에서 제외합니다.
    ignored_argument_keys={"read_file": {"trace_id"}},
)

agent = create_deep_agent(
    model=local_model,
    tools=[],  # 패치노트, Java 소스, git 이력 조회 및 테스트 실행 도구
    middleware=guardrails,
)

# 선언형 subagent는 별도의 미들웨어 목록을 사용합니다. 애플리케이션에서
# 작업 위임을 사용한다면 각 subagent에 새로운 보호 미들웨어 목록을 지정합니다.
# subagents=[
#     {
#         "name": "java-test-writer",
#         "description": "Java 테스트를 생성하고 검증합니다.",
#         "system_prompt": "...",
#         "tools": [...],
#         "middleware": build_agent_guardrails(),
#     }
# ]
