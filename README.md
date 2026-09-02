# DeepAgents repeated tool-call middleware

기존 DeepAgents/LangChain 애플리케이션 코드에 직접 추가해서 사용하는 stateless
반복 호출 차단 미들웨어입니다. 별도 Python 패키지로 배포하거나 설치하는 것을
전제로 하지 않습니다.

## 동작

기본값 `repeat_threshold=3`은 다음 의미입니다.

1. 같은 도구와 같은 인자를 사용한 첫 두 호출은 실행합니다.
2. 두 호출의 결과까지 동일하면 세 번째 호출은 실제 도구를 실행하지 않고
   `LOOP_GUARD_BLOCKED` ToolMessage를 모델에 반환합니다.
3. 모델이 경고를 무시하고 바로 같은 호출을 다시 요청하면 해당 agent run을
   종료합니다.
4. 서로 다른 인자, 서로 다른 출력, 중간의 다른 도구 호출은 연속 반복을
   초기화합니다.

미들웨어 인스턴스에는 실행 중 카운터가 없습니다. 모든 판단은 graph state의
message history에서 계산하므로 thread, 병렬 invocation 및 subagent 환경에서
공유 가변 상태로 인한 경합이 발생하지 않습니다.

## 기존 코드에 적용하기

다음 파일의 내용을 기존 프로젝트의 middleware 모듈로 복사합니다.

```text
src/deepagents_loop_guard/middleware.py
```

예를 들어 기존 프로젝트의 아래 위치에 붙였다면:

```text
your_project/agent/middleware/repeated_tool_call.py
```

실제 프로젝트 경로에서 import해서 `create_deep_agent()`에 전달합니다.

## 적용 예시

```python
from deepagents import create_deep_agent
from your_project.agent.middleware.repeated_tool_call import build_agent_guardrails

agent = create_deep_agent(
    model=local_model,
    tools=[read_patch_notes, read_java_file, git_log, write_test, run_tests],
    middleware=build_agent_guardrails(
        repeat_threshold=3,
        hard_stop_after_blocks=1,
        tool_run_limit=30,
        model_run_limit=20,
        per_tool_thresholds={
            # 같은 mutation을 두 번 요청하면 두 번째 실행부터 차단
            "write_file": 2,
            "edit_file": 2,
        },
        # 자체 timeout/backoff가 있는 polling 도구는 exact-repeat 검사에서 제외
        excluded_tools={"poll_job_status"},
    ),
)
```

`build_agent_guardrails()`는 다음 세 가지를 반환합니다.

- `RepeatedToolCallMiddleware`: 동일 name/input과 동일한 이전 output 감지
- `ToolCallLimitMiddleware`: run 전체 tool-call 상한(`tool_run_limit=None`이면 제외)
- `ModelCallLimitMiddleware`: run 전체 model-call 상한(`model_run_limit=None`이면 제외)

전체 호출 상한을 사용하지 않으려면 다음처럼 설정합니다. `0`은 비활성화가
아니라 실제 허용 횟수 0회를 의미합니다.

```python
middleware = build_agent_guardrails(
    tool_run_limit=None,
    model_run_limit=None,
)
```

## Subagent

DeepAgents의 선언형 subagent는 별도의 middleware stack을 사용합니다. 각
subagent에도 새 guardrail 목록을 지정해야 합니다.

```python
subagents = [
    {
        "name": "java-test-writer",
        "description": "Java 테스트를 생성하고 검증합니다.",
        "system_prompt": "...",
        "tools": [read_java_file, write_test, run_tests],
        "middleware": build_agent_guardrails(),
    }
]
```

## 판정 규칙

입력 signature는 아래 값을 SHA-256으로 계산합니다.

```text
canonical-json({"name": tool_name, "args": arguments})
```

dictionary key 순서는 무시하지만 list 순서와 문자열 내용은 유지합니다. 출력은
`ToolMessage.content + status`의 hash로 비교하며 원문을 별도로 저장하지 않습니다.

`ignored_argument_keys`는 trace ID처럼 도구 결과에 영향을 주지 않는 필드에만
사용해야 합니다. timestamp, git revision 등 실제 결과를 바꾸는 필드를 무시하면
정상 호출을 차단할 수 있습니다.

현재 구현은 false positive를 최소화하기 위해 **exact repeat**만 감지합니다.
인자 문자열의 semantic similarity나 작은 편집 진동(A→B→A)은 별도 정책으로
추가하는 것이 안전합니다.

## 테스트

`tests/test_middleware.py`는 구현 검증용 참고 테스트입니다. 기존 프로젝트에 옮길
때 import 경로를 실제 middleware 경로에 맞게 변경한 뒤, 프로젝트에서 사용 중인
테스트 명령으로 실행하면 됩니다.

이 폴더의 `pyproject.toml`과 `src/deepagents_loop_guard/__init__.py`는 현재 예제와
회귀 테스트를 독립적으로 검증하기 위한 scaffold일 뿐입니다. 운영 프로젝트에
복사할 필요가 없습니다.
