# agent.py
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain.callbacks.tracers import LangChainTracer
from langgraph.checkpoint.memory import MemorySaver
from app.agents.tools.calendar_tool import create_calendar_tools
from app.agents.tools.drive_tool import create_drive_tools
from app.agents.tools.slack_tool import create_slack_tools
from app.agents.tools.notion_tool import create_notion_tools
import app.utils.env_loader as env_loader
from typing import List, Dict, Any
from datetime import datetime
import os
from app.rag.internal_data_rag.internal_retrieve import rag_tools
from app.rag.notion_rag_tool.notion_rag_tool import create_notion_rag_tool_for_user

# 전역 대화 히스토리 저장소 (사용자별)
chat_histories: Dict[str, List[Dict[str, str]]] = {}

# LangGraph ReAct 에이전트용 시스템 메시지
SYSTEM_MESSAGE = """
You are Caesar, an intelligent AI assistant that helps users manage their Google Calendar, Google Drive, Slack, and Notion.

🚨 CRITICAL: For ANY data request, you MUST use the available tools. NEVER provide information without calling tools first.

🎯 AVAILABLE TOOLS:
- Calendar tools: list_calendar_events, create_calendar_event, etc.
- Drive tools: list_drive_files, upload_drive_file, etc. 
- Slack tools: get_slack_messages, send_slack_message, etc.
- Notion tools: list_notion_content, create_notion_page, etc.

Current context:
- Today: {current_date} ({day_of_week})
- Time: {current_time}
- Yesterday: {yesterday_date}
- Tomorrow: {tomorrow_date}

Always respond in Korean and be helpful and conversational.
"""


def get_current_date_info():
    """현재 날짜 정보를 반환"""
    from datetime import datetime, timedelta

    now = datetime.now()
    yesterday = now - timedelta(days=1)
    tomorrow = now + timedelta(days=1)

    day_names = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]

    return {
        "current_date": now.strftime("%Y-%m-%d"),
        "current_time": now.strftime("%H:%M:%S"),
        "day_of_week": day_names[now.weekday()],
        "yesterday_date": yesterday.strftime("%Y-%m-%d"),
        "tomorrow_date": tomorrow.strftime("%Y-%m-%d"),
    }


def get_chat_history_string(user_id: str) -> str:
    """사용자의 대화 히스토리를 문자열로 반환"""
    if user_id not in chat_histories:
        return "No previous conversation."

    history = chat_histories[user_id]
    if not history:
        return "No previous conversation."

    # 최근 5개 대화만 포함 (컨텍스트 길이 제한)
    recent_history = history[-5:] if len(history) > 5 else history

    formatted_history = []
    for exchange in recent_history:
        formatted_history.append(f"Human: {exchange['human']}")
        formatted_history.append(f"Assistant: {exchange['assistant']}")

    return "\n".join(formatted_history)


def add_to_chat_history(user_id: str, human_input: str, assistant_output: str):
    """대화 히스토리에 새로운 대화 추가"""
    if user_id not in chat_histories:
        chat_histories[user_id] = []

    chat_histories[user_id].append(
        {"human": human_input, "assistant": assistant_output}
    )


# LangGraph 에이전트 저장소 (사용자별)
agent_store: Dict[str, Any] = {}


def create_agent(user_id: str, openai_api_key: str, cookies: dict = None):
    """사용자별 LangGraph ReAct Agent 생성"""

    # 이미 생성된 에이전트가 있으면 재사용
    if user_id in agent_store:
        return agent_store[user_id]

    # LLM 초기화
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)

    # 모든 도구 수집
    tools = []

    try:
        # Google Calendar 도구 (쿠키에서 액세스 토큰 사용)
        calendar_tools = create_calendar_tools(user_id, cookies)
        tools.extend(calendar_tools)
        print(f"✅ Calendar 도구 {len(calendar_tools)}개 로드됨")
    except Exception as e:
        print(f"❌ Calendar 도구 초기화 실패: {e}")

    try:
        # Google Drive 도구 (쿠키에서 액세스 토큰 사용)
        drive_tools = create_drive_tools(user_id, cookies)
        tools.extend(drive_tools)
        print(f"✅ Drive 도구 {len(drive_tools)}개 로드됨")
    except Exception as e:
        print(f"❌ Drive 도구 초기화 실패: {e}")

    try:
        # Slack 도구 (DB에서 user_id로 토큰 조회)
        slack_tools = create_slack_tools(user_id)
        tools.extend(slack_tools)
        print(f"✅ Slack 도구 {len(slack_tools)}개 로드됨")
    except Exception as e:
        print(f"❌ Slack 도구 초기화 실패: {e}")

    try:
        # Notion 도구 (DB에서 user_id로 토큰 조회)
        notion_tools = create_notion_tools(user_id)
        tools.extend(notion_tools)
        print(f"✅ Notion 도구 {len(notion_tools)}개 로드됨")
    except Exception as e:
        print(f"❌ Notion 도구 초기화 실패: {e}")

    if not tools:
        raise Exception("사용 가능한 도구가 없습니다. 토큰을 확인해주세요.")

    print(f"🔧 총 {len(tools)}개 도구로 에이전트 생성")

    try:
        # 내부 RAG 도구
        tools.extend(rag_tools)
        print("✅ 내부 문서 RAG 도구 로드됨")
    except Exception as e:
        print(f"❌ 내부 문서 RAG 도구 초기화 실패: {e}")

    try:
        # 사용자별 Notion RAG 도구
        notion_rag_tool = create_notion_rag_tool_for_user(user_id)
        tools.append(notion_rag_tool)
        print("✅ 사용자별 Notion RAG 도구 로드됨")
    except Exception as e:
        print(f"❌ Notion RAG 도구 초기화 실패: {e}")

    # 현재 날짜 정보 가져오기
    date_info = get_current_date_info()

    # 시스템 메시지 포맷팅
    system_message = SYSTEM_MESSAGE.format(
        current_date=date_info["current_date"],
        current_time=date_info["current_time"],
        day_of_week=date_info["day_of_week"],
        yesterday_date=date_info["yesterday_date"],
        tomorrow_date=date_info["tomorrow_date"],
    )

    # 메모리 저장소 설정 (대화 히스토리 관리)
    memory = MemorySaver()

    # LangGraph ReAct 에이전트 생성
    agent = create_react_agent(
        model=llm, tools=tools, prompt=system_message, checkpointer=memory
    )

    # 에이전트 저장
    agent_store[user_id] = agent

    return agent


def run_agent(user_id: str, openai_api_key: str, query: str, cookies: dict = None):
    """LangGraph ReAct Agent 실행"""
    try:
        agent = create_agent(user_id, openai_api_key, cookies)

        # LangSmith 콜백 설정
        callbacks = []
        if os.getenv("LANGCHAIN_TRACING_V2") == "true":
            callbacks.append(
                LangChainTracer(
                    project_name=os.getenv("LANGCHAIN_PROJECT", "caesar-agent")
                )
            )

        # 대화 히스토리를 고려한 스레드 ID 생성
        thread_id = f"thread_{user_id}"
        config = {"configurable": {"thread_id": thread_id}, "callbacks": callbacks}

        # LangGraph 에이전트 실행 (비용 추적 포함)
        with get_openai_callback() as cb:
            # 사용자 메시지로 에이전트 호출
            result = agent.invoke({"messages": [("user", query)]}, config=config)

            # 비용 추적 로그
            if cb.total_cost > 0:
                print(f"💰 OpenAI 비용: ${cb.total_cost:.4f} (토큰: {cb.total_tokens})")

        # 최종 응답 추출
        if result and "messages" in result:
            messages = result["messages"]
            # 마지막 AI 메시지 찾기
            for msg in reversed(messages):
                if hasattr(msg, "type") and msg.type == "ai":
                    output = msg.content
                    break
                elif isinstance(msg, tuple) and msg[0] == "assistant":
                    output = msg[1]
                    break
            else:
                output = "응답을 생성할 수 없습니다."
        else:
            output = "응답을 생성할 수 없습니다."

        # 대화 히스토리에 추가
        add_to_chat_history(user_id, query, output)

        # 중간 단계 및 RAG 결과 추출
        intermediate_steps = []
        rag_results = []

        if result and "messages" in result:
            for msg in result["messages"]:
                if hasattr(msg, "type") and msg.type == "tool":
                    intermediate_steps.append(f"도구 사용: {msg.name}")
                    # RAG 도구의 응답에서 결과 추출
                    if hasattr(msg, "content") and msg.name in [
                        "notion_rag_search",
                        "internal_rag_search",
                    ]:
                        try:
                            print(
                                f"🔍 RAG 도구 응답: {msg.name} -> {type(msg.content)}"
                            )
                            print(f"🔍 RAG 내용: {str(msg.content)[:300]}...")

                            # 도구 응답이 문자열이면 바로 사용
                            if (
                                isinstance(msg.content, str)
                                and len(msg.content.strip()) > 0
                            ):
                                rag_results.append(
                                    {
                                        "source": msg.name,
                                        "content": msg.content.strip(),
                                        "score": 0.85,
                                    }
                                )
                            elif isinstance(msg.content, list):
                                # 리스트 형태의 검색 결과 처리
                                for idx, item in enumerate(msg.content):
                                    if hasattr(item, "page_content"):
                                        rag_results.append(
                                            {
                                                "source": f"{msg.name}_result_{idx+1}",
                                                "content": item.page_content,
                                                "score": getattr(
                                                    item, "metadata", {}
                                                ).get("score", 0.8),
                                            }
                                        )
                                    elif isinstance(item, dict):
                                        rag_results.append(item)
                                    elif isinstance(item, str):
                                        rag_results.append(
                                            {
                                                "source": f"{msg.name}_result_{idx+1}",
                                                "content": item,
                                                "score": 0.8,
                                            }
                                        )
                            elif isinstance(msg.content, dict):
                                rag_results.append(msg.content)

                            print(f"🔍 추출된 RAG 결과 수: {len(rag_results)}")
                        except Exception as e:
                            print(f"RAG 결과 파싱 오류: {e}")
                            print(f"RAG 메시지 상세: {msg}")

                elif hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        intermediate_steps.append(f"도구 호출: {tool_call['name']}")

        return {
            "success": True,
            "output": output,
            "intermediate_steps": intermediate_steps,
            "rag_results": rag_results,
            "chat_history": chat_histories.get(user_id, []),
        }

    except Exception as e:
        import traceback

        error_message = f"Agent 실행 중 오류가 발생했습니다: {str(e)}"
        print(f"❌ Agent 오류: {e}")
        print(f"🔍 트레이스백: {traceback.format_exc()}")

        # 에러도 히스토리에 기록
        add_to_chat_history(user_id, query, error_message)

        return {
            "success": False,
            "error": str(e),
            "output": error_message,
            "chat_history": chat_histories.get(user_id, []),
        }


def clear_chat_history(user_id: str):
    """사용자의 대화 히스토리 및 에이전트 상태 초기화"""
    if user_id in chat_histories:
        chat_histories[user_id] = []
        print(f"✅ {user_id}의 대화 히스토리가 초기화되었습니다.")
    else:
        print(f"📭 {user_id}의 대화 히스토리가 없습니다.")

    # 에이전트 캐시도 초기화 (새로운 메모리 상태로 시작)
    if user_id in agent_store:
        del agent_store[user_id]
        print(f"✅ {user_id}의 에이전트 상태가 초기화되었습니다.")


def get_chat_history(user_id: str) -> List[Dict[str, str]]:
    """사용자의 대화 히스토리 반환"""
    return chat_histories.get(user_id, [])


def clear_agent_cache(user_id: str = None):
    """에이전트 캐시 초기화 (특정 사용자 또는 전체)"""
    if user_id:
        if user_id in agent_store:
            del agent_store[user_id]
            print(f"✅ {user_id}의 에이전트 캐시가 초기화되었습니다.")
    else:
        agent_store.clear()
        print("✅ 모든 에이전트 캐시가 초기화되었습니다.")
