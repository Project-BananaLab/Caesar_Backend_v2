# backend/agent_manager.py
"""
Agent 초기화 및 싱글턴 관리 모듈
사용자별 Agent 인스턴스를 생성하고 관리하는 매니저 클래스
"""

from typing import Dict, Optional
import os
import threading
from app.agents.agent import create_agent, clear_agent_cache


class AgentManager:
    """Agent 인스턴스를 관리하는 싱글턴 클래스"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(AgentManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.agents: Dict[str, any] = {}
        self.default_openai_key = os.getenv("OPENAI_API_KEY")
        self._initialized = True
        print("✅ AgentManager 싱글턴 인스턴스 초기화됨")

    def get_agent(
        self,
        user_id: str,
        openai_api_key: Optional[str] = None,
        cookies: Optional[dict] = None,
    ):
        """
        사용자별 Agent 인스턴스를 가져오거나 생성

        Args:
            user_id: 사용자 ID
            openai_api_key: OpenAI API 키 (선택사항)
            cookies: 에이전트에 전달할 쿠키 데이터

        Returns:
            Agent 인스턴스

        Raises:
            ValueError: API 키가 없는 경우
            Exception: Agent 생성 실패 시
        """
        # API 키 검증
        api_key = openai_api_key or self.default_openai_key
        if not api_key:
            raise ValueError(
                "OpenAI API 키가 필요합니다. 환경변수 OPENAI_API_KEY를 설정하거나 매개변수로 제공하세요."
            )

        # 기존 Agent가 있으면 반환
        if user_id in self.agents:
            print(f"♻️ 기존 Agent 재사용: {user_id}")
            return self.agents[user_id]

        try:
            # 새 Agent 생성 (쿠키 데이터를 포함하여 전달)
            print(f"🔧 새 Agent 생성 중: {user_id}")
            agent = create_agent(user_id, api_key, cookies)

            # 캐시에 저장
            self.agents[user_id] = agent
            print(f"✅ Agent 생성 완료: {user_id}")

            return agent

        except Exception as e:
            print(f"❌ Agent 생성 실패 ({user_id}): {e}")
            raise Exception(f"Agent 생성 중 오류 발생: {str(e)}")

    def remove_agent(self, user_id: str):
        """
        특정 사용자의 Agent 인스턴스 제거

        Args:
            user_id: 사용자 ID
        """
        if user_id in self.agents:
            del self.agents[user_id]
            print(f"🗑️ Agent 제거됨: {user_id}")

            # agent.py의 캐시도 정리
            clear_agent_cache(user_id)
        else:
            print(f"📭 제거할 Agent가 없음: {user_id}")

    def clear_all_agents(self):
        """모든 Agent 인스턴스 제거"""
        user_count = len(self.agents)
        self.agents.clear()

        # agent.py의 전체 캐시도 정리
        clear_agent_cache()

        print(f"🧹 모든 Agent 제거됨 ({user_count}개)")

    def get_agent_count(self) -> int:
        """현재 관리 중인 Agent 수 반환"""
        return len(self.agents)

    def get_user_list(self) -> list:
        """현재 Agent가 생성된 사용자 목록 반환"""
        return list(self.agents.keys())

    def is_agent_exists(self, user_id: str) -> bool:
        """특정 사용자의 Agent 존재 여부 확인"""
        return user_id in self.agents


# 싱글턴 인스턴스 생성 함수
def get_agent_manager() -> AgentManager:
    """AgentManager 싱글턴 인스턴스 반환"""
    return AgentManager()


# 편의 함수들
def get_user_agent(
    user_id: str, openai_api_key: Optional[str] = None, cookies: Optional[dict] = None
):
    """사용자 Agent 가져오기 (편의 함수)"""
    manager = get_agent_manager()
    return manager.get_agent(user_id, openai_api_key, cookies)


def remove_user_agent(user_id: str):
    """사용자 Agent 제거 (편의 함수)"""
    manager = get_agent_manager()
    manager.remove_agent(user_id)


def clear_all_user_agents():
    """모든 사용자 Agent 제거 (편의 함수)"""
    manager = get_agent_manager()
    manager.clear_all_agents()


def get_agent_stats():
    """Agent 통계 정보 반환 (편의 함수)"""
    manager = get_agent_manager()
    return {
        "total_agents": manager.get_agent_count(),
        "user_list": manager.get_user_list(),
        "default_api_key_set": bool(manager.default_openai_key),
    }
