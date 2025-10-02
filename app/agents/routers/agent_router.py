# backend/routers/agent_router.py
"""
Agent 관련 FastAPI 라우터
simple_test.py의 질문→응답 로직을 API 엔드포인트로 구현
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import traceback
from app.agents.agent_manager import get_agent_manager, get_agent_stats
from app.agents.agent import run_agent, clear_chat_history, get_chat_history


# FastAPI 라우터 생성
router = APIRouter(prefix="/agent", tags=["Agent"])


# 요청/응답 모델 정의
class AgentQueryRequest(BaseModel):
    """Agent 질의 요청 모델"""

    user_id: str = Field(..., description="사용자 ID")
    query: str = Field(..., description="사용자 질문")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API 키 (선택사항)")


class AgentQueryResponse(BaseModel):
    """Agent 질의 응답 모델"""

    success: bool = Field(..., description="성공 여부")
    user_id: str = Field(..., description="사용자 ID")
    query: str = Field(..., description="사용자 질문")
    output: str = Field(..., description="Agent 응답")
    intermediate_steps: List[str] = Field(default=[], description="중간 실행 단계")
    rag_results: Optional[List[Dict[str, Any]]] = Field(
        default=[], description="RAG 검색 결과"
    )
    sources: Optional[List[Dict[str, Any]]] = Field(
        default=[], description="추출된 문서 소스 정보"
    )
    drive_files: Optional[List[Dict[str, Any]]] = Field(
        default=[], description="구글 드라이브 파일 정보"
    )
    total_conversations: int = Field(..., description="총 대화 수")
    cost_info: Optional[Dict[str, Any]] = Field(None, description="비용 정보")


class ChatHistoryResponse(BaseModel):
    """대화 히스토리 응답 모델"""

    user_id: str = Field(..., description="사용자 ID")
    total_conversations: int = Field(..., description="총 대화 수")
    chat_history: List[Dict[str, str]] = Field(..., description="대화 히스토리")


class AgentStatsResponse(BaseModel):
    """Agent 통계 응답 모델"""

    total_agents: int = Field(..., description="총 Agent 수")
    user_list: List[str] = Field(..., description="사용자 목록")
    default_api_key_set: bool = Field(..., description="기본 API 키 설정 여부")


@router.post("/query", response_model=AgentQueryResponse)
async def agent_query(request: AgentQueryRequest, http_request: Request):
    """
    Agent에게 질문하고 응답 받기
    simple_test.py의 핵심 기능을 API 엔드포인트로 구현
    """
    try:
        print(f"🔍 Agent 질의 요청: {request.user_id} - {request.query[:50]}...")

        # 쿠키에서 토큰 정보 추출
        cookies = http_request.cookies
        print(f"🍪 수신된 쿠키: {list(cookies.keys())[:5]}...")  # 일부만 로그

        # OpenAI API 키 검증
        api_key = request.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="OpenAI API 키가 필요합니다. 요청에 포함하거나 환경변수 OPENAI_API_KEY를 설정하세요.",
            )

        # 입력 검증
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

        # Agent 실행 (쿠키 데이터를 포함하여 전달)
        print(f"🤖 Caesar Agent 실행 중... (사용자: {request.user_id})")
        result = run_agent(
            user_id=request.user_id,
            openai_api_key=api_key,
            query=request.query,
            cookies=cookies,
        )

        if result["success"]:
            # 성공 응답
            chat_history = result.get("chat_history", [])

            # 안전한 데이터 추출
            try:
                drive_files = result.get("drive_files", [])
                if not isinstance(drive_files, list):
                    print(f"⚠️ drive_files가 리스트가 아님: {type(drive_files)}")
                    drive_files = []

                sources = result.get("sources", [])
                if not isinstance(sources, list):
                    print(f"⚠️ sources가 리스트가 아님: {type(sources)}")
                    sources = []

                print(f"✅ drive_files 안전 처리 완료: {len(drive_files)}개")
                print(f"✅ sources 안전 처리 완료: {len(sources)}개")

            except Exception as data_error:
                print(f"❌ 응답 데이터 처리 중 오류: {data_error}")
                drive_files = []
                sources = []

            return AgentQueryResponse(
                success=True,
                user_id=request.user_id,
                query=request.query,
                output=result.get("output", "응답을 생성할 수 없습니다."),
                intermediate_steps=result.get("intermediate_steps", []),
                rag_results=result.get("rag_results", []),
                sources=sources,
                drive_files=drive_files,  # 구글 드라이브 파일 정보 추가
                total_conversations=len(chat_history),
            )
        else:
            # Agent 실행 실패
            error_msg = result.get("error", "알 수 없는 오류")
            print(f"❌ Agent 실행 실패: {error_msg}")
            raise HTTPException(
                status_code=500, detail=f"Agent 실행 중 오류 발생: {error_msg}"
            )

    except HTTPException:
        # FastAPI HTTPException은 그대로 전달
        raise
    except Exception as e:
        # 예상치 못한 오류 처리
        error_msg = f"Agent 질의 처리 중 오류 발생: {str(e)}"
        print(f"❌ 예상치 못한 오류: {error_msg}")
        print(f"🔍 트레이스백: {traceback.format_exc()}")

        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/history/{user_id}", response_model=ChatHistoryResponse)
async def get_user_chat_history(user_id: str):
    """사용자의 대화 히스토리 조회"""
    try:
        history = get_chat_history(user_id)

        return ChatHistoryResponse(
            user_id=user_id, total_conversations=len(history), chat_history=history
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"대화 히스토리 조회 중 오류 발생: {str(e)}"
        )


@router.delete("/history/{user_id}")
async def clear_user_chat_history(user_id: str):
    """사용자의 대화 히스토리 초기화"""
    try:
        clear_chat_history(user_id)

        return {
            "success": True,
            "message": f"사용자 '{user_id}'의 대화 히스토리가 초기화되었습니다.",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"대화 히스토리 초기화 중 오류 발생: {str(e)}"
        )


@router.get("/stats", response_model=AgentStatsResponse)
async def get_agent_statistics():
    """Agent 관리 통계 정보 조회"""
    try:
        stats = get_agent_stats()

        return AgentStatsResponse(
            total_agents=stats["total_agents"],
            user_list=stats["user_list"],
            default_api_key_set=stats["default_api_key_set"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Agent 통계 조회 중 오류 발생: {str(e)}"
        )


@router.post("/reset/{user_id}")
async def reset_user_agent(user_id: str):
    """특정 사용자의 Agent 인스턴스 및 히스토리 초기화"""
    try:
        # Agent 매니저에서 Agent 제거
        manager = get_agent_manager()
        manager.remove_agent(user_id)

        # 대화 히스토리도 초기화
        clear_chat_history(user_id)

        return {
            "success": True,
            "message": f"사용자 '{user_id}'의 Agent와 대화 히스토리가 초기화되었습니다.",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Agent 초기화 중 오류 발생: {str(e)}"
        )


@router.post("/reset-all")
async def reset_all_agents():
    """모든 사용자의 Agent 인스턴스 초기화 (관리자용)"""
    try:
        # 모든 Agent 제거
        manager = get_agent_manager()
        user_count = manager.get_agent_count()
        user_list = manager.get_user_list().copy()

        manager.clear_all_agents()

        # 모든 사용자 대화 히스토리도 초기화
        for user_id in user_list:
            clear_chat_history(user_id)

        return {
            "success": True,
            "message": f"총 {user_count}명의 사용자 Agent와 대화 히스토리가 초기화되었습니다.",
            "reset_users": user_list,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"전체 Agent 초기화 중 오류 발생: {str(e)}"
        )


@router.get("/health")
async def agent_health_check():
    """Agent 라우터 상태 확인"""
    try:
        stats = get_agent_stats()

        return {
            "status": "healthy",
            "message": "Agent 라우터가 정상적으로 작동 중입니다.",
            "agent_stats": stats,
            "default_api_key_available": bool(os.getenv("OPENAI_API_KEY")),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Agent 상태 확인 중 오류 발생: {str(e)}"
        )
