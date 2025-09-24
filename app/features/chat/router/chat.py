# app/features/chat/chat.py
"""
Chat API - 채팅 관련 CRUD 엔드포인트
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime
from pydantic import Field

# 새로운 스키마 임포트
from app.features.chat.schemas.chat_schemas import (
    ChatCreate,
    ChatUpdate,
    ChatResponse,
    ChatListResponse,
    ChatSearchRequest,
)

# FastAPI 라우터 생성
router = APIRouter(prefix="/chats", tags=["Chat"])


# 메모리 기반 임시 데이터 저장소 (실제 프로젝트에서는 DB 사용)
from typing import List

chats_db: List[dict] = []
chat_id_counter = 1


@router.post("/", response_model=ChatResponse, status_code=201)
async def create_chat(chat: ChatCreate):
    """
    새로운 채팅 생성

    Args:
        chat: 채팅 생성 데이터

    Returns:
        생성된 채팅 정보

    Raises:
        HTTPException: 채널이 존재하지 않는 경우 404
    """
    global chat_id_counter

    # 여기서는 간단히 구현. 실제로는 channel 존재 여부를 확인해야 함
    # 임시로 channel_id가 1-100 범위에 있다고 가정
    if chat.channel_id < 1 or chat.channel_id > 100:
        raise HTTPException(
            status_code=404, detail=f"Channel with id {chat.channel_id} not found"
        )

    # 새 채팅 생성
    new_chat = {
        "id": chat_id_counter,
        "channel_id": chat.channel_id,
        "user_id": getattr(chat, "user_id", None),  # 옵셔널 필드
        "message": chat.message,
        "message_type": getattr(chat, "message_type", "text"),
        "is_edited": False,
        "is_deleted": False,
        "created_at": datetime.now(),
        "updated_at": None,
        "channel_name": None,  # JOIN 결과용 (실제 DB에서는 JOIN으로 가져옴)
        "user_name": None,  # JOIN 결과용
    }

    chats_db.append(new_chat)
    chat_id_counter += 1

    return ChatResponse(**new_chat)


@router.get("/", response_model=ChatListResponse)
async def get_all_chats(
    channel_id: Optional[int] = Query(None, description="필터링할 채널 ID"),
    limit: int = Query(100, ge=1, le=1000, description="최대 조회 개수"),
    skip: int = Query(0, ge=0, description="건너뛸 개수"),
):
    """
    모든 채팅 조회 (옵션: 특정 채널의 채팅만 조회)

    Args:
        channel_id: 필터링할 채널 ID (선택사항)
        limit: 최대 조회 개수
        skip: 건너뛸 개수

    Returns:
        채팅 목록과 총 개수
    """
    # 채널 ID로 필터링 (선택사항)
    filtered_chats = chats_db
    if channel_id is not None:
        filtered_chats = [chat for chat in chats_db if chat["channel_id"] == channel_id]

    # 페이징 적용
    total = len(filtered_chats)
    paginated_chats = filtered_chats[skip : skip + limit]

    # 응답 생성
    chat_responses = [ChatResponse(**chat) for chat in paginated_chats]

    # 페이지 정보 계산
    page = (skip // limit) + 1
    total_pages = (total + limit - 1) // limit

    return ChatListResponse(
        chats=chat_responses,
        total=total,
        page=page,
        page_size=limit,
        total_pages=total_pages,
    )


@router.get("/{chat_id}", response_model=ChatResponse)
async def get_chat(chat_id: int):
    """
    특정 채팅 조회

    Args:
        chat_id: 채팅 ID

    Returns:
        채팅 정보

    Raises:
        HTTPException: 채팅이 존재하지 않는 경우 404
    """
    for chat in chats_db:
        if chat["id"] == chat_id:
            return ChatResponse(**chat)

    raise HTTPException(status_code=404, detail=f"Chat with id {chat_id} not found")


@router.delete("/{chat_id}", status_code=204)
async def delete_chat(chat_id: int):
    """
    채팅 삭제

    Args:
        chat_id: 삭제할 채팅 ID

    Raises:
        HTTPException: 채팅이 존재하지 않는 경우 404
    """
    for i, chat in enumerate(chats_db):
        if chat["id"] == chat_id:
            chats_db.pop(i)
            return

    raise HTTPException(status_code=404, detail=f"Chat with id {chat_id} not found")


@router.get("/channel/{channel_id}", response_model=ChatListResponse)
async def get_chats_by_channel(
    channel_id: int,
    limit: int = Query(100, ge=1, le=1000, description="최대 조회 개수"),
    skip: int = Query(0, ge=0, description="건너뛸 개수"),
):
    """
    특정 채널의 모든 채팅 조회

    Args:
        channel_id: 채널 ID
        limit: 최대 조회 개수
        skip: 건너뛸 개수

    Returns:
        해당 채널의 채팅 목록
    """
    return await get_all_chats(channel_id=channel_id, limit=limit, skip=skip)
