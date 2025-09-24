# app/features/channel/channel.py
"""
Channel API - 채널 관련 CRUD 엔드포인트
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime

# 새로운 스키마 임포트
from app.features.channel.schemas.channel_schemas import (
    ChannelCreate,
    ChannelUpdate,
    ChannelResponse,
    ChannelListResponse,
    ChannelDeleteResponse,
    ChannelSearchRequest,
)

# FastAPI 라우터 생성
router = APIRouter(prefix="/channels", tags=["Channel"])


# 메모리 기반 임시 데이터 저장소 (실제 프로젝트에서는 DB 사용)
channels_db: List[dict] = []
channel_id_counter = 1


# 기본 채널 데이터 추가
def init_default_channels():
    """기본 채널 데이터 초기화"""
    global channel_id_counter
    if not channels_db:
        default_channels = [
            {
                "id": 1,
                "name": "일반",
                "description": "일반적인 대화를 위한 채널입니다.",
                "channel_type": "public",
                "is_active": True,
                "max_members": None,
                "created_by": None,
                "created_at": datetime.now(),
                "updated_at": None,
                "member_count": 0,
                "message_count": 0,
                "last_message_at": None,
            },
            {
                "id": 2,
                "name": "공지사항",
                "description": "중요한 공지사항을 전달하는 채널입니다.",
                "channel_type": "public",
                "is_active": True,
                "max_members": None,
                "created_by": None,
                "created_at": datetime.now(),
                "updated_at": None,
                "member_count": 0,
                "message_count": 0,
                "last_message_at": None,
            },
            {
                "id": 3,
                "name": "자유게시판",
                "description": "자유롭게 이야기할 수 있는 채널입니다.",
                "channel_type": "public",
                "is_active": True,
                "max_members": None,
                "created_by": None,
                "created_at": datetime.now(),
                "updated_at": None,
                "member_count": 0,
                "message_count": 0,
                "last_message_at": None,
            },
        ]
        channels_db.extend(default_channels)
        channel_id_counter = 4


# 서버 시작 시 기본 채널 생성
init_default_channels()


@router.get("/", response_model=ChannelListResponse)
async def get_all_channels(
    limit: int = Query(100, ge=1, le=1000, description="최대 조회 개수"),
    skip: int = Query(0, ge=0, description="건너뛸 개수"),
):
    """
    모든 채널 조회

    Args:
        limit: 최대 조회 개수
        skip: 건너뛸 개수

    Returns:
        채널 목록과 총 개수
    """
    # 페이징 적용
    total = len(channels_db)
    paginated_channels = channels_db[skip : skip + limit]

    # 응답 생성
    channel_responses = [ChannelResponse(**channel) for channel in paginated_channels]

    # 페이지 정보 계산
    page = (skip // limit) + 1
    total_pages = (total + limit - 1) // limit

    return ChannelListResponse(
        channels=channel_responses,
        total=total,
        page=page,
        page_size=limit,
        total_pages=total_pages,
    )


@router.post("/", response_model=ChannelResponse, status_code=201)
async def create_channel(channel: ChannelCreate):
    """
    새로운 채널 생성

    Args:
        channel: 채널 생성 데이터

    Returns:
        생성된 채널 정보

    Raises:
        HTTPException: 같은 이름의 채널이 이미 존재하는 경우 400
    """
    global channel_id_counter

    # 채널 이름 중복 확인
    for existing_channel in channels_db:
        if existing_channel["name"].lower() == channel.name.lower():
            raise HTTPException(
                status_code=400,
                detail=f"Channel with name '{channel.name}' already exists",
            )

    # 새 채널 생성
    new_channel = {
        "id": channel_id_counter,
        "name": channel.name,
        "description": getattr(channel, "description", None),
        "channel_type": getattr(channel, "channel_type", "public"),
        "is_active": getattr(channel, "is_active", True),
        "max_members": getattr(channel, "max_members", None),
        "created_by": getattr(channel, "created_by", None),
        "created_at": datetime.now(),
        "updated_at": None,
        "member_count": 0,
        "message_count": 0,
        "last_message_at": None,
        "creator_name": None,  # JOIN 결과용
    }

    channels_db.append(new_channel)
    channel_id_counter += 1

    return ChannelResponse(**new_channel)


@router.get("/{channel_id}", response_model=ChannelResponse)
async def get_channel(channel_id: int):
    """
    특정 채널 조회

    Args:
        channel_id: 채널 ID

    Returns:
        채널 정보

    Raises:
        HTTPException: 채널이 존재하지 않는 경우 404
    """
    for channel in channels_db:
        if channel["id"] == channel_id:
            return ChannelResponse(**channel)

    raise HTTPException(
        status_code=404, detail=f"Channel with id {channel_id} not found"
    )


@router.put("/{channel_id}", response_model=ChannelResponse)
async def update_channel(channel_id: int, channel_update: ChannelUpdate):
    """
    채널 이름 수정

    Args:
        channel_id: 수정할 채널 ID
        channel_update: 수정할 채널 데이터

    Returns:
        수정된 채널 정보

    Raises:
        HTTPException: 채널이 존재하지 않는 경우 404, 같은 이름의 채널이 이미 존재하는 경우 400
    """
    # 채널 존재 확인
    channel_to_update = None
    channel_index = -1

    for i, channel in enumerate(channels_db):
        if channel["id"] == channel_id:
            channel_to_update = channel
            channel_index = i
            break

    if channel_to_update is None:
        raise HTTPException(
            status_code=404, detail=f"Channel with id {channel_id} not found"
        )

    # 다른 채널과 이름 중복 확인 (자기 자신 제외)
    for channel in channels_db:
        if (
            channel["id"] != channel_id
            and channel["name"].lower() == channel_update.name.lower()
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Channel with name '{channel_update.name}' already exists",
            )

    # 채널 정보 업데이트
    channels_db[channel_index]["name"] = channel_update.name
    channels_db[channel_index]["updated_at"] = datetime.now()

    return ChannelResponse(**channels_db[channel_index])


@router.delete("/{channel_id}", response_model=ChannelDeleteResponse)
async def delete_channel(channel_id: int):
    """
    채널 삭제

    Args:
        channel_id: 삭제할 채널 ID

    Returns:
        삭제 결과 메시지

    Raises:
        HTTPException: 채널이 존재하지 않는 경우 404
    """
    for i, channel in enumerate(channels_db):
        if channel["id"] == channel_id:
            deleted_channel = channels_db.pop(i)
            return ChannelDeleteResponse(
                message=f"Channel '{deleted_channel['name']}' has been successfully deleted",
                deleted_channel_id=channel_id,
            )

    raise HTTPException(
        status_code=404, detail=f"Channel with id {channel_id} not found"
    )


@router.get("/search/{name}", response_model=ChannelListResponse)
async def search_channels_by_name(
    name: str,
    limit: int = Query(100, ge=1, le=1000, description="최대 조회 개수"),
    skip: int = Query(0, ge=0, description="건너뛸 개수"),
):
    """
    채널 이름으로 검색

    Args:
        name: 검색할 채널 이름 (부분 검색 지원)
        limit: 최대 조회 개수
        skip: 건너뛸 개수

    Returns:
        검색된 채널 목록
    """
    # 이름으로 필터링 (대소문자 구분 없이 부분 검색)
    filtered_channels = [
        channel for channel in channels_db if name.lower() in channel["name"].lower()
    ]

    # 페이징 적용
    total = len(filtered_channels)
    paginated_channels = filtered_channels[skip : skip + limit]

    # 응답 생성
    channel_responses = [ChannelResponse(**channel) for channel in paginated_channels]

    # 페이지 정보 계산
    page = (skip // limit) + 1
    total_pages = (total + limit - 1) // limit

    return ChannelListResponse(
        channels=channel_responses,
        total=total,
        page=page,
        page_size=limit,
        total_pages=total_pages,
    )
