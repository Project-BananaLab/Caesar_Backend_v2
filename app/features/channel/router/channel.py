# app/features/channel/router/channel.py
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from typing import List
from app.utils.db import get_db
from app.features.channel.models.channel_models import Channel
from app.features.channel.schemas.channel_schemas import (
    ChannelCreate,
    ChannelUpdate,
    ChannelResponse,
    ChannelListResponse,
    ChannelDeleteResponse,
)

router = APIRouter(prefix="/channels", tags=["Channel"])


@router.get("/", response_model=ChannelListResponse)
async def get_channels_by_employee(
    employee_id: int = None, db: Session = Depends(get_db)
):
    """사용자별 채널 조회"""
    try:
        if employee_id:
            # 특정 사용자의 채널만 조회
            channels = (
                db.query(Channel)
                .filter(Channel.employee_id == employee_id)
                .order_by(Channel.created_at.desc())
                .all()
            )
        else:
            # employee_id가 없으면 전체 조회 (관리자용)
            channels = db.query(Channel).order_by(Channel.created_at.desc()).all()

        return ChannelListResponse(
            channels=[ChannelResponse.from_orm(channel) for channel in channels],
            total=len(channels),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"채널 조회 중 오류가 발생했습니다: {str(e)}",
        )


@router.post("/", response_model=ChannelResponse, status_code=status.HTTP_201_CREATED)
async def create_channel(channel_data: ChannelCreate, db: Session = Depends(get_db)):
    """새 채널 생성"""
    try:
        # 새 채널 생성
        new_channel = Channel(
            employee_id=channel_data.employee_id, title=channel_data.title
        )

        db.add(new_channel)
        db.commit()
        db.refresh(new_channel)

        return ChannelResponse.from_orm(new_channel)

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"채널 생성 중 오류가 발생했습니다: {str(e)}",
        )


@router.get("/{channel_id}", response_model=ChannelResponse)
async def get_channel(channel_id: int, db: Session = Depends(get_db)):
    """특정 채널 조회"""
    try:
        channel = db.query(Channel).filter(Channel.id == channel_id).first()

        if not channel:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"채널 ID {channel_id}를 찾을 수 없습니다.",
            )

        return ChannelResponse.from_orm(channel)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"채널 조회 중 오류가 발생했습니다: {str(e)}",
        )


@router.put("/{channel_id}", response_model=ChannelResponse)
async def update_channel(
    channel_id: int, channel_data: ChannelUpdate, db: Session = Depends(get_db)
):
    """채널 제목 수정"""
    try:
        channel = db.query(Channel).filter(Channel.id == channel_id).first()

        if not channel:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"채널 ID {channel_id}를 찾을 수 없습니다.",
            )

        # 제목 업데이트 (None이 아닌 경우에만)
        if channel_data.title is not None:
            channel.title = channel_data.title

        db.commit()
        db.refresh(channel)

        return ChannelResponse.from_orm(channel)

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"채널 수정 중 오류가 발생했습니다: {str(e)}",
        )


@router.delete("/{channel_id}", response_model=ChannelDeleteResponse)
async def delete_channel(channel_id: int, db: Session = Depends(get_db)):
    """채널 삭제"""
    try:
        channel = db.query(Channel).filter(Channel.id == channel_id).first()

        if not channel:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"채널 ID {channel_id}를 찾을 수 없습니다.",
            )

        db.delete(channel)
        db.commit()

        return ChannelDeleteResponse(
            message=f"채널 '{channel.title or channel_id}'가 성공적으로 삭제되었습니다.",
            deleted_channel_id=channel_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"채널 삭제 중 오류가 발생했습니다: {str(e)}",
        )
