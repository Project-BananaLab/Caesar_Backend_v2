# app/features/chat/router/chat.py
from fastapi import APIRouter, HTTPException, Depends, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.utils.db import get_db
from app.features.chat.models.chat_models import Chat
from app.features.channel.models.channel_models import Channel
from app.features.chat.schemas.chat_schemas import (
    ChatCreate,
    ChatResponse,
    ChatListResponse,
    ChatUpdate,
)
import json

router = APIRouter(prefix="/chats", tags=["Chat"])


# only update messages
@router.put("/{chat_id}", response_model=ChatResponse)
async def update_chat(
    chat_id: int, chat_data: ChatUpdate, db: Session = Depends(get_db)
):
    """채팅 업데이트 (messages 배열 통째로 저장)"""
    try:
        chat = db.query(Chat).filter(Chat.id == chat_id).first()
        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"채팅 ID {chat_id}를 찾을 수 없습니다.",
            )
        # 기존 메시지 가져오기 (이미 dict 형태로 저장되어 있음)
        existing_messages = (
            chat.messages
            if isinstance(chat.messages, list)
            else json.loads(chat.messages)
        )

        # 새 메시지를 dict 형태로 변환
        new_messages_dict = [message.dict() for message in chat_data.messages]
        print(f"💾 업데이트할 새 메시지: {new_messages_dict}")

        # 기존 메시지에 새 메시지 추가
        updated_messages = existing_messages + new_messages_dict
        print(f"💾 최종 업데이트된 메시지: {updated_messages}")
        chat.messages = updated_messages
        db.commit()
        return ChatResponse.from_orm(chat)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"채팅 메시지 업데이트 중 오류가 발생했습니다: {str(e)}",
        )


@router.post("/", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
async def create_chat(chat_data: ChatCreate, db: Session = Depends(get_db)):
    """채널에 메시지 세션 생성 (messages 배열 통째로 저장)"""
    try:
        # 채널 존재 확인
        channel = db.query(Channel).filter(Channel.id == chat_data.channel_id).first()
        if not channel:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"채널 ID {chat_data.channel_id}를 찾을 수 없습니다.",
            )

        # 메시지 데이터를 dict 형태로 변환
        messages_dict = [message.dict() for message in chat_data.messages]
        print(f"💾 저장할 메시지 데이터: {messages_dict}")

        # 새 채팅 생성
        new_chat = Chat(channel_id=chat_data.channel_id, messages=messages_dict)

        db.add(new_chat)
        db.commit()
        db.refresh(new_chat)

        return ChatResponse.from_orm(new_chat)

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"채팅 생성 중 오류가 발생했습니다: {str(e)}",
        )


@router.get("/{chat_id}", response_model=ChatResponse)
async def get_chat(chat_id: int, db: Session = Depends(get_db)):
    """특정 chat 조회"""
    try:
        chat = db.query(Chat).filter(Chat.id == chat_id).first()

        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"채팅 ID {chat_id}를 찾을 수 없습니다.",
            )

        return ChatResponse.from_orm(chat)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"채팅 조회 중 오류가 발생했습니다: {str(e)}",
        )


@router.get("/", response_model=ChatListResponse)
async def get_chats_by_channel(
    channel_id: int = Query(..., description="채널 ID (필수)"),
    db: Session = Depends(get_db),
):
    """특정 채널의 모든 chat 조회"""
    try:
        # 채널 존재 확인
        channel = db.query(Channel).filter(Channel.id == channel_id).first()
        if not channel:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"채널 ID {channel_id}를 찾을 수 없습니다.",
            )

        # 해당 채널의 채팅들 조회
        chats = (
            db.query(Chat)
            .filter(Chat.channel_id == channel_id)
            .order_by(Chat.created_at.desc())
            .all()
        )

        # 불러온 채팅 메시지에 previewFile 정보가 있는지 확인
        for chat in chats:
            for msg in chat.messages:
                if isinstance(msg, dict) and msg.get("previewFile"):
                    print(
                        f"📂 불러온 메시지에 previewFile 있음: {msg.get('previewFile')}"
                    )
                    break

        return ChatListResponse(
            chats=[ChatResponse.from_orm(chat) for chat in chats],
            total=len(chats),
            channel_id=channel_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"채팅 조회 중 오류가 발생했습니다: {str(e)}",
        )


@router.delete("/{chat_id}")
async def delete_chat(chat_id: int, db: Session = Depends(get_db)):
    """채팅 삭제"""
    try:
        chat = db.query(Chat).filter(Chat.id == chat_id).first()

        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"채팅 ID {chat_id}를 찾을 수 없습니다.",
            )

        db.delete(chat)
        db.commit()

        return {
            "message": f"채팅 ID {chat_id}가 성공적으로 삭제되었습니다.",
            "deleted_chat_id": chat_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"채팅 삭제 중 오류가 발생했습니다: {str(e)}",
        )
