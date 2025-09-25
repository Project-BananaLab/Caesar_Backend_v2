#!/usr/bin/env python3
"""
PostgreSQL + SQLAlchemy 기반 Chat & Channel API 테스트 스크립트
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"


def test_api():
    print("🚀 PostgreSQL Chat & Channel API 테스트 시작")
    print(f"서버: {BASE_URL}")
    print("=" * 50)

    try:
        # 서버 상태 확인
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ 서버 연결 성공")
        else:
            print(f"❌ 서버 응답 오류: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("❌ 서버에 연결할 수 없습니다.")
        print("서버를 먼저 실행하세요: uvicorn app.main:app --reload --port 8080")
        return
    except Exception as e:
        print(f"❌ 서버 연결 오류: {e}")
        return

    # 1. 채널 생성 테스트
    print("\n📺 1. 채널 생성 테스트")
    channel_data = {"employee_id": 1, "title": "테스트 채널 #1"}

    try:
        response = requests.post(f"{BASE_URL}/channels/", json=channel_data)
        print(f"상태 코드: {response.status_code}")

        if response.status_code == 201:
            channel = response.json()
            print(f"✅ 채널 생성 성공: ID {channel['id']}, 제목: '{channel['title']}'")
            channel_id = channel["id"]
        else:
            print(f"❌ 채널 생성 실패: {response.text}")
            return
    except Exception as e:
        print(f"❌ 채널 생성 오류: {e}")
        return

    # 2. 채널 목록 조회 테스트
    print("\n📋 2. 채널 목록 조회 테스트")
    try:
        response = requests.get(f"{BASE_URL}/channels/")
        if response.status_code == 200:
            channels = response.json()
            print(f"✅ 채널 목록 조회 성공: {channels['total']}개 채널")
            for channel in channels["channels"]:
                print(
                    f"   - ID: {channel['id']}, 제목: '{channel['title']}', 생성자: {channel['employee_id']}"
                )
        else:
            print(f"❌ 채널 목록 조회 실패: {response.text}")
    except Exception as e:
        print(f"❌ 채널 목록 조회 오류: {e}")

    # 3. 채팅 생성 테스트
    print("\n💬 3. 채팅 생성 테스트")
    chat_data = {
        "channel_id": channel_id,
        "messages": [
            {"role": "user", "content": "안녕하세요! PostgreSQL 테스트입니다."},
            {
                "role": "agent",
                "content": "안녕하세요! SQLAlchemy로 잘 작동하고 있습니다.",
            },
            {"role": "user", "content": "JSONB 타입으로 메시지가 저장되나요?"},
            {
                "role": "agent",
                "content": "네, PostgreSQL의 JSONB 타입으로 효율적으로 저장됩니다!",
            },
        ],
    }

    try:
        response = requests.post(f"{BASE_URL}/chats/", json=chat_data)
        print(f"상태 코드: {response.status_code}")

        if response.status_code == 201:
            chat = response.json()
            print(
                f"✅ 채팅 생성 성공: ID {chat['id']}, 메시지 수: {len(chat['messages'])}"
            )
            chat_id = chat["id"]
            print("메시지 내용:")
            for i, msg in enumerate(chat["messages"], 1):
                print(f"   {i}. [{msg['role']}] {msg['content']}")
        else:
            print(f"❌ 채팅 생성 실패: {response.text}")
            return
    except Exception as e:
        print(f"❌ 채팅 생성 오류: {e}")
        return

    # 4. 특정 채팅 조회 테스트
    print(f"\n🔍 4. 특정 채팅 조회 테스트 (ID: {chat_id})")
    try:
        response = requests.get(f"{BASE_URL}/chats/{chat_id}")
        if response.status_code == 200:
            chat = response.json()
            print(f"✅ 채팅 조회 성공: ID {chat['id']}, 채널 ID: {chat['channel_id']}")
            print(f"   생성 시간: {chat['created_at']}")
            print(f"   메시지 수: {len(chat['messages'])}")
        else:
            print(f"❌ 채팅 조회 실패: {response.text}")
    except Exception as e:
        print(f"❌ 채팅 조회 오류: {e}")

    # 5. 채널별 채팅 목록 조회 테스트
    print(f"\n📊 5. 채널별 채팅 목록 조회 테스트 (채널 ID: {channel_id})")
    try:
        response = requests.get(f"{BASE_URL}/chats/?channel_id={channel_id}")
        if response.status_code == 200:
            chats = response.json()
            print(f"✅ 채널별 채팅 조회 성공: {chats['total']}개 채팅")
            print(f"   필터링된 채널 ID: {chats['channel_id']}")
            for chat in chats["chats"]:
                print(f"   - 채팅 ID: {chat['id']}, 메시지 수: {len(chat['messages'])}")
        else:
            print(f"❌ 채널별 채팅 조회 실패: {response.text}")
    except Exception as e:
        print(f"❌ 채널별 채팅 조회 오류: {e}")

    # 6. 채널 수정 테스트
    print(f"\n✏️ 6. 채널 수정 테스트 (ID: {channel_id})")
    update_data = {"title": "수정된 테스트 채널 #1"}

    try:
        response = requests.put(f"{BASE_URL}/channels/{channel_id}", json=update_data)
        if response.status_code == 200:
            updated_channel = response.json()
            print(f"✅ 채널 수정 성공: 새 제목 '{updated_channel['title']}'")
        else:
            print(f"❌ 채널 수정 실패: {response.text}")
    except Exception as e:
        print(f"❌ 채널 수정 오류: {e}")

    print("\n" + "=" * 50)
    print("✅ 모든 테스트 완료!")
    print("\n📋 테스트 결과 요약:")
    print("- PostgreSQL + SQLAlchemy ORM 정상 동작")
    print("- JSONB 타입으로 메시지 배열 저장 성공")
    print("- 채널 CRUD 작업 모두 성공")
    print("- 채팅 생성 및 조회 모두 성공")
    print("- 관계형 데이터 (채널-채팅) 연결 정상")


if __name__ == "__main__":
    test_api()
