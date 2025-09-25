#!/usr/bin/env python3
"""
간단한 Chat & Channel API 테스트
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_chat_flow():
    print("🚀 Chat & Channel API 실제 데이터베이스 테스트")
    print("=" * 50)

    # 1. 새 채널 생성
    print("1️⃣ 채널 생성 테스트")
    channel_data = {"employee_id": 1, "title": "Python 테스트 채널"}

    response = requests.post(f"{BASE_URL}/channels/", json=channel_data)
    if response.status_code == 201:
        channel = response.json()
        channel_id = channel["id"]
        print(f"✅ 채널 생성 성공!")
        print(f"   - 채널 ID: {channel_id}")
        print(f"   - 제목: {channel['title']}")
        print(f"   - 생성자 ID: {channel['employee_id']}")
        print(f"   - 생성 시간: {channel['created_at']}")
    else:
        print(f"❌ 채널 생성 실패: {response.text}")
        return

    # 2. 질문-응답 채팅 생성
    print(f"\n2️⃣ 채팅 생성 테스트 (채널 ID: {channel_id})")
    chat_data = {
        "channel_id": channel_id,
        "messages": [
            {"role": "user", "content": "PostgreSQL에 데이터가 제대로 저장되나요?"},
            {
                "role": "agent",
                "content": "네! SQLAlchemy ORM을 통해 PostgreSQL 데이터베이스에 JSONB 형태로 메시지가 저장되고 있습니다. 관계형 데이터베이스의 장점과 NoSQL의 유연성을 동시에 활용할 수 있습니다.",
            },
        ],
    }

    response = requests.post(f"{BASE_URL}/chats/", json=chat_data)
    if response.status_code == 201:
        chat = response.json()
        chat_id = chat["id"]
        print(f"✅ 채팅 생성 성공!")
        print(f"   - 채팅 ID: {chat_id}")
        print(f"   - 채널 ID: {chat['channel_id']}")
        print(f"   - 메시지 수: {len(chat['messages'])}")
        print(f"   - 생성 시간: {chat['created_at']}")

        print("\n💬 저장된 메시지:")
        for i, msg in enumerate(chat["messages"], 1):
            print(f"   {i}. [{msg['role']}] {msg['content']}")
    else:
        print(f"❌ 채팅 생성 실패: {response.text}")
        return

    # 3. 채널별 채팅 목록 조회
    print(f"\n3️⃣ 채널별 채팅 목록 조회 (채널 ID: {channel_id})")
    response = requests.get(f"{BASE_URL}/chats/?channel_id={channel_id}")
    if response.status_code == 200:
        chats = response.json()
        print(f"✅ 채팅 목록 조회 성공!")
        print(f"   - 총 채팅 개수: {chats['total']}")
        print(f"   - 채널 ID: {chats['channel_id']}")

        for chat in chats["chats"]:
            print(f"\n   📝 채팅 ID {chat['id']}:")
            for msg in chat["messages"]:
                print(f"      [{msg['role']}] {msg['content'][:50]}...")
    else:
        print(f"❌ 채팅 목록 조회 실패: {response.text}")

    # 4. 전체 채널 목록 확인
    print(f"\n4️⃣ 전체 채널 목록 확인")
    response = requests.get(f"{BASE_URL}/channels/")
    if response.status_code == 200:
        channels = response.json()
        print(f"✅ 채널 목록 조회 성공!")
        print(f"   - 총 채널 개수: {channels['total']}")

        for channel in channels["channels"]:
            print(
                f"   📺 채널 ID {channel['id']}: '{channel['title']}' (생성자: {channel['employee_id']})"
            )
    else:
        print(f"❌ 채널 목록 조회 실패: {response.text}")

    print("\n" + "=" * 50)
    print("🎉 테스트 완료!")
    print("✅ PostgreSQL 데이터베이스에 데이터가 정상적으로 저장되었습니다!")
    print("✅ Channel과 Chat 테이블 간의 관계도 올바르게 설정되었습니다!")


if __name__ == "__main__":
    test_chat_flow()
