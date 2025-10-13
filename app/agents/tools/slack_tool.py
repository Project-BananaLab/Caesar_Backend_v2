# tools/slack_tool.py
from langchain.tools import Tool
import requests
import json
from app.utils.db import get_service_token


def create_slack_tools(user_id: str):
    """Slack Tool 생성"""

    def get_slack_headers():
        """Slack API 헤더 생성"""
        from app.utils.db import get_service_token_enhanced

        token_info = get_service_token_enhanced(user_id, "slack")
        if not token_info:
            raise Exception(
                "Slack 토큰이 없습니다. 직원 DB에 SLACK_API를 설정하거나 .env 파일에 SLACK_USER_TOKEN을 설정해주세요."
            )

        user_token = token_info.get("user_token")
        if not user_token:
            raise Exception(
                "Slack User 토큰이 없습니다. 직원 DB에 SLACK_API를 설정하거나 .env 파일에 SLACK_USER_TOKEN을 설정해주세요."
            )

        return {
            "Authorization": f"Bearer {user_token}",
            "Content-Type": "application/json",
        }

    def send_message(query: str) -> str:
        """Slack 메시지 전송
        Args:
            query (str): JSON 형태 {"channel": "채널명", "text": "메시지내용"}
        """
        try:
            headers = get_slack_headers()

            # JSON 파싱 시도
            try:
                message_data = json.loads(query)
                channel = message_data.get("channel")
                # text 또는 message 필드 모두 지원
                text = message_data.get("text") or message_data.get("message")
            except json.JSONDecodeError:
                # JSON이 아닌 경우, 단순 문자열로 처리
                print(f"⚠️ JSON이 아닌 단순 문자열로 전달됨: {query}")
                # 기본 채널명 추출 시도 (한국어 채널명 패턴)
                if "채널" in query and "에" in query:
                    # "cnp유튜브 채널에 메시지" 형태에서 채널명 추출
                    parts = query.split("채널에")
                    if len(parts) >= 2:
                        channel = parts[0].strip()
                        text = parts[1].strip()
                    else:
                        return f'채널명을 찾을 수 없습니다. JSON 형식으로 다시 시도해주세요: {{"channel": "채널명", "text": "메시지내용"}}'
                else:
                    return f'잘못된 형식입니다. JSON 형식으로 다시 시도해주세요: {{"channel": "채널명", "text": "메시지내용"}}'

            payload = {"channel": channel, "text": text}

            response = requests.post(
                "https://slack.com/api/chat.postMessage", headers=headers, json=payload
            )

            result = response.json()

            if result.get("ok"):
                return f"메시지가 #{channel}에 성공적으로 전송되었습니다."
            else:
                return f"메시지 전송 실패: {result.get('error', '알 수 없는 오류')}"

        except Exception as e:
            return f"Slack 메시지 전송 중 오류: {str(e)}"

    def list_channels(query: str) -> str:
        """Slack 채널 목록 조회
        Args:
            query (str): "all" 또는 빈 문자열
        """
        try:
            headers = get_slack_headers()

            response = requests.get(
                "https://slack.com/api/conversations.list",
                headers=headers,
                params={"types": "public_channel,private_channel"},
            )

            result = response.json()

            if not result.get("ok"):
                return f"채널 조회 실패: {result.get('error', '알 수 없는 오류')}"

            channels = result.get("channels", [])

            if not channels:
                return "채널을 찾을 수 없습니다."

            channel_list = []
            for channel in channels[:10]:  # 최대 10개만 표시
                name = channel.get("name")
                is_private = channel.get("is_private", False)
                member_count = channel.get("num_members", 0)
                privacy = "비공개" if is_private else "공개"
                channel_list.append(f"- #{name} ({privacy}, 멤버: {member_count}명)")

            return "Slack 채널 목록:\n" + "\n".join(channel_list)

        except Exception as e:
            return f"Slack 채널 조회 중 오류: {str(e)}"

    def get_recent_messages(query: str) -> str:
        """최근 메시지 조회
        Args:
            query (str): 채널명
        """
        try:
            headers = get_slack_headers()

            # 채널 ID 조회
            response = requests.get(
                "https://slack.com/api/conversations.list", headers=headers
            )

            result = response.json()
            if not result.get("ok"):
                return f"채널 조회 실패: {result.get('error')}"

            channel_id = None
            for channel in result.get("channels", []):
                if channel.get("name") == query:
                    channel_id = channel.get("id")
                    break

            if not channel_id:
                return f"채널 '{query}'를 찾을 수 없습니다."

            # 메시지 조회
            response = requests.get(
                "https://slack.com/api/conversations.history",
                headers=headers,
                params={"channel": channel_id, "limit": 5},
            )

            result = response.json()
            if not result.get("ok"):
                return f"메시지 조회 실패: {result.get('error')}"

            messages = result.get("messages", [])

            if not messages:
                return f"#{query} 채널에 메시지가 없습니다."

            message_list = []
            for msg in reversed(messages):  # 시간순으로 정렬
                text = msg.get("text", "")
                user = msg.get("user", "Unknown")
                timestamp = msg.get("ts", "")
                message_list.append(f"- {user}: {text}")

            return f"#{query} 최근 메시지:\n" + "\n".join(message_list)

        except Exception as e:
            return f"Slack 메시지 조회 중 오류: {str(e)}"

    def update_message(query: str) -> str:
        """Slack 메시지 수정
        Args:
            query (str): JSON 형태 {"channel": "채널명", "timestamp": "메시지타임스탬프", "text": "새메시지내용"}
        """
        try:
            headers = get_slack_headers()

            # JSON 파싱
            message_data = json.loads(query)
            channel = message_data.get("channel")
            timestamp = message_data.get("timestamp")
            new_text = message_data.get("text")

            if not channel or not timestamp or not new_text:
                return "채널명, 타임스탬프, 새 메시지 내용이 모두 필요합니다."

            # 채널 ID 조회
            channels_response = requests.get(
                "https://slack.com/api/conversations.list",
                headers=headers,
            )

            if channels_response.status_code != 200:
                return "채널 목록을 가져올 수 없습니다."

            channels = channels_response.json().get("channels", [])
            channel_id = None

            for ch in channels:
                if ch["name"] == channel.replace("#", ""):
                    channel_id = ch["id"]
                    break

            if not channel_id:
                return f"채널 #{channel}을 찾을 수 없습니다."

            # 메시지 수정
            data = {"channel": channel_id, "ts": timestamp, "text": new_text}

            response = requests.post(
                "https://slack.com/api/chat.update",
                headers=headers,
                json=data,
            )

            result = response.json()

            if response.status_code == 200 and result.get("ok"):
                return f"#{channel}에서 메시지가 수정되었습니다: {new_text[:50]}..."
            else:
                return f"메시지 수정 실패: {result.get('error', '알 수 없는 오류')}"

        except Exception as e:
            return f"Slack 메시지 수정 중 오류: {str(e)}"

    def delete_message(query: str) -> str:
        """Slack 메시지 삭제
        Args:
            query (str): JSON 형태 {"channel": "채널명", "timestamp": "메시지타임스탬프"}
        """
        try:
            headers = get_slack_headers()

            # JSON 파싱
            message_data = json.loads(query)
            channel = message_data.get("channel")
            timestamp = message_data.get("timestamp")

            if not channel or not timestamp:
                return "채널명과 타임스탬프가 필요합니다."

            # 채널 ID 조회
            channels_response = requests.get(
                "https://slack.com/api/conversations.list",
                headers=headers,
            )

            if channels_response.status_code != 200:
                return "채널 목록을 가져올 수 없습니다."

            channels = channels_response.json().get("channels", [])
            channel_id = None

            for ch in channels:
                if ch["name"] == channel.replace("#", ""):
                    channel_id = ch["id"]
                    break

            if not channel_id:
                return f"채널 #{channel}을 찾을 수 없습니다."

            # 메시지 삭제
            data = {"channel": channel_id, "ts": timestamp}

            response = requests.post(
                "https://slack.com/api/chat.delete",
                headers=headers,
                json=data,
            )

            result = response.json()

            if response.status_code == 200 and result.get("ok"):
                return (
                    f"#{channel}에서 메시지가 삭제되었습니다 (타임스탬프: {timestamp})"
                )
            else:
                return f"메시지 삭제 실패: {result.get('error', '알 수 없는 오류')}"

        except Exception as e:
            return f"Slack 메시지 삭제 중 오류: {str(e)}"

    return [
        Tool(
            name="send_slack_message",
            description="Slack 채널에 메시지를 전송합니다. JSON 형태로 채널명과 메시지를 입력하세요.",
            func=send_message,
        ),
        Tool(
            name="list_slack_channels",
            description="Slack 워크스페이스의 채널 목록을 조회합니다.",
            func=list_channels,
        ),
        Tool(
            name="get_slack_messages",
            description="특정 Slack 채널의 최근 메시지를 조회합니다. 채널명을 입력하세요.",
            func=get_recent_messages,
        ),
        Tool(
            name="update_slack_message",
            description="Slack 메시지를 수정합니다. JSON 형태로 채널명, 타임스탬프, 새 메시지를 입력하세요.",
            func=update_message,
        ),
        Tool(
            name="delete_slack_message",
            description="Slack 메시지를 삭제합니다. JSON 형태로 채널명과 타임스탬프를 입력하세요.",
            func=delete_message,
        ),
    ]
