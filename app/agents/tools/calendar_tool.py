# tools/calendar_tool.py - 완전한 CRUD 기능
from langchain.tools import Tool
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import json
import re


def parse_natural_event_input(query: str):
    """자연어 입력을 파싱하여 이벤트 정보 추출"""
    # 기본값
    title = "새 일정"
    start_input = ""
    end_input = ""
    description = ""

    # 다양한 패턴 시도 (더 유연하게)
    patterns = [
        # "포케, 오후 5시~6시"
        r"(.+?)[,\s]+(오전|오후|AM|PM)\s*(\d{1,2})[시:]?[\s~-]+(\d{1,2})[시:]?",
        # "포케, 5시~6시"
        r"(.+?)[,\s]+(\d{1,2})[시:]?[\s~-]+(\d{1,2})[시:]?",
        # "포케, 17:00~18:00"
        r"(.+?)[,\s]+(\d{1,2}:\d{2})[\s~-]+(\d{1,2}:\d{2})",
        # "오늘 7시 멘토링" (시간이 중간에 있는 경우)
        r"(.+?)\s+(\d{1,2})[시:]?\s+(.+)",
        # "7시 멘토링" (시간이 앞에 있는 경우)
        r"(\d{1,2})[시:]?\s+(.+)",
        # "오후 7시 멘토링"
        r"(오전|오후|AM|PM)\s*(\d{1,2})[시:]?\s+(.+)",
    ]

    for i, pattern in enumerate(patterns):
        match = re.search(pattern, query)
        if match:
            if i == 0:  # "포케, 오후 5시~6시"
                title = match.group(1).strip()
                period = match.group(2)
                start_hour = int(match.group(3))
                end_hour = int(match.group(4))

                # 12시간 형식을 24시간으로 변환
                if period in ["오후", "PM"] and start_hour != 12:
                    start_hour += 12
                    end_hour += 12
                elif period in ["오전", "AM"] and start_hour == 12:
                    start_hour = 0

                start_input = f"{start_hour:02d}:00"
                end_input = f"{end_hour:02d}:00"

            elif i == 1:  # "포케, 5시~6시"
                title = match.group(1).strip()
                start_hour = int(match.group(2))
                end_hour = int(match.group(3))
                start_input = f"{start_hour:02d}:00"
                end_input = f"{end_hour:02d}:00"

            elif i == 2:  # "포케, 17:00~18:00"
                title = match.group(1).strip()
                start_input = match.group(2)
                end_input = match.group(3)

            elif i == 3:  # "오늘 7시 멘토링"
                prefix = match.group(1).strip()
                hour = int(match.group(2))
                suffix = match.group(3).strip()
                title = f"{suffix}"  # 멘토링
                start_input = f"{hour:02d}:00"
                end_input = f"{(hour + 1):02d}:00"  # 1시간 후

            elif i == 4:  # "7시 멘토링"
                hour = int(match.group(1))
                title = match.group(2).strip()
                start_input = f"{hour:02d}:00"
                end_input = f"{(hour + 1):02d}:00"  # 1시간 후

            elif i == 5:  # "오후 7시 멘토링"
                period = match.group(1)
                hour = int(match.group(2))
                title = match.group(3).strip()

                # 12시간 형식을 24시간으로 변환
                if period in ["오후", "PM"] and hour != 12:
                    hour += 12
                elif period in ["오전", "AM"] and hour == 12:
                    hour = 0

                start_input = f"{hour:02d}:00"
                end_input = f"{(hour + 1):02d}:00"  # 1시간 후

            break

    # 패턴 매칭이 안된 경우 전체를 제목으로
    if not start_input:
        title = query.strip()
        # 기본 시간 설정하지 않고 빈 값으로 두어 parse_event_times에서 처리

    return title, start_input, end_input, description


def parse_event_times(start_input: str, end_input: str):
    """시간 입력을 Google Calendar API 형식으로 변환"""
    try:
        now = datetime.now()
        today = now.date()

        # 기본 시간 설정 (입력이 없는 경우)
        if not start_input and not end_input:
            # 현재 시간 기준으로 1시간 후 일정 생성
            start_dt = now + timedelta(hours=1)
            end_dt = start_dt + timedelta(hours=1)
        else:
            # 시간 파싱
            if ":" in start_input:  # HH:MM 형식
                start_hour, start_min = map(int, start_input.split(":"))
            else:  # 단순 숫자
                start_hour = int(start_input) if start_input.isdigit() else 17
                start_min = 0

            if ":" in end_input:  # HH:MM 형식
                end_hour, end_min = map(int, end_input.split(":"))
            else:  # 단순 숫자
                end_hour = int(end_input) if end_input.isdigit() else start_hour + 1
                end_min = 0

            # 오늘 날짜로 datetime 생성
            start_dt = datetime.combine(
                today, datetime.min.time().replace(hour=start_hour, minute=start_min)
            )
            end_dt = datetime.combine(
                today, datetime.min.time().replace(hour=end_hour, minute=end_min)
            )

        # Context7 권장 형식: RFC3339 with timezone
        start_time = {
            "dateTime": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "timeZone": "Asia/Seoul",
        }

        end_time = {
            "dateTime": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "timeZone": "Asia/Seoul",
        }

        return start_time, end_time

    except Exception as e:
        print(f"시간 파싱 오류: {e}")
        return None, None


def format_time_for_display(time_obj):
    """시간 객체를 사용자 친화적 형식으로 변환"""
    try:
        if "dateTime" in time_obj:
            dt = datetime.fromisoformat(time_obj["dateTime"])
            return dt.strftime("%Y-%m-%d %H:%M")
        elif "date" in time_obj:
            return time_obj["date"]
        return str(time_obj)
    except:
        return str(time_obj)


def parse_natural_update_input(query: str):
    """자연어 수정 입력을 JSON 형태로 변환"""
    # "멘토링 일정을 서일근 멘토링으로 수정해줘" 패턴 처리
    patterns = [
        # "A를 B로 수정" / "A를 B로 변경"
        r"(.+?)[을를]\s*(.+?)[으로로]\s*(수정|변경)",
        # "A 일정을 B로 수정"
        r"(.+?)\s*일정[을를]\s*(.+?)[으로로]\s*(수정|변경)",
        # "A에서 B로 수정"
        r"(.+?)[에서]\s*(.+?)[으로로]\s*(수정|변경)",
    ]

    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            old_title = match.group(1).strip()
            new_title = match.group(2).strip()

            # 불필요한 단어 제거
            old_title = old_title.replace("일정", "").strip()
            new_title = new_title.replace("일정", "").strip()

            return {
                "title": old_title,  # 찾을 제목
                "summary": new_title,  # 새로운 제목
                "event_id": None,
            }

    # 패턴 매칭 실패 시 기본 처리
    return {"title": query.strip(), "summary": "", "event_id": None}


def create_calendar_tools(user_id: str):
    """Google Calendar Tool 생성"""

    def get_calendar_service():
        """Google Calendar API 서비스 생성"""
        from app.utils.google_auth import get_google_service_credentials

        try:
            creds = get_google_service_credentials("google", user_id)
            return build("calendar", "v3", credentials=creds)
        except Exception as e:
            raise Exception(f"Google Calendar 서비스 초기화 실패: {str(e)}")

    def list_events(query: str) -> str:
        """캘린더 이벤트 목록 조회 - 향상된 진단 정보"""
        try:
            service = get_calendar_service()
            now = datetime.utcnow()

            if query.lower() in ["today", "오늘"]:
                time_min = now.replace(hour=0, minute=0, second=0, microsecond=0)
                time_max = time_min + timedelta(days=1)
                period_name = "오늘"
            elif query.lower() in ["this week", "이번주"]:
                time_min = now - timedelta(days=now.weekday())
                time_max = time_min + timedelta(days=7)
                period_name = "이번 주"
            elif query.lower() in ["recent", "최근", "all", "전체"]:
                time_min = now - timedelta(days=14)  # 2주 전부터
                time_max = now + timedelta(days=14)  # 2주 후까지
                period_name = "최근 4주"
            else:
                time_min = now - timedelta(days=3)  # 3일 전부터
                time_max = now + timedelta(days=7)  # 1주일 후까지
                period_name = "최근 3일~앞으로 1주일"

            print(
                f"🔍 캘린더 조회 중: {period_name} ({time_min.date()} ~ {time_max.date()})"
            )

            events_result = (
                service.events()
                .list(
                    calendarId="primary",
                    timeMin=time_min.isoformat() + "Z",
                    timeMax=time_max.isoformat() + "Z",
                    maxResults=20,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

            events = events_result.get("items", [])

            if not events:
                return f"📅 {period_name}에 등록된 일정이 없습니다.\n🔍 검색 범위: {time_min.date()} ~ {time_max.date()}\n💡 다른 기간을 확인하려면 'recent' 또는 'all'로 검색해보세요."

            event_list = [f"📅 **{period_name} 일정 목록** ({len(events)}개)"]
            event_list.append("-" * 50)

            for i, event in enumerate(events, 1):
                start = event["start"].get("dateTime", event["start"].get("date"))
                title = event.get("summary", "제목 없음")
                event_id = event.get("id")

                # 시간 포맷팅 개선
                if "T" in start:
                    # datetime 형식
                    dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    time_str = dt.strftime("%m/%d %H:%M")
                else:
                    # date 형식 (종일 일정)
                    time_str = f"{start} (종일)"

                event_list.append(f"{i:2d}. {title}")
                event_list.append(f"    ⏰ {time_str}")
                event_list.append(f"    🆔 {event_id[:12]}...")

                if i < len(events):  # 마지막이 아니면 구분선
                    event_list.append("")

            return "\n".join(event_list)

        except Exception as e:
            return f"❌ 일정 조회 중 오류: {str(e)}\n💡 Google Calendar 인증을 확인해주세요."

    def create_event(query: str) -> str:
        """캘린더 이벤트 생성 - 자연어 처리 지원"""
        try:
            service = get_calendar_service()

            # JSON 파싱 시도, 실패하면 자연어 처리
            try:
                event_data = json.loads(query)
                title = event_data.get("summary", event_data.get("title", "새 일정"))
                start_input = event_data.get("start", "")
                end_input = event_data.get("end", "")
                description = event_data.get("description", "")
            except (json.JSONDecodeError, TypeError):
                # 자연어 입력 처리
                title, start_input, end_input, description = parse_natural_event_input(
                    query
                )

            # 시간 파싱 및 변환
            start_time, end_time = parse_event_times(start_input, end_input)

            if not start_time or not end_time:
                return "시작 시간과 종료 시간을 올바르게 지정해주세요. 예: '오후 5시~6시' 또는 '17:00-18:00'"

            # Context7 권장 형식에 따른 이벤트 생성
            event = {
                "summary": title,
                "description": description,
                "start": start_time,
                "end": end_time,
                "timeZone": "Asia/Seoul",
            }

            created_event = (
                service.events().insert(calendarId="primary", body=event).execute()
            )

            # 성공 메시지에 시간 정보 포함
            start_str = format_time_for_display(start_time)
            end_str = format_time_for_display(end_time)

            return f"✅ 일정이 생성되었습니다!\n📅 제목: {title}\n⏰ 시간: {start_str} ~ {end_str}\n🔗 이벤트 ID: {created_event.get('id', '')[:8]}..."

        except Exception as e:
            return f'❌ 이벤트 생성 중 오류: {str(e)}\n💡 다음 형식으로 시도해보세요: "포케, 오후 5시~6시" 또는 JSON 형태'

    def find_event_by_title(service, title: str, days_range: int = 14):
        """제목으로 이벤트 검색 - 디버깅 정보 포함"""
        try:
            now = datetime.utcnow()
            time_min = now - timedelta(days=days_range)
            time_max = now + timedelta(days=days_range)

            print(
                f"🔍 이벤트 검색 중: '{title}' (범위: {time_min.date()} ~ {time_max.date()})"
            )

            events_result = (
                service.events()
                .list(
                    calendarId="primary",
                    timeMin=time_min.isoformat() + "Z",
                    timeMax=time_max.isoformat() + "Z",
                    maxResults=100,  # 더 많은 결과 검색
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

            events = events_result.get("items", [])
            print(f"📋 총 {len(events)}개 일정 발견")

            # 모든 일정 제목 출력 (디버깅용)
            all_titles = [event.get("summary", "제목없음") for event in events]
            print(f"📝 발견된 일정들: {all_titles}")

            # 제목으로 검색 (부분 일치 및 정확 일치 모두 시도)
            matching_events = []
            search_title = title.lower().strip()

            # 빈 제목이면 모든 이벤트 반환 (디버깅용)
            if not search_title:
                print(f"🔍 모든 이벤트 반환 (검색어 없음)")
                return events

            for event in events:
                event_title = event.get("summary", "").lower().strip()

                # 정확 일치 우선
                if search_title == event_title:
                    matching_events.insert(0, event)  # 맨 앞에 추가
                    print(f"✅ 정확 일치 발견: '{event.get('summary', '')}'")
                # 부분 일치 (양방향)
                elif search_title and (
                    search_title in event_title or event_title in search_title
                ):
                    matching_events.append(event)
                    print(f"🔍 부분 일치 발견: '{event.get('summary', '')}'")

            print(f"🎯 최종 매칭: {len(matching_events)}개")
            return matching_events

        except Exception as e:
            print(f"❌ 검색 오류: {e}")
            return []

    def update_event(query: str) -> str:
        """캘린더 이벤트 수정 - 제목으로 검색 지원"""
        try:
            service = get_calendar_service()

            # JSON 파싱 시도
            try:
                event_data = json.loads(query)
                event_id = event_data.get("event_id")
                title_to_find = event_data.get("title") or event_data.get("summary")
            except (json.JSONDecodeError, TypeError):
                # 자연어 입력 처리 - "멘토링 일정을 서일근 멘토링으로 수정"
                event_data = parse_natural_update_input(query)
                event_id = event_data.get("event_id")
                title_to_find = event_data.get("title") or event_data.get("summary")

            # 이벤트 ID가 없으면 제목으로 검색
            if not event_id and title_to_find:
                print(f"🔎 '{title_to_find}' 제목으로 일정 검색 시작...")
                matching_events = find_event_by_title(service, title_to_find)

                if not matching_events:
                    return f"❌ '{title_to_find}' 제목의 일정을 찾을 수 없습니다.\n\n💡 다음을 확인해보세요:\n1. 'list_calendar_events' 도구로 all 또는 recent 검색하여 실제 일정 제목 확인\n2. 정확한 제목으로 다시 시도\n3. 일정이 최근 2주 내에 있는지 확인"
                elif len(matching_events) > 1:
                    event_list = []
                    for i, event in enumerate(matching_events[:5], 1):
                        start = event["start"].get(
                            "dateTime", event["start"].get("date")
                        )
                        event_list.append(
                            f"{i}. {event.get('summary', '제목없음')} ({start[:16]}) [ID: {event.get('id')[:8]}...]"
                        )
                    return (
                        f"'{title_to_find}' 제목의 일정이 여러 개 있습니다:\n"
                        + "\n".join(event_list)
                        + "\n\n정확한 이벤트 ID를 사용해주세요."
                    )
                else:
                    # 하나만 찾은 경우 해당 이벤트 사용
                    event_id = matching_events[0]["id"]

            if not event_id:
                return "이벤트 ID 또는 검색할 제목이 필요합니다."

            # 이벤트 가져오기
            try:
                existing_event = (
                    service.events()
                    .get(calendarId="primary", eventId=event_id)
                    .execute()
                )
            except Exception:
                return f"이벤트 ID '{event_id}'를 찾을 수 없습니다."

            # 수정사항 적용
            if "summary" in event_data:
                existing_event["summary"] = event_data["summary"]
            if "description" in event_data:
                existing_event["description"] = event_data["description"]
            if "start" in event_data:
                start_time, _ = parse_event_times(
                    event_data["start"], existing_event["end"].get("dateTime", "")
                )
                if start_time:
                    existing_event["start"] = start_time
            if "end" in event_data:
                _, end_time = parse_event_times(
                    existing_event["start"].get("dateTime", ""), event_data["end"]
                )
                if end_time:
                    existing_event["end"] = end_time

            # 이벤트 업데이트
            updated_event = (
                service.events()
                .update(calendarId="primary", eventId=event_id, body=existing_event)
                .execute()
            )

            return f"✅ 이벤트가 수정되었습니다!\n📅 제목: {updated_event.get('summary', '제목 없음')}\n🔗 ID: {event_id[:8]}..."

        except Exception as e:
            return f"❌ 이벤트 수정 중 오류: {str(e)}"

    def delete_event(query: str) -> str:
        """캘린더 이벤트 삭제 - 제목으로도 검색 가능"""
        try:
            service = get_calendar_service()
            event_id = None
            event_title = None

            # 먼저 이벤트 ID로 시도
            print(f"🔍 '{query}' 이벤트 ID로 검색 시도...")
            try:
                event = (
                    service.events().get(calendarId="primary", eventId=query).execute()
                )
                event_id = query
                event_title = event.get("summary", "제목 없음")
                print(f"✅ 이벤트 ID로 찾음: {event_title}")
            except Exception as e:
                print(f"❌ 이벤트 ID 검색 실패: {str(e)}")
                # 이벤트 ID로 못 찾으면 제목으로 검색
                print(f"🔎 '{query}' 제목으로 이벤트 검색 중...")
                matching_events = find_event_by_title(service, query)

                if not matching_events:
                    return f"❌ '{query}' 제목의 일정을 찾을 수 없습니다.\n💡 다음을 확인해보세요:\n1. 'list_calendar_events' 도구로 실제 일정 제목 확인\n2. 정확한 제목으로 다시 시도\n3. 또는 정확한 이벤트 ID 사용"
                elif len(matching_events) > 1:
                    event_list = []
                    for i, event in enumerate(matching_events[:5], 1):
                        start = event["start"].get(
                            "dateTime", event["start"].get("date")
                        )
                        event_list.append(
                            f"{i}. {event.get('summary', '제목없음')} ({start[:16]}) [ID: {event.get('id')[:8]}...]"
                        )
                    return (
                        f"❌ '{query}' 제목의 일정이 여러 개 있습니다:\n"
                        + "\n".join(event_list)
                        + f"\n\n💡 정확한 이벤트 ID로 다시 삭제해주세요:\n예: delete_calendar_event('{matching_events[0].get('id')}')"
                    )
                else:
                    # 하나만 찾은 경우
                    event_id = matching_events[0]["id"]
                    event_title = matching_events[0].get("summary", "제목 없음")
                    print(f"✅ 제목으로 찾음: {event_title} (ID: {event_id[:8]}...)")

            if not event_id:
                return f"❌ '{query}' 이벤트를 찾을 수 없습니다."

            # 이벤트 삭제 실행
            service.events().delete(calendarId="primary", eventId=event_id).execute()
            return f"✅ 이벤트가 삭제되었습니다!\n📅 제목: {event_title}\n🔗 ID: {event_id[:8]}..."

        except Exception as e:
            return f"❌ 이벤트 삭제 중 오류: {str(e)}"

    return [
        Tool(
            name="list_calendar_events",
            description="Google Calendar에서 이벤트 목록을 조회합니다. 'today', 'this week' 또는 특정 날짜를 입력하세요.",
            func=list_events,
        ),
        Tool(
            name="create_calendar_event",
            description="Google Calendar에 새 이벤트를 생성합니다. JSON 형태로 이벤트 정보를 입력하세요.",
            func=create_event,
        ),
        Tool(
            name="update_calendar_event",
            description="Google Calendar 이벤트를 수정합니다. JSON 형태로 이벤트 ID와 수정할 정보를 입력하세요.",
            func=update_event,
        ),
        Tool(
            name="delete_calendar_event",
            description="Google Calendar 이벤트를 삭제합니다. 이벤트 ID 또는 제목을 입력하세요. 제목으로 검색하면 자동으로 해당 이벤트를 찾아 삭제합니다.",
            func=delete_event,
        ),
    ]
