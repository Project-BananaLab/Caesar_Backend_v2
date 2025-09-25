# tools/notion_tool.py
from langchain.tools import Tool
import requests
import json
from app.utils.db import get_service_token


def create_notion_tools(user_id: str):
    """Notion Tool 생성"""

    def get_notion_headers():
        """Notion API 헤더 생성"""
        from app.utils.db import get_service_token_enhanced
        
        token_info = get_service_token_enhanced(user_id, "notion")
        print(f"🔍 노션 토큰 정보: {token_info}")

        if not token_info:
            raise Exception(
                "Notion 토큰이 없습니다. 직원 DB에 NOTION_API를 설정하거나 .env 파일에 NOTION_TOKEN을 설정해주세요."
            )

        token = token_info.get("token")
        if not token:
            raise Exception(
                "Notion 토큰이 없습니다. 직원 DB에 NOTION_API를 설정하거나 .env 파일에 NOTION_TOKEN을 설정해주세요."
            )

        print(f"🔑 노션 토큰 (앞 10자): {token[:10]}...")

        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

    def list_all_content(query: str = "") -> str:
        """모든 Notion 페이지/데이터베이스 목록 조회
        Args:
            query (str): 선택적 검색 키워드 (비워두거나 '모든', '전체', '목록' 등을 입력하면 모든 항목 조회)
        """
        try:
            headers = get_notion_headers()

            # 검색 파라미터 설정
            payload = {
                "sort": {"direction": "descending", "timestamp": "last_edited_time"},
                "page_size": 20,
            }

            # 특정 키워드들은 전체 조회로 처리
            show_all_keywords = [
                "모든",
                "전체",
                "all",
                "목록",
                "리스트",
                "페이지",
                "전부",
            ]
            should_show_all = (
                not query
                or query.strip() == ""
                or any(keyword in query.lower() for keyword in show_all_keywords)
            )

            # 검색어가 있고 전체 조회가 아닌 경우에만 쿼리 추가
            if query and query.strip() and not should_show_all:
                payload["query"] = query.strip()

            print(f"🌐 노션 API 호출: {payload}")
            response = requests.post(
                "https://api.notion.com/v1/search", headers=headers, json=payload
            )

            print(f"📊 응답 상태: {response.status_code}")
            result = response.json()
            print(f"📄 응답 내용: {result}")

            if response.status_code != 200:
                error_msg = result.get("message", "알 수 없는 오류")
                error_code = result.get("code", "unknown")
                return f"❌ 노션 API 오류: {error_msg} (코드: {error_code})\n💡 확인사항:\n1. NOTION_TOKEN이 올바른지 확인\n2. Integration이 워크스페이스에 추가되었는지 확인\n3. 페이지/데이터베이스가 Integration과 공유되었는지 확인"

            items = result.get("results", [])

            if not items:
                return "📭 접근 가능한 Notion 페이지나 데이터베이스가 없습니다.\n\n💡 해결 방법:\n1. Notion Integration을 워크스페이스에 추가\n2. 공유하고 싶은 페이지에서 '공유' → Integration 선택\n3. 데이터베이스도 Integration과 공유 필요"

            pages = []
            databases = []

            for item in items:
                title = "제목 없음"
                item_id = item.get("id")

                if item.get("object") == "page":
                    # 페이지 제목 추출
                    properties = item.get("properties", {})
                    for prop_name, prop_value in properties.items():
                        if prop_value.get("type") == "title":
                            title_array = prop_value.get("title", [])
                            if title_array:
                                title = (
                                    title_array[0]
                                    .get("text", {})
                                    .get("content", "제목 없음")
                                )
                            break

                    last_edited = item.get("last_edited_time", "")[:10]  # 날짜만
                    pages.append(f"📄 {title}")
                    pages.append(f"   🆔 ID: {item_id} | 📅 수정: {last_edited}")

                elif item.get("object") == "database":
                    # 데이터베이스 제목 추출
                    title_array = item.get("title", [])
                    if title_array:
                        title = (
                            title_array[0].get("text", {}).get("content", "제목 없음")
                        )

                    last_edited = item.get("last_edited_time", "")[:10]
                    databases.append(f"🗃️ {title}")
                    databases.append(f"   🆔 ID: {item_id} | 📅 수정: {last_edited}")

            result_parts = []

            if should_show_all:
                result_parts.append("📋 Notion 워크스페이스 전체 목록:")
            else:
                result_parts.append(f"🔍 '{query}' 검색 결과:")

            if databases:
                result_parts.append("\n📊 데이터베이스:")
                result_parts.extend(databases)

            if pages:
                result_parts.append("\n📄 페이지:")
                result_parts.extend(pages)

            total_count = len(databases) // 2 + len(pages) // 2
            result_parts.append(f"\n📈 총 {total_count}개 항목")
            result_parts.append(
                "\n💡 특정 항목을 보려면 'ID로 페이지 내용 보여줘' 또는 'ID로 데이터베이스 보여줘'라고 말해주세요."
            )

            return "\n".join(result_parts)

        except Exception as e:
            return f"Notion 목록 조회 중 오류: {str(e)}"

    def create_page(query: str) -> str:
        """Notion 페이지 생성
        Args:
            query (str): JSON 형태 {"title": "페이지제목", "content": "내용"}
                        또는 {"parent_id": "부모페이지ID", "title": "페이지제목", "content": "내용"}
                        parent_id가 없으면 워크스페이스 루트에 생성
        """
        try:
            headers = get_notion_headers()

            # JSON 파싱 시도, 실패하면 간단 형식 처리
            try:
                page_data = json.loads(query)
                title = page_data.get("title", "새 페이지")
                content = page_data.get("content", "")
                parent_id = page_data.get("parent_id")
            except (json.JSONDecodeError, TypeError):
                # JSON이 아니면 제목으로 처리
                title = query.strip() if query.strip() else "새 페이지"
                content = ""
                parent_id = None

            # parent 설정: parent_id가 있으면 하위 페이지, 없으면 워크스페이스 루트
            if parent_id:
                parent_config = {"page_id": parent_id}
            else:
                parent_config = {"type": "workspace", "workspace": True}

            payload = {
                "parent": parent_config,
                "properties": {"title": {"title": [{"text": {"content": title}}]}},
            }

            # 내용이 있으면 children 추가
            if content:
                payload["children"] = [
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {"type": "text", "text": {"content": content}}
                            ]
                        },
                    }
                ]

            response = requests.post(
                "https://api.notion.com/v1/pages", headers=headers, json=payload
            )

            result = response.json()

            if response.status_code != 200:
                return f"페이지 생성 실패: {result.get('message', '알 수 없는 오류')}"

            page_id = result.get("id")
            return f"페이지가 생성되었습니다: {title} (ID: {page_id})"

        except Exception as e:
            return f"Notion 페이지 생성 중 오류: {str(e)}"

    def find_page_by_title(title: str):
        """제목으로 페이지 검색"""
        try:
            headers = get_notion_headers()

            payload = {
                "query": title.strip(),
                "filter": {"property": "object", "value": "page"},
                "sort": {"direction": "descending", "timestamp": "last_edited_time"},
                "page_size": 10,
            }

            response = requests.post(
                "https://api.notion.com/v1/search", headers=headers, json=payload
            )

            if response.status_code != 200:
                return []

            results = response.json().get("results", [])
            matching_pages = []

            for page in results:
                page_title = "제목 없음"
                properties = page.get("properties", {})
                for prop_name, prop_value in properties.items():
                    if prop_value.get("type") == "title":
                        title_array = prop_value.get("title", [])
                        if title_array:
                            page_title = (
                                title_array[0]
                                .get("text", {})
                                .get("content", "제목 없음")
                            )
                        break

                # 제목 매칭 (부분 일치)
                if (
                    title.lower() in page_title.lower()
                    or page_title.lower() in title.lower()
                ):
                    matching_pages.append(
                        {
                            "id": page.get("id"),
                            "title": page_title,
                            "last_edited": page.get("last_edited_time", ""),
                        }
                    )

            return matching_pages

        except Exception as e:
            print(f"❌ 제목 검색 오류: {e}")
            return []

    def get_page_content(query: str) -> str:
        """Notion 페이지 내용 조회 - ID 또는 제목으로 검색 가능
        Args:
            query (str): 페이지 ID 또는 페이지 제목
        """
        try:
            # JSON이 전달된 경우 제목만 추출
            if query.strip().startswith("{") and "page_title" in query:
                try:
                    import json

                    data = json.loads(query)
                    query = data.get("page_title", query)
                    print(f"📝 JSON에서 제목 추출: {query}")
                except:
                    print(f"⚠️ JSON 파싱 실패, 원본 사용: {query}")

            headers = get_notion_headers()
            page_id = None

            # 먼저 페이지 ID로 시도
            print(f"🔍 '{query}' 페이지 ID로 검색 시도...")
            try:
                response = requests.get(
                    f"https://api.notion.com/v1/pages/{query}", headers=headers
                )
                if response.status_code == 200:
                    page_id = query
                    print(f"✅ 페이지 ID로 찾음")
                else:
                    raise Exception("페이지 ID로 찾을 수 없음")
            except Exception:
                # 페이지 ID로 못 찾으면 제목으로 검색
                print(f"🔎 '{query}' 제목으로 페이지 검색 중...")
                matching_pages = find_page_by_title(query)

                if not matching_pages:
                    return f"❌ '{query}' 제목의 페이지를 찾을 수 없습니다.\n💡"
                elif len(matching_pages) > 1:
                    page_list = []
                    for i, page in enumerate(matching_pages[:5], 1):
                        page_list.append(
                            f"{i}. {page['title']} [ID: {page['id'][:8]}...]"
                        )
                    return (
                        f"❌ '{query}' 제목의 페이지가 여러 개 있습니다:\n"
                        + "\n".join(page_list)
                        + f"\n\n💡 정확한 페이지 ID를 사용해주세요."
                    )
                else:
                    # 하나만 찾은 경우
                    page_id = matching_pages[0]["id"]
                    print(
                        f"✅ 제목으로 찾음: {matching_pages[0]['title']} (ID: {page_id[:8]}...)"
                    )

            if not page_id:
                return f"❌ '{query}' 페이지를 찾을 수 없습니다."

            # 페이지 정보 조회
            response = requests.get(
                f"https://api.notion.com/v1/pages/{page_id}", headers=headers
            )

            if response.status_code != 200:
                return f"페이지 조회 실패: {response.json().get('message', '알 수 없는 오류')}"

            page_info = response.json()

            # 페이지 제목 추출
            title = "제목 없음"
            properties = page_info.get("properties", {})
            for prop_name, prop_value in properties.items():
                if prop_value.get("type") == "title":
                    title_array = prop_value.get("title", [])
                    if title_array:
                        title = (
                            title_array[0].get("text", {}).get("content", "제목 없음")
                        )
                    break

            # 페이지 블록 내용 조회 (실제 페이지 ID 사용)
            actual_page_id = page_id if page_id else query
            response = requests.get(
                f"https://api.notion.com/v1/blocks/{actual_page_id}/children",
                headers=headers,
            )

            if response.status_code != 200:
                return f"페이지 내용 조회 실패: {response.json().get('message', '알 수 없는 오류')}"

            blocks = response.json().get("results", [])

            content_list = [f"페이지 제목: {title}\n내용:"]

            for block in blocks:
                block_type = block.get("type")
                if block_type == "paragraph":
                    rich_text = block.get("paragraph", {}).get("rich_text", [])
                    text = "".join(
                        [rt.get("text", {}).get("content", "") for rt in rich_text]
                    )
                    if text:
                        content_list.append(f"- {text}")

            return "\n".join(content_list)

        except Exception as e:
            return f"Notion 페이지 조회 중 오류: {str(e)}"

    def update_page(query: str) -> str:
        """Notion 페이지 내용 수정 - ID 또는 제목으로 검색 가능
        Args:
            query (str): JSON 형태 {"page_id": "페이지ID", "title": "새제목(선택)", "content": "새내용"}
                        또는 {"page_title": "기존페이지제목", "title": "새제목", "content": "새내용"}
        """
        try:
            headers = get_notion_headers()

            # JSON 파싱
            page_data = json.loads(query)
            page_id = page_data.get("page_id")
            page_title_to_find = page_data.get("page_title")
            new_title = page_data.get("title")
            new_content = page_data.get("content")

            # 페이지 ID가 없으면 제목으로 검색
            if not page_id and page_title_to_find:
                print(f"🔎 '{page_title_to_find}' 제목으로 페이지 검색 중...")
                matching_pages = find_page_by_title(page_title_to_find)

                if not matching_pages:
                    return (
                        f"❌ '{page_title_to_find}' 제목의 페이지를 찾을 수 없습니다."
                    )
                elif len(matching_pages) > 1:
                    page_list = []
                    for i, page in enumerate(matching_pages[:5], 1):
                        page_list.append(
                            f"{i}. {page['title']} [ID: {page['id'][:8]}...]"
                        )
                    return (
                        f"❌ '{page_title_to_find}' 제목의 페이지가 여러 개 있습니다:\n"
                        + "\n".join(page_list)
                        + f"\n\n💡 정확한 페이지 ID를 사용해주세요."
                    )
                else:
                    page_id = matching_pages[0]["id"]
                    print(
                        f"✅ 제목으로 찾음: {matching_pages[0]['title']} (ID: {page_id[:8]}...)"
                    )

            if not page_id:
                return "페이지 ID 또는 page_title이 필요합니다."

            # 페이지 존재 확인
            response = requests.get(
                f"https://api.notion.com/v1/pages/{page_id}", headers=headers
            )

            if response.status_code != 200:
                return f"페이지 ID '{page_id}'를 찾을 수 없습니다."

            # 제목 수정 (페이지 속성 업데이트)
            if new_title:
                page_update_data = {
                    "properties": {
                        "title": {"title": [{"text": {"content": new_title}}]}
                    }
                }

                title_response = requests.patch(
                    f"https://api.notion.com/v1/pages/{page_id}",
                    headers=headers,
                    json=page_update_data,
                )

                if title_response.status_code != 200:
                    return f"페이지 제목 수정 실패: {title_response.json().get('message', '알 수 없는 오류')}"

            # 내용 수정 (기존 블록 삭제 후 새 블록 추가)
            if new_content:
                # 기존 블록들 조회
                blocks_response = requests.get(
                    f"https://api.notion.com/v1/blocks/{page_id}/children",
                    headers=headers,
                )

                if blocks_response.status_code == 200:
                    existing_blocks = blocks_response.json().get("results", [])

                    # 기존 블록들 삭제
                    for block in existing_blocks:
                        requests.delete(
                            f"https://api.notion.com/v1/blocks/{block['id']}",
                            headers=headers,
                        )

                # 새 내용 추가
                new_blocks = {
                    "children": [
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [
                                    {"type": "text", "text": {"content": new_content}}
                                ]
                            },
                        }
                    ]
                }

                content_response = requests.patch(
                    f"https://api.notion.com/v1/blocks/{page_id}/children",
                    headers=headers,
                    json=new_blocks,
                )

                if content_response.status_code != 200:
                    return f"페이지 내용 수정 실패: {content_response.json().get('message', '알 수 없는 오류')}"

            update_parts = []
            if new_title:
                update_parts.append(f"제목: '{new_title}'")
            if new_content:
                update_parts.append("내용")

            return f"페이지가 수정되었습니다: {', '.join(update_parts)} (ID: {page_id})"

        except Exception as e:
            return f"Notion 페이지 수정 중 오류: {str(e)}"

    def delete_page(query: str) -> str:
        """Notion 페이지 삭제 (아카이브) - ID 또는 제목으로 검색 가능
        Args:
            query (str): 삭제할 페이지 ID 또는 페이지 제목
        """
        try:
            headers = get_notion_headers()
            page_id = None

            # 먼저 페이지 ID로 시도
            print(f"🔍 '{query}' 페이지 ID로 검색 시도...")
            try:
                response = requests.get(
                    f"https://api.notion.com/v1/pages/{query}", headers=headers
                )
                if response.status_code == 200:
                    page_id = query
                    print(f"✅ 페이지 ID로 찾음")
                else:
                    raise Exception("페이지 ID로 찾을 수 없음")
            except Exception:
                # 페이지 ID로 못 찾으면 제목으로 검색
                print(f"🔎 '{query}' 제목으로 페이지 검색 중...")
                matching_pages = find_page_by_title(query)

                if not matching_pages:
                    return f"❌ '{query}' 제목의 페이지를 찾을 수 없습니다.\n💡 다음을 확인해보세요:\n1. 'list_notion_content' 도구로 실제 페이지 제목 확인\n2. 정확한 제목으로 다시 시도\n3. 또는 정확한 페이지 ID 사용"
                elif len(matching_pages) > 1:
                    page_list = []
                    for i, page in enumerate(matching_pages[:5], 1):
                        page_list.append(
                            f"{i}. {page['title']} [ID: {page['id'][:8]}...]"
                        )
                    return (
                        f"❌ '{query}' 제목의 페이지가 여러 개 있습니다:\n"
                        + "\n".join(page_list)
                        + f"\n\n💡 정확한 페이지 ID를 사용해주세요."
                    )
                else:
                    # 하나만 찾은 경우
                    page_id = matching_pages[0]["id"]
                    print(
                        f"✅ 제목으로 찾음: {matching_pages[0]['title']} (ID: {page_id[:8]}...)"
                    )

            if not page_id:
                return f"❌ '{query}' 페이지를 찾을 수 없습니다."

            # 페이지 정보 조회
            response = requests.get(
                f"https://api.notion.com/v1/pages/{page_id}", headers=headers
            )

            if response.status_code != 200:
                return f"페이지 조회 실패: {response.json().get('message', '알 수 없는 오류')}"

            page_info = response.json()

            # 페이지 제목 추출
            title = "제목 없음"
            properties = page_info.get("properties", {})
            for prop_name, prop_value in properties.items():
                if prop_value.get("type") == "title":
                    title_array = prop_value.get("title", [])
                    if title_array:
                        title = (
                            title_array[0].get("text", {}).get("content", "제목 없음")
                        )
                    break

            # 페이지 아카이브 (Notion은 삭제 대신 아카이브)
            archive_data = {"archived": True}

            archive_response = requests.patch(
                f"https://api.notion.com/v1/pages/{query}",
                headers=headers,
                json=archive_data,
            )

            if archive_response.status_code == 200:
                return f"페이지가 아카이브되었습니다: {title} (ID: {query})"
            else:
                result = archive_response.json()
                return (
                    f"페이지 아카이브 실패: {result.get('message', '알 수 없는 오류')}"
                )

        except Exception as e:
            return f"Notion 페이지 삭제 중 오류: {str(e)}"

    return [
        Tool(
            name="list_notion_content",
            description="Notion 워크스페이스의 모든 페이지와 데이터베이스 목록을 조회합니다. '모든 페이지', '전체 목록' 등으로 요청하거나 특정 키워드로 검색할 수 있습니다.",
            func=list_all_content,
        ),
        Tool(
            name="create_notion_page",
            description="Notion에 새 페이지를 생성합니다. JSON 형태로 부모페이지ID, 제목, 내용을 입력하세요.",
            func=create_page,
        ),
        Tool(
            name="get_notion_content",
            description="Notion 페이지의 내용을 조회합니다. 페이지 제목만 입력하세요. 예: '시저회의공간', '프로젝트 계획'",
            func=get_page_content,
        ),
        Tool(
            name="update_notion_page",
            description='Notion 페이지를 수정합니다. JSON 형태로 입력하세요: {"page_title": "기존제목", "title": "새제목", "content": "새내용"}',
            func=update_page,
        ),
        Tool(
            name="delete_notion_page",
            description="Notion 페이지를 삭제합니다. 페이지 제목만 입력하세요. 예: '임시문서', '회의록'",
            func=delete_page,
        ),
    ]
