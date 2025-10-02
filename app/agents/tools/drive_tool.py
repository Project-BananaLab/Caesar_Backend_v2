# tools/drive_tool.py
from langchain.tools import Tool
from googleapiclient.discovery import build
import json
from google.oauth2.credentials import Credentials


def create_drive_tools(user_id: str, cookies: dict = None):
    """Google Drive Tool 생성"""

    def get_drive_service():
        """Google Drive API 서비스 생성"""
        try:
            # 쿠키에서 Google 액세스 토큰 추출
            if not cookies:
                raise Exception("쿠키 정보가 없습니다.")

            # 쿠키에서 Google 액세스 토큰 찾기 (다양한 키 이름 시도)
            access_token = None
            possible_keys = [
                "google_access_token",
                "access_token",
                "googleAccessToken",
                "token",
            ]

            for key in possible_keys:
                if key in cookies:
                    access_token = cookies[key]
                    print(f"✅ 쿠키에서 Google 토큰 찾음: {key}")
                    break

            if not access_token:
                available_keys = list(cookies.keys())
                raise Exception(
                    f"쿠키에서 Google 액세스 토큰을 찾을 수 없습니다. 사용 가능한 키: {available_keys}"
                )

            # Google Credentials 객체 생성
            creds = Credentials(token=access_token)

            return build("drive", "v3", credentials=creds)
        except Exception as e:
            raise Exception(f"Google Drive 서비스 초기화 실패: {str(e)}")

    def list_files(query: str) -> str:
        """Drive 파일 목록 조회
        Args:
            query (str): 검색할 파일명 또는 "all"로 전체 조회
        """
        try:
            service = get_drive_service()

            # 검색 쿼리 설정
            search_query = ""
            if query.lower() != "all":
                search_query = f"name contains '{query}'"

            results = (
                service.files()
                .list(
                    q=search_query,
                    pageSize=10,
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime, webViewLink, webContentLink)",
                )
                .execute()
            )

            files = results.get("files", [])

            if not files:
                return "파일을 찾을 수 없습니다."

            result = []
            for i, file in enumerate(files):
                try:
                    name = file.get("name", "알 수 없는 파일")
                    file_id = file.get("id", "")
                    file_type = (
                        "폴더"
                        if file.get("mimeType") == "application/vnd.google-apps.folder"
                        else "파일"
                    )
                    modified = (
                        file.get("modifiedTime", "")[:10]
                        if file.get("modifiedTime")
                        else "알 수 없음"
                    )

                    # 다운로드 링크 생성 (폴더가 아닌 경우에만)
                    if file_type == "파일" and file_id:
                        # Google Drive 다운로드 링크 생성
                        download_link = (
                            f"https://drive.google.com/uc?export=download&id={file_id}"
                        )
                        view_link = file.get("webViewLink", "")

                        result.append(
                            f"• {name} ({file_type}) - 수정일: {modified}\n"
                            f"  📥 다운로드: {download_link}\n"
                            f"  👁️ 미리보기: {view_link}"
                        )
                        print(f"✅ 파일 {i+1} 처리 완료: {name}")
                    else:
                        result.append(f"• {name} ({file_type}) - 수정일: {modified}")
                        print(f"✅ 폴더 {i+1} 처리 완료: {name}")

                except Exception as file_error:
                    print(f"❌ 파일 {i+1} 처리 중 오류: {file_error}")
                    # 오류가 발생해도 계속 진행
                    continue

            return f"Drive 파일 목록 ({len(files)}개):\n" + "\n".join(result)

        except Exception as e:
            return f"Drive 파일 조회 중 오류: {str(e)}"

    def create_folder(query: str) -> str:
        """Drive 폴더 생성
        Args:
            query (str): 생성할 폴더명
        """
        try:
            service = get_drive_service()

            folder_metadata = {
                "name": query,
                "mimeType": "application/vnd.google-apps.folder",
            }

            folder = service.files().create(body=folder_metadata, fields="id").execute()

            return f"폴더가 생성되었습니다: {query} (ID: {folder.get('id')})"

        except Exception as e:
            return f"폴더 생성 중 오류: {str(e)}"

    def share_file(query: str) -> str:
        """파일 공유 설정
        Args:
            query (str): JSON 형태 {"file_id": "파일ID", "email": "공유할이메일", "role": "reader|writer"}
        """
        try:
            service = get_drive_service()

            # JSON 파싱
            share_data = json.loads(query)
            file_id = share_data.get("file_id")
            email = share_data.get("email")
            role = share_data.get("role", "reader")

            permission = {"type": "user", "role": role, "emailAddress": email}

            service.permissions().create(
                fileId=file_id, body=permission, sendNotificationEmail=True
            ).execute()

            return f"파일이 {email}에게 {role} 권한으로 공유되었습니다."

        except json.JSONDecodeError:
            return 'JSON 형식이 올바르지 않습니다. 예: {"file_id": "abc123", "email": "user@example.com", "role": "reader"}'
        except Exception as e:
            return f"파일 공유 중 오류: {str(e)}"

    def rename_file(query: str) -> str:
        """Drive 파일/폴더 이름 변경
        Args:
            query (str): JSON 형태 {"file_id": "파일ID", "new_name": "새이름"}
        """
        try:
            service = get_drive_service()
            file_data = json.loads(query)

            file_id = file_data.get("file_id")
            new_name = file_data.get("new_name")

            if not file_id or not new_name:
                return "파일 ID와 새 이름이 필요합니다."

            # 파일 존재 확인
            try:
                file_info = service.files().get(fileId=file_id).execute()
                old_name = file_info.get("name", "알 수 없음")
            except Exception:
                return f"파일 ID '{file_id}'를 찾을 수 없습니다."

            # 이름 변경
            body = {"name": new_name}
            updated_file = service.files().update(fileId=file_id, body=body).execute()

            return f"파일 이름이 변경되었습니다: '{old_name}' -> '{new_name}' (ID: {file_id})"

        except json.JSONDecodeError:
            return 'JSON 형식이 올바르지 않습니다. 예: {"file_id": "abc123", "new_name": "새파일명.txt"}'
        except Exception as e:
            return f"파일 이름 변경 중 오류: {str(e)}"

    def delete_file(query: str) -> str:
        """Drive 파일/폴더 삭제
        Args:
            query (str): 삭제할 파일 ID
        """
        try:
            service = get_drive_service()

            # 파일 존재 확인
            try:
                file_info = service.files().get(fileId=query).execute()
                file_name = file_info.get("name", "알 수 없음")
                file_type = (
                    "폴더"
                    if file_info.get("mimeType") == "application/vnd.google-apps.folder"
                    else "파일"
                )
            except Exception:
                return f"파일 ID '{query}'를 찾을 수 없습니다."

            # 파일 삭제
            service.files().delete(fileId=query).execute()

            return f"{file_type}이 삭제되었습니다: {file_name} (ID: {query})"

        except Exception as e:
            return f"파일 삭제 중 오류: {str(e)}"

    def upload_file(query: str) -> str:
        """Drive에 텍스트 파일 업로드
        Args:
            query (str): JSON 형태 {"name": "파일명.txt", "content": "파일내용", "parent_id": "부모폴더ID(선택)"}
        """
        try:
            service = get_drive_service()
            file_data = json.loads(query)

            file_name = file_data.get("name", "새파일.txt")
            file_content = file_data.get("content", "")
            parent_id = file_data.get("parent_id")

            file_metadata = {"name": file_name}
            if parent_id:
                file_metadata["parents"] = [parent_id]

            # 파일 업로드 (간단한 텍스트 파일)
            from googleapiclient.http import MediaInMemoryUpload

            media = MediaInMemoryUpload(
                file_content.encode("utf-8"), mimetype="text/plain"
            )

            file = (
                service.files()
                .create(body=file_metadata, media_body=media, fields="id,name")
                .execute()
            )

            return f"파일이 업로드되었습니다: {file.get('name')} (ID: {file.get('id')})"

        except json.JSONDecodeError:
            return 'JSON 형식이 올바르지 않습니다. 예: {"name": "test.txt", "content": "파일 내용"}'
        except Exception as e:
            return f"파일 업로드 중 오류: {str(e)}"

    return [
        Tool(
            name="list_drive_files",
            description="Google Drive에서 파일 목록을 조회합니다. 파일명으로 검색하거나 'all'로 전체 조회할 수 있습니다.",
            func=list_files,
        ),
        Tool(
            name="create_drive_folder",
            description="Google Drive에 새 폴더를 생성합니다. 폴더명을 입력하세요.",
            func=create_folder,
        ),
        Tool(
            name="share_drive_file",
            description="Google Drive 파일을 다른 사용자와 공유합니다. JSON 형태로 파일ID, 이메일, 권한을 입력하세요.",
            func=share_file,
        ),
        Tool(
            name="rename_drive_file",
            description="Google Drive 파일/폴더의 이름을 변경합니다. JSON 형태로 파일ID와 새 이름을 입력하세요.",
            func=rename_file,
        ),
        Tool(
            name="delete_drive_file",
            description="Google Drive 파일/폴더를 삭제합니다. 삭제할 파일 ID를 입력하세요.",
            func=delete_file,
        ),
        Tool(
            name="upload_drive_file",
            description="Google Drive에 텍스트 파일을 업로드합니다. JSON 형태로 파일명, 내용, 부모폴더ID를 입력하세요.",
            func=upload_file,
        ),
    ]
