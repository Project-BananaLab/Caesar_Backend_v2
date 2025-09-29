import os
import sys
import requests
import tempfile
from dotenv import load_dotenv
from notion_client import Client
from langchain_community.document_loaders import NotionDBLoader
from openai import OpenAI
from app.utils.db import get_notion_token_by_company
from app.features.auth.company_auth import get_current_company_admin
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

load_dotenv()

# def get_notion_token_by_company(company_id: int) -> str:
#     """회사 ID로 Notion API 토큰 가져오기"""
#     db = SessionLocal()
#     try:
#         company = db.query(Company).filter(Company.id == company_id).first()
#         if company and company.co_notion_API:
#             return decrypt_data(company.co_notion_API, return_type="string")
#     finally:
#         db.close()

def update_notion_token(company_id: int):
    """회사 ID로 NOTION_TOKEN 업데이트"""
    global NOTION_TOKEN, notion
    NOTION_TOKEN = get_notion_token_by_company(company_id)
    notion = Client(auth=NOTION_TOKEN)

def update_notion_token_from_auth(token: HTTPAuthorizationCredentials):
    """인증 토큰으로부터 자동으로 회사 ID를 가져와서 NOTION_TOKEN 업데이트"""
    current_company = get_current_company_admin(token)
    company_id = current_company["company_id"]
    update_notion_token(company_id)

# START_PAGE_ID = (
#     "264120560ff680198c0fefbbe17bfc2c"  # 시작 페이지 ID. 나중에 Frontend에서 받아올 것
# )

# DB에서 Notion API 토큰을 가져와서 초기화하는 함수
def initialize_notion_with_first_company():
    """서버 시작 시 첫 번째 회사의 Notion API로 초기화"""
    global NOTION_TOKEN, notion
    try:
        from app.utils.db import SessionLocal
        from app.features.login.company.models import Company
        
        db = SessionLocal()
        try:
            # 첫 번째 회사 가져오기 (또는 특정 조건으로 회사 선택)
            first_company = db.query(Company).filter(Company.co_notion_API.isnot(None)).first()
            if first_company:
                NOTION_TOKEN = get_notion_token_by_company(first_company.id)
                notion = Client(auth=NOTION_TOKEN)
                print(f"✅ Notion 초기화 완료 - 회사 ID: {first_company.id}")
                return True
        finally:
            db.close()
    except Exception as e:
        print(f"⚠️ Notion 초기화 실패: {e}")

# 초기값 설정
NOTION_TOKEN = None
notion = None

# 서버 시작 시 자동 초기화
initialize_notion_with_first_company()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 전처리된 데이터를 저장할 전역 리스트들
processed_images = []
processed_tables = []
processed_databases = []

# -------------------------------------------------------------------------------------------------------------------#


def download_image_temporarily(image_url, block_id):
    """이미지를 임시로 다운로드하는 함수"""
    try:
        # 이미지 다운로드
        response = requests.get(image_url)
        response.raise_for_status()

        # 임시 파일 생성
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        # macOS/Linux에서는 NamedTemporaryFile을 사용하면 되지만, Windows에서는 임시 파일 생성이 좀 더 복잡함.
        # 이를 해결하기 위해서는 tempfile.mkstemp() 또는 tempfile.TemporaryFile()를 사용할 수 있음.
        # 하지만 이 경우 파일 삭제 처리가 필요함.
        # macOS/Linux: 보통 /tmp/ 디렉토리
        # Windows: 보통 C:\Users\[사용자명]\AppData\Local\Temp\ 디렉토리
        temp_file.write(response.content)
        temp_file.close()

        print(f"이미지 다운로드 완료: {temp_file.name}")
        return temp_file.name

    except Exception as e:
        print(f"이미지 다운로드 중 오류 발생: {e}")
        return None


def analyze_image_with_gpt(image_path):
    """gpt-4o-mini를 사용해서 이미지를 분석하는 함수"""
    try:
        # 이미지 파일을 base64로 인코딩해서 GPT에 전송
        with open(image_path, "rb") as image_file:
            import base64

            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
**Role**

You are an expert AI that precisely analyzes and interprets images. Your task is to perform a multifaceted and in-depth analysis of a given image and provide a detailed, structured explanation of the results. Please write everything, including your final answer, in Korean.

**Analysis Procedure**
Follow the three-step process below to systematically conduct your analysis.

**Step 1: Identify the Core Types of the Image**

First, understand the overall composition of the image and select all applicable types from the list below. If multiple types appear, list the most core type first, followed by the remaining types.

- Portrait: An image featuring one or more people.
- Landscape: An image featuring a natural or cityscape.
- Object/Still Life: An image featuring a specific object or group of objects.
- Graph/Chart: An image visualizing data.
- Table: An image featuring a data table consisting of rows and columns.
- Composite: An image featuring a significant combination of two or more of the above elements.

**Step 2: Detailed Analysis by Identified Type**

For each type identified in Step 1, perform a detailed analysis according to the applicable guidelines below.

**[A] Character Analysis**
- **Basic Information:**
	- **People:** The total number of people visible in the image.
	- **Demographic Information:** The estimated age, gender, and ethnicity of each person.
	- **Appearance:** Facial features such as hairstyle, facial expressions, and gaze.
- **Dress and Style:**
	- **Clothes:** The type, color, design, and style of clothing worn (e.g., formal, casual, sportswear).
	- **Accessories:** Accessories worn, such as glasses, hats, watches, and jewelry.
- **Behaviors and Emotions:**
	- **Behaviors:** The specific actions or postures the person is currently performing.
	- **Emotional Inferences:** The emotional state inferred from facial expressions, gestures, and the situation (e.g., happiness, sadness, concentration, surprise).
- **Context and Background:**
	- **Location:** The space (indoors, outdoors, etc.) and surroundings where the person is located. 
	- **Context:** The overall situation as perceived through the surroundings and interactions with other characters.

**[B] Landscape Analysis**
- **Location and Geography:**
	- **Type of Place:** The type of landscape, such as mountains, ocean, city, forest, desert, or countryside.
	- **Geographical Features:** Visible features such as distinctive landforms, vegetation, or bodies of water.
	- **Artifacts:** Human-made structures, such as buildings, bridges, roads, or utility poles.
- **Time of Day and Weather:**
	- **Time of Day:** The time of day, as inferred from the direction and color of light (e.g., dawn, noon, dusk, or night).
	- **Weather:** The state of the sky (clear, cloudy, rainy, or snowy), and the weather as judged by the texture of the air.
- **Key Elements and Composition:**
	- **Primary Subject:** The natural or man-made object that receives the most visual emphasis.
	- **Composition:** The arrangement of the foreground, middle ground, and background, and the overall composition of the frame. 
- **Mood and Impression:**
	- **Overall Feeling:** The atmosphere evoked by the landscape, such as peace, grandeur, dynamism, solitude, and mystery.

**[C] Object/Still Life Analysis**
- **Object Identification:**
	- **Central Object:** The main object that serves as the focus of the image.
	- **Peripheral Objects:** Other objects arranged around the central object.
- **Form and Material:**
	- **Visual Characteristics:** The shape, color, size, pattern, and texture of the object.
	- **Material Inference:** The material from which the object is composed, such as wood, metal, plastic, glass, or fabric.
- **Function and Condition:**
	- **Use:** The object's original purpose or function.
	- **Condition:** The current condition of the object, such as new, worn, or otherwise clean, or dirty.

**[D] Graph/Chart Analysis**
- **Basic Information:**
	- **Graph Type:** The type of graph, including bar, line, circle, scatter plot, and area plot.
	- **Title and Axis:** The full title of the graph, and the variables and units represented by the X-axis and Y-axis, respectively.
	- **Legend:** The meaning of each data series.
- **Data Interpretation:**
	- **Key Figures:** The maximum and minimum values, and important data at specific points on the graph.
	- **Trends and Patterns:** Changes over time (increases, decreases, fluctuations), comparisons between items, data distribution, and correlations.
- **Key Insights:**
	- **Message:** The most important information or conclusion this graph visually emphasizes.

**[E] Table Analysis**
- **Data Extraction:** Recognize all text data in the table in the image and convert it into an accurate Markdown table format, as shown below.
- **Structure Description:** Briefly explain what each row and column in the table represents.

Markdown

| Header 1 | Header 2 | Header 3 |
|----------|----------|-------------|
| Data 1-1 | Data 1-2 | Data 1-3 |
| Data 2-1 | Data 2-2 | Data 2-3 |

**[F] Composite Image Analysis**
- **Element Identification:** Identify each major element (e.g., person, object, background, text) in the image and briefly describe the characteristics of each element, referring to the items above (A, B, C, D, E).
- **Relationships Between Elements:** Analyze how each element is spatially arranged and what interactions or relationships they have with each other (e.g., a person using an object, a background emphasizing a person's emotions).
- **Overall Meaning:** Explain the overall story or message of the image formed through the interaction of each element.

**Step 3: Overall Conclusion**

Based on the detailed analysis above, draw a final conclusion about the image as a whole.

- **Summary of Analysis:** Briefly summarize the key points from Step 2 of your analysis, explaining what and how the image represents.
- **Theme and Interpretation:** Add your overall interpretation of what you believe the image's core theme or message is, and what emotions or thoughts it evokes in the viewer.

**Now, begin your analysis of the provided image.**
                            """,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_completion_tokens=2000,
        )

        # GPT 응답에서 텍스트 추출
        description = response.choices[0].message.content
        print(f"이미지 분석 완료: {len(description)}자의 설명 생성됨")
        return description

    except Exception as e:
        print(f"이미지 분석 중 오류 발생: {e}")
        return f"이미지 분석 실패: {str(e)}"


def delete_temporary_file(file_path):
    """임시 파일을 삭제하는 함수"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"임시 파일 삭제 완료: {file_path}")
        else:
            print(f"삭제할 파일이 존재하지 않음: {file_path}")
    except Exception as e:
        print(f"파일 삭제 중 오류 발생: {e}")


def process_image_block(block: dict) -> str:
    """이미지 블록을 전처리하는 함수"""
    try:
        block_id = block["id"]
        image_data = block["image"]
        # 이미지 URL 추출 (external 또는 file 타입)
        image_url = None

        if image_data.get("type") == "external":
            image_url = image_data.get("external", {}).get("url")
        elif image_data.get("type") == "file":
            image_url = image_data.get("file", {}).get("url")
        if not image_url:
            return "이미지 URL을 찾을 수 없습니다."

        temp_file_path = download_image_temporarily(image_url, block_id)
        if not temp_file_path:
            return "이미지 다운로드 실패"

        description = analyze_image_with_gpt(temp_file_path)

        delete_temporary_file(temp_file_path)

        return description

    except Exception as e:
        return f"이미지 처리 중 오류 발생: {str(e)}"


# -------------------------------------------------------------------------------------------------------------------#


def process_table_block_enhanced(block: dict) -> str:
    """표 블록을 마크다운 형식으로 전처리하는 함수"""
    try:
        table_block_id = block["id"]
        all_rows_data = []

        # 표 블록의 자식인 'table_row' 블록들을 가져옵니다. - pagination 처리
        table_rows = []
        start_cursor = None

        while True:
            response = notion.blocks.children.list(
                block_id=table_block_id, start_cursor=start_cursor
            )
            table_rows.extend(response.get("results", []))
            if response.get("has_more"):
                start_cursor = response.get("next_cursor")
            else:
                break

        # 각 행(table_row)을 순회하며 셀 데이터를 추출합니다
        for row in table_rows:
            row_cells = row["table_row"]["cells"]
            row_data = []

            # 각 셀의 텍스트를 추출합니다.
            for cell in row_cells:
                cell_text = "".join([part["plain_text"] for part in cell])
                row_data.append(cell_text)

            all_rows_data.append(row_data)

        if not all_rows_data:
            return "빈 표입니다."

        # 마크다운 형식으로 변환
        markdown_table = ""
        for i, row_content in enumerate(all_rows_data):
            # 셀 내용을 | 로 구분
            markdown_table += "| " + " | ".join(row_content) + " |\n"

            # 첫 번째 행 다음에 헤더 구분선 추가
            if i == 0:
                markdown_table += "|" + "|".join([" --- " for _ in row_content]) + "|\n"

        # 전처리된 표 데이터 저장
        processed_tables.append(
            {
                "type": "table",
                "block_id": table_block_id,
                "content": markdown_table,
                "metadata": {
                    "block_id": table_block_id,
                    "content_type": "table_markdown",
                    "rows_count": len(all_rows_data),
                    "columns_count": len(all_rows_data[0]) if all_rows_data else 0,
                },
            }
        )

        return f"[표]\n{markdown_table}"

    except Exception as e:
        return f"표 처리 중 오류 발생: {str(e)}"


# -------------------------------------------------------------------------------------------------------------------#


def get_property_value(prop):
    """
    속성(property) 객체에서 실제 값을 추출합니다.
    """
    prop_type = prop.get("type")

    if prop_type == "title":
        return prop["title"][0]["plain_text"] if prop["title"] else None
    if prop_type == "rich_text":
        return prop["rich_text"][0]["plain_text"] if prop["rich_text"] else None
    if prop_type == "number":
        return prop["number"]
    if prop_type == "select":
        return prop["select"]["name"] if prop["select"] else None
    if prop_type == "status":
        return prop["status"]["name"] if prop["status"] else None
    if prop_type == "multi_select":
        return [s["name"] for s in prop["multi_select"]]
    if prop_type == "date":
        date_info = prop["date"]
        if date_info:
            return (
                f"{date_info['start']} ~ {date_info['end']}"
                if date_info["end"]
                else date_info["start"]
            )
        return None
    if prop_type == "formula":
        return prop["formula"][prop["formula"]["type"]]
    if prop_type == "relation":
        return [r["id"] for r in prop["relation"]]
    if prop_type == "rollup":
        # 롤업 타입에 따라 데이터 구조가 다를 수 있습니다.
        rollup_type = prop["rollup"]["type"]
        return prop["rollup"][rollup_type]
    if prop_type == "people":
        return [p["name"] for p in prop["people"]]
    if prop_type == "files":
        return [f["name"] for f in prop["files"]]
    if prop_type == "checkbox":
        return prop["checkbox"]
    if prop_type == "url":
        return prop["url"]
    if prop_type == "email":
        return prop["email"]
    if prop_type == "phone_number":
        return prop["phone_number"]
    if prop_type == "created_time":
        return prop["created_time"]
    if prop_type == "created_by":
        return prop["created_by"]["name"]
    if prop_type == "last_edited_time":
        return prop["last_edited_time"]
    if prop_type == "last_edited_by":
        return prop["last_edited_by"]["name"]
    if prop_type == "unique_id":
        prefix = prop["unique_id"].get("prefix") or ""
        number = prop["unique_id"]["number"]
        return f"{prefix}-{number}"

    # Button과 같은 값 없는 타입은 처리하지 않음
    return "Unsupported property type"


def process_database_block_enhanced(block: dict) -> str:
    """데이터베이스 블록을 전처리하는 함수"""
    try:
        child_db_id = block["id"]
        database_title = block["child_database"]["title"]

        # 데이터베이스의 모든 페이지 가져오기 - pagination 처리
        pages = []
        start_cursor = None

        while True:
            response = notion.databases.query(
                database_id=child_db_id, start_cursor=start_cursor
            )
            pages.extend(response.get("results", []))
            if response.get("has_more"):
                start_cursor = response.get("next_cursor")
            else:
                break

        result = f"[데이터베이스: {database_title}]\n"

        # 각 페이지를 순회하며 정보 출력
        for page in pages:
            page_id = page["id"]
            properties = page.get("properties", {})

            # 페이지 타이틀 추출
            page_title = "제목 없음"
            for prop_name, prop_data in properties.items():
                if prop_data.get("type") == "title":
                    page_title = get_property_value(prop_data) or "제목 없음"
                    break

            result += f"\n=== 페이지: {page_title} ===\n"

            # 페이지 속성 정보 추가
            result += f"\n--- 페이지 속성 ---\n"
            for prop_name, prop_data in properties.items():
                value = get_property_value(prop_data)
                result += f"- {prop_name} ({prop_data['type']}): {value}\n"

            # 페이지 본문 내용을 재귀적으로 가져오기
            result += f"\n--- 페이지 본문 ---\n"
            page_content = process_all_content_recursively(page_id, depth=1)
            result += page_content + "\n"

        return result

    except Exception as e:
        return f"데이터베이스 처리 중 오류 발생: {str(e)}"


# -------------------------------------------------------------------------------------------------------------------#


def get_text_from_block(block: dict) -> str:
    """다양한 블록 타입에서 텍스트를 추출하는 함수"""
    block_type = block["type"]

    # 블록 타입에 따라 텍스트가 담긴 위치가 다름
    if block_type in [
        "paragraph",
        "heading_1",
        "heading_2",
        "heading_3",
        "bulleted_list_item",
        "numbered_list_item",
        "quote",
        "callout",
        "code",
        "toggle",
        "breadcrumb",
    ]:
        # 대부분의 텍스트는 해당 타입 이름의 키 값 안에 'rich_text' 배열로 존재
        text_parts = block[block_type].get("rich_text", [])

    elif block_type == "to_do":
        text_parts = block["to_do"].get("rich_text", [])
        checked = block["to_do"]["checked"]
        return f"[{'x' if checked else ' '}] {''.join([part['plain_text'] for part in text_parts])}"

    elif block_type == "child_page":
        return f"{block['child_page']['title']} (하위 페이지)"

    elif block_type == "child_database":
        return process_database_block_enhanced(block)

    elif block_type == "bookmark":
        return f"{block['bookmark']['url']} (북마크)"

    elif block_type == "table":
        return process_table_block_enhanced(block)

    elif block_type == "file":
        return f"{block['file']['name']} (파일)"

    elif block_type == "image":
        return process_image_block(block)

    else:
        # 지원하지 않는 블록 타입은 건너뜀
        return ""

    # rich_text 배열의 모든 텍스트 조각을 하나로 합침
    return "".join([part["plain_text"] for part in text_parts])


# -------------------------------------------------------------------------------------------------------------------#


def search_notion_pages(query: str, notion_client=None) -> list:
    """
    Notion API search를 사용해서 특정 페이지나 데이터베이스를 검색하는 함수
    
    Args:
        query (str): 검색할 키워드 (페이지 제목, 데이터베이스 이름 등)
        notion_client: Notion 클라이언트 (선택사항, 없으면 전역 notion 사용)
    
    Returns:
        list: 검색 결과 리스트
    """
    client = notion_client if notion_client else notion
    
    if not client:
        raise Exception("Notion 클라이언트가 초기화되지 않았습니다.")
    
    try:
        # Notion API search 호출
        response = client.search(
            query=query,
            sort={
                "direction": "descending",
                "timestamp": "last_edited_time"
            },
            page_size=100  # 최대 100개 결과
        )
        
        results = response.get("results", [])
        
        # 결과를 정리해서 반환
        formatted_results = []
        for result in results:
            result_info = {
                "id": result["id"],
                "object": result["object"],  # "page" 또는 "database"
                "last_edited_time": result["last_edited_time"],
                "url": result.get("url", "")
            }
            
            # 페이지의 경우
            if result["object"] == "page":
                # 제목 추출
                properties = result.get("properties", {})
                title = "제목 없음"
                
                # title 속성 찾기
                for prop_name, prop_data in properties.items():
                    if prop_data.get("type") == "title":
                        title_parts = prop_data.get("title", [])
                        if title_parts:
                            title = title_parts[0].get("plain_text", "제목 없음")
                        break
                
                result_info["title"] = title
                
            # 데이터베이스의 경우
            elif result["object"] == "database":
                title_parts = result.get("title", [])
                title = "제목 없음"
                if title_parts:
                    title = title_parts[0].get("plain_text", "제목 없음")
                result_info["title"] = title
            
            formatted_results.append(result_info)
        
        return formatted_results
        
    except Exception as e:
        print(f"🔥 Notion 검색 중 오류 발생: {e}")
        return []


def get_workspace_root_pages(notion_client=None) -> list:
    """
    Notion 워크스페이스의 루트 레벨 페이지들을 가져오는 함수
    
    Args:
        notion_client: Notion 클라이언트 (선택사항, 없으면 전역 notion 사용)
    
    Returns:
        list: 루트 페이지들의 정보 리스트
    """
    client = notion_client if notion_client else notion
    
    if not client:
        raise Exception("Notion 클라이언트가 초기화되지 않았습니다.")
    
    try:
        # 빈 쿼리로 검색하면 모든 페이지/데이터베이스를 가져옴
        response = client.search(
            query="",
            sort={
                "direction": "descending",
                "timestamp": "last_edited_time"
            },
            filter={
                "property": "object",
                "value": "page"
            },
            page_size=100
        )
        
        results = response.get("results", [])
        root_pages = []
        
        for result in results:
            # parent가 workspace인 페이지만 선택 (루트 레벨)
            parent = result.get("parent", {})
            if parent.get("type") == "workspace":
                page_info = {
                    "id": result["id"],
                    "title": "제목 없음",
                    "url": result.get("url", ""),
                    "last_edited_time": result["last_edited_time"]
                }
                
                # 제목 추출
                properties = result.get("properties", {})
                for prop_name, prop_data in properties.items():
                    if prop_data.get("type") == "title":
                        title_parts = prop_data.get("title", [])
                        if title_parts:
                            page_info["title"] = title_parts[0].get("plain_text", "제목 없음")
                        break
                
                root_pages.append(page_info)
        
        return root_pages
        
    except Exception as e:
        print(f"🔥 워크스페이스 페이지 조회 중 오류 발생: {e}")
        return []


def find_start_page_by_title(title_keyword: str, notion_client=None) -> str:
    """
    페이지 제목으로 START_PAGE_ID를 찾는 함수
    
    Args:
        title_keyword (str): 찾고자 하는 페이지 제목의 키워드
        notion_client: Notion 클라이언트 (선택사항, 없으면 전역 notion 사용)
    
    Returns:
        str: 찾은 페이지의 ID, 없으면 빈 문자열
    """
    search_results = search_notion_pages(title_keyword, notion_client)
    
    # 정확히 일치하는 제목 우선 검색
    for result in search_results:
        if result["title"].lower() == title_keyword.lower():
            print(f"✅ 페이지 발견: {result['title']} (ID: {result['id']})")
            return result["id"]
    
    # 부분 일치하는 제목 검색
    for result in search_results:
        if title_keyword.lower() in result["title"].lower():
            print(f"✅ 페이지 발견 (부분일치): {result['title']} (ID: {result['id']})")
            return result["id"]
    
    print(f"⚠️ '{title_keyword}'와 일치하는 페이지를 찾을 수 없습니다.")
    return ""


def get_available_start_pages(notion_client=None) -> list:
    """
    시작 페이지로 사용 가능한 페이지들의 목록을 반환하는 함수
    
    Args:
        notion_client: Notion 클라이언트 (선택사항, 없으면 전역 notion 사용)
    
    Returns:
        list: 사용 가능한 페이지들의 정보 리스트 (id, title, url 포함)
    """
    # 1. 워크스페이스 루트 페이지들 조회
    root_pages = get_workspace_root_pages(notion_client)
    
    # 2. 모든 페이지 검색 (빈 쿼리)
    all_pages = search_notion_pages("", notion_client)
    
    # 결과를 합치고 중복 제거
    all_available = {page["id"]: page for page in root_pages + all_pages}
    
    # 리스트로 변환하고 제목순 정렬
    available_pages = list(all_available.values())
    available_pages.sort(key=lambda x: x["title"])
    
    return available_pages


def update_start_page_id(new_start_page_id: str):
    """
    START_PAGE_ID를 동적으로 업데이트하는 함수
    
    Args:
        new_start_page_id (str): 새로운 시작 페이지 ID
    """
    global START_PAGE_ID
    
    if not new_start_page_id:
        raise ValueError("START_PAGE_ID는 빈 문자열일 수 없습니다.")
    
    # 페이지가 실제로 존재하는지 확인
    client = notion if notion else None
    if client:
        try:
            page_info = client.pages.retrieve(page_id=new_start_page_id)
            properties = page_info.get("properties", {})
            title = "제목 없음"
            
            for prop_name, prop_data in properties.items():
                if prop_data.get("type") == "title":
                    title_parts = prop_data.get("title", [])
                    if title_parts:
                        title = title_parts[0].get("plain_text", "제목 없음")
                    break
            
            START_PAGE_ID = new_start_page_id
            print(f"✅ START_PAGE_ID 업데이트 완료: {title} (ID: {START_PAGE_ID})")
            
        except Exception as e:
            raise ValueError(f"유효하지 않은 페이지 ID입니다: {e}")
    else:
        # Notion 클라이언트가 없는 경우 그냥 업데이트
        START_PAGE_ID = new_start_page_id
        print(f"⚠️ Notion 클라이언트 없음 - START_PAGE_ID 업데이트: {START_PAGE_ID}")


def process_all_content_recursively(parent_id: str, depth: int = 0, notion_client=None):
    """
    페이지와 블록의 모든 계층 구조를 재귀적으로 탐색하는 통합 함수
    - parent_id: 페이지 또는 블록의 ID
    - depth: 현재 탐색 깊이 (들여쓰기용)
    - notion_client: Notion 클라이언트 (선택사항, 없으면 전역 notion 사용)
    """
    indent = "  " * depth
    all_text = ""
    
    # notion_client가 제공되지 않으면 전역 notion 사용 (기존 호환성)
    client = notion_client if notion_client else notion

    try:
        # parent_id에 속한 자식 블록들을 가져옴 (페이지 또는 블록) - pagination 처리
        blocks = []
        start_cursor = None

        while True:
            response = client.blocks.children.list(
                block_id=parent_id, start_cursor=start_cursor
            )
            blocks.extend(response.get("results", []))
            if response.get("has_more"):
                start_cursor = response.get("next_cursor")
            else:
                break

        for block in blocks:
            # 1. 현재 블록의 내용을 먼저 가져옴
            block_text = get_text_from_block(block)
            if block_text:
                all_text += f"{indent}- {block_text}\n"

            # 2. 이 블록이 '하위 페이지'인지 확인하고 재귀 호출
            if block["type"] == "child_page":
                all_text += process_all_content_recursively(block["id"], depth + 1, notion_client)

            # 3. '하위 페이지'가 아니면서 다른 자식 블록(들여쓰기)을 가졌는지 확인하고 재귀 호출
            elif block["has_children"]:
                all_text += process_all_content_recursively(block["id"], depth + 1, notion_client)

    except Exception as e:
        all_text += f"{indent}🔥 ID({parent_id}) 처리 중 오류 발생: {e}\n"

    return all_text


# --- 스크립트 실행 ---
if __name__ == "__main__":
    try:
        print("=== Notion 페이지 탐색 도구 ===")
        
        # 1. 사용 가능한 페이지 목록 표시
        print("\n📋 사용 가능한 시작 페이지들:")
        available_pages = get_available_start_pages()
        
        if available_pages:
            for i, page in enumerate(available_pages[:10], 1):  # 최대 10개만 표시
                print(f"{i:2d}. {page['title']} (ID: {page['id'][:8]}...)")
        
        # 2. 현재 설정된 START_PAGE_ID로 시작
        print(f"\n🎯 현재 시작 페이지 ID: {START_PAGE_ID}")
        
        # 시작 페이지 정보 가져오기
        start_page_info = notion.pages.retrieve(page_id=START_PAGE_ID)
        start_page_title_parts = start_page_info["properties"]["title"]["title"]
        start_page_title = (
            start_page_title_parts[0]["plain_text"]
            if start_page_title_parts
            else "(제목 없음)"
        )

        print(f"✅ 탐색 시작: {start_page_title} (ID: {START_PAGE_ID})\n" + "=" * 40)
        
        # 3. 페이지 내용 재귀 탐색
        result = process_all_content_recursively(START_PAGE_ID)
        
        print("=" * 40 + f"\n✅ 탐색 완료: 총 {len(result.splitlines())}줄")
        
        # 4. 결과 요약 표시
        lines = result.splitlines()
        image_count = len([line for line in lines if "[이미지]" in line or "이미지 분석" in line])
        table_count = len([line for line in lines if "[표]" in line])
        database_count = len([line for line in lines if "[데이터베이스" in line])
        
        print(f"📊 처리된 콘텐츠:")
        print(f"   - 이미지: {image_count}개")
        print(f"   - 표: {table_count}개")
        print(f"   - 데이터베이스: {database_count}개")

        # 5. 임시 파일로 저장 (확인용)
        # import tempfile
        # with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as tmpfile:
        #     tmpfile.write(result)
        #     print(f"[INFO] 결과가 임시 파일에 저장됨: {tmpfile.name}")

    except Exception as e:
        print(f"🔥 시작 페이지({START_PAGE_ID})에 접근할 수 없습니다: {e}")
        print("\n💡 다른 페이지를 시도해보세요:")
        
        # 오류 발생 시 대안 제시
        try:
            available_pages = get_available_start_pages()
            if available_pages:
                print("사용 가능한 페이지 목록:")
                for page in available_pages[:5]:
                    print(f"  - {page['title']} (ID: {page['id']})")
        except:
            print("사용 가능한 페이지 목록을 가져올 수 없습니다.")


# === 사용 예시 함수들 ===

def demo_search_pages():
    """Notion API 검색 기능 데모"""
    print("\n=== Notion 페이지 검색 데모 ===")
    
    # 키워드로 검색
    search_keyword = input("검색할 키워드를 입력하세요 (빈 입력 시 모든 페이지): ")
    results = search_notion_pages(search_keyword)
    
    if results:
        print(f"\n🔍 '{search_keyword}' 검색 결과: {len(results)}개")
        for i, result in enumerate(results[:5], 1):
            print(f"{i}. {result['title']} ({result['object']})")
            print(f"   ID: {result['id']}")
            print(f"   URL: {result['url']}")
            print()
    else:
        print("검색 결과가 없습니다.")


def demo_set_start_page():
    """START_PAGE_ID 설정 데모"""
    print("\n=== 시작 페이지 설정 데모 ===")
    
    # 사용 가능한 페이지 표시
    pages = get_available_start_pages()
    if not pages:
        print("사용 가능한 페이지가 없습니다.")
        return
    
    print("사용 가능한 페이지:")
    for i, page in enumerate(pages[:10], 1):
        print(f"{i:2d}. {page['title']}")
    
    try:
        choice = input("\n페이지 번호를 선택하세요 (1-10, 또는 't'로 제목 검색): ")
        
        if choice.lower() == 't':
            title = input("찾을 페이지 제목을 입력하세요: ")
            page_id = find_start_page_by_title(title)
            if page_id:
                update_start_page_id(page_id)
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(pages):
                update_start_page_id(pages[idx]['id'])
            else:
                print("잘못된 번호입니다.")
    
    except (ValueError, IndexError):
        print("잘못된 입력입니다.")


def interactive_mode():
    """대화형 모드"""
    print("\n=== Notion API 도구 대화형 모드 ===")
    
    while True:
        print("\n메뉴:")
        print("1. 페이지 검색")
        print("2. 시작 페이지 설정")
        print("3. 현재 설정 확인")
        print("4. 페이지 탐색 실행")
        print("5. 종료")
        
        choice = input("\n선택하세요 (1-5): ").strip()
        
        if choice == '1':
            demo_search_pages()
        elif choice == '2':
            demo_set_start_page()
        elif choice == '3':
            print(f"\n현재 START_PAGE_ID: {START_PAGE_ID}")
            try:
                page_info = notion.pages.retrieve(page_id=START_PAGE_ID)
                properties = page_info.get("properties", {})
                title = "제목 없음"
                
                for prop_name, prop_data in properties.items():
                    if prop_data.get("type") == "title":
                        title_parts = prop_data.get("title", [])
                        if title_parts:
                            title = title_parts[0].get("plain_text", "제목 없음")
                        break
                
                print(f"페이지 제목: {title}")
            except Exception as e:
                print(f"페이지 정보를 가져올 수 없습니다: {e}")
                
        elif choice == '4':
            try:
                print("\n페이지 탐색을 시작합니다...")
                result = process_all_content_recursively(START_PAGE_ID)
                print(f"✅ 탐색 완료: 총 {len(result.splitlines())}줄")
            except Exception as e:
                print(f"탐색 중 오류: {e}")
                
        elif choice == '5':
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다.")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--interactive":
    interactive_mode()
