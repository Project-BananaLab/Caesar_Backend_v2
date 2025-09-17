import os
import requests
import tempfile
from dotenv import load_dotenv
from notion_client import Client
from langchain_community.document_loaders import NotionDBLoader
from openai import OpenAI

load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_TOKEN2")
START_PAGE_ID = (
    "264120560ff680198c0fefbbe17bfc2c"  # 시작 페이지 ID. 나중에 Frontend에서 받아올 것
)

notion = Client(auth=NOTION_TOKEN)
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


def process_all_content_recursively(parent_id: str, depth: int = 0):
    """
    페이지와 블록의 모든 계층 구조를 재귀적으로 탐색하는 통합 함수
    - parent_id: 페이지 또는 블록의 ID
    - depth: 현재 탐색 깊이 (들여쓰기용)
    """
    indent = "  " * depth
    all_text = ""

    try:
        # parent_id에 속한 자식 블록들을 가져옴 (페이지 또는 블록) - pagination 처리
        blocks = []
        start_cursor = None

        while True:
            response = notion.blocks.children.list(
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
                all_text += process_all_content_recursively(block["id"], depth + 1)

            # 3. '하위 페이지'가 아니면서 다른 자식 블록(들여쓰기)을 가졌는지 확인하고 재귀 호출
            elif block["has_children"]:
                all_text += process_all_content_recursively(block["id"], depth + 1)

    except Exception as e:
        all_text += f"{indent}🔥 ID({parent_id}) 처리 중 오류 발생: {e}\n"

    return all_text


# --- 스크립트 실행 ---
if __name__ == "__main__":
    try:
        start_page_info = notion.pages.retrieve(page_id=START_PAGE_ID)
        start_page_title_parts = start_page_info["properties"]["title"]["title"]
        start_page_title = (
            start_page_title_parts[0]["plain_text"]
            if start_page_title_parts
            else "(제목 없음)"
        )

        print(f"탐색 시작: {start_page_title} (ID: {START_PAGE_ID})\n" + "=" * 40)
        result = process_all_content_recursively(START_PAGE_ID)
        # print(result)
        print("=" * 40 + "\n탐색 완료.")

        # 임시 파일로 저장 (확인용)
        # import tempfile
        # with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as tmpfile:
        #     tmpfile.write(result)
        #     print(f"[INFO] 결과가 임시 파일에 저장됨: {tmpfile.name}")

    except Exception as e:
        print(f"🔥 시작 페이지({START_PAGE_ID})에 접근할 수 없습니다: {e}")
