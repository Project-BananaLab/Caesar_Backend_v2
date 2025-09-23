import os
from dotenv import load_dotenv
from notion_client import Client
from get_text_from_notion import process_all_content_recursively
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

# .env 파일에서 환경 변수 로드
load_dotenv()

# Notion API 토큰과 시작 페이지 ID 설정
NOTION_TOKEN = os.getenv("NOTION_TOKEN2")
START_PAGE_ID = '264120560ff680198c0fefbbe17bfc2c' # Caesar 프로젝트의 시작 페이지 ID

# OpenAI API 키 존재 여부 확인
if os.getenv("OPENAI_API_KEY") is None:
    print("경고: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

# Chroma Cloud API 키 존재 여부 확인
if os.getenv("CHROMA_API_KEY") is None:
    print("경고: CHROMA_API_KEY 환경 변수가 설정되지 않았습니다.")
if os.getenv("CHROMA_TENANT") is None:
    print("경고: CHROMA_TENANT 환경 변수가 설정되지 않았습니다.")
if os.getenv("CHROMA_DATABASE") is None:
    print("경고: CHROMA_DATABASE 환경 변수가 설정되지 않았습니다.")

# Notion 클라이언트 초기화
notion = Client(auth=NOTION_TOKEN)

# Notion 페이지에서 텍스트 추출
print("Notion 페이지에서 텍스트를 추출합니다...")
text_content = process_all_content_recursively(START_PAGE_ID)
print("텍스트 추출 완료.")

# 텍스트를 청크로 분할
print("텍스트를 청크로 분할합니다...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=128,
)
texts = text_splitter.split_text(text_content)
print(f"{len(texts)}개의 청크로 분할되었습니다.")

# 임베딩 모델 초기화
print("임베딩을 생성하고 ChromaDB에 저장합니다...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#-------------------Chroma Cloud 사용---------------#
# ChromaDB 컬렉션 이름 설정
collection_name = "notion-collection"

# Chroma Cloud 클라이언트 초기화
client = chromadb.CloudClient(
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
    api_key=os.getenv("CHROMA_API_KEY")
)


# ChromaDB에 데이터 저장
vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    collection_name=collection_name,
    client=client
)
#--------------------------------------------------#

#-------------------로컬 ChromaDB 사용---------------#
# persist_directory = "./chroma_db"
# collection_name = "notion-collection"

# vectorstore = Chroma.from_texts(
#     texts=texts,
#     embedding=embeddings,
#     collection_name=collection_name,
#     persist_directory=persist_directory
# )
#--------------------------------------------------#

print("임베딩 및 저장이 완료되었습니다.")