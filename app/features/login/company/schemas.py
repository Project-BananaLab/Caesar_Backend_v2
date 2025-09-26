# app/features/login/company/schemas.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class CompanyLoginIn(BaseModel):
    # 요청 본문은 { "coId": "..." } 형태로 받되, 코드에서는 payload.co_id 로 접근
    model_config = ConfigDict(populate_by_name=True)
    co_id: str = Field(alias="coId")

class CompanyLoginOut(BaseModel):
    # 응답을 camelCase로 직렬화해서 내려가도록 설정
    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)

    company_id: int = Field(alias="companyId")
    co_id: str = Field(alias="coId")
    co_name: Optional[str] = Field(default=None, alias="coName")
    role: str
    access_token: str = Field(alias="accessToken")

class NotionApiUpdateIn(BaseModel):
    # 요청 본문은 { "notionApiKey": "..." } 형태로 받음
    model_config = ConfigDict(populate_by_name=True)
    notion_api_key: str = Field(alias="notionApiKey")
