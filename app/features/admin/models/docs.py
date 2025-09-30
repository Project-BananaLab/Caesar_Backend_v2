# app/features/admin/models/docs.py
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, String, BigInteger, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.sql import func
from app.utils.db import Base

# 🆕 추가: 회사+해시 중복 방지
from sqlalchemy import UniqueConstraint, Index  # ← 추가 임포트

class Doc(Base):
    __tablename__ = "docs"
    """
    업로드된 파일 메타를 보관하고, VectorDB(Chroma) 메타데이터와 매핑하는 테이블.
    """

    # 🆕 수정: DB 제약 조건 제거, 애플리케이션 레벨에서 중복 처리
    __table_args__ = (
        # 조회 최적화용 인덱스만 유지
        Index("ix_docs_company_checksum", "company_id", "checksum_sha256"),
        Index("ix_docs_user_private", "employee_id", "is_private"),
    )

    # PK → VectorDB 메타데이터의 doc_id 로 저장 (1:1 연결)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # 회사 FK → 회사별 컬렉션 분리 기준
    company_id:  Mapped[int] = mapped_column(Integer, ForeignKey("company.id"), nullable=False)
    # 직원 FK (개인문서인 경우만 사용; 회사 공개문서는 NULL)
    employee_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # True=개인문서(업로드한 직원만 조회), False=회사공개(회사 직원 모두 조회)
    is_private: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # 업로드된 원본 파일명
    file_name: Mapped[str] = mapped_column(String(512), nullable=False)
    # S3 URL (퍼블릭 or 사내 접근용)
    file_url:  Mapped[str] = mapped_column(String(1024), nullable=False)
    
    # 파일 바이트 크기
    file_size:        Mapped[int | None] = mapped_column(BigInteger)
    # 컨텐츠 SHA256 (중복 방지/무결성 체크)
    checksum_sha256:  Mapped[str | None] = mapped_column(String(64))

    # 'pending'|'processing'|'succeeded'|'failed' (인덱싱 상태)
    ingest_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    # VectorDB에 저장된 청크 수
    chunks_count:  Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # 실패 사유 등 에러 메시지
    error_text:    Mapped[str | None] = mapped_column(Text)

    # 레코드 생성 시각 (DB now)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    # 레코드 갱신 시각 (DB now on update)
    updated_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    # VectorDB 반영 완료 시각
    ingested_at: Mapped[str | None] = mapped_column(DateTime(timezone=True))
