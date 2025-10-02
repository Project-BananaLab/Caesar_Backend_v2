# RAG 성능 테스트

내부 문서 RAG 시스템의 성능을 측정하고 시각화하는 도구입니다.

## 📁 파일 구조

```
test/
├── performance_test.py    # 핵심 성능 테스트 모듈
├── run_test.py           # 테스트 실행 스크립트
└── README.md             # 이 파일
```

## 🚀 빠른 시작

### 1. 필수 패키지 설치
```bash
pip install matplotlib
```

### 2. 테스트 실행
```bash
# 방법 1: 직접 실행
python app/rag/internal_data_rag/test/performance_test.py

# 방법 2: 실행 스크립트 사용
python app/rag/internal_data_rag/test/run_test.py
```

## 📊 측정 항목

### 성능 지표
1. **검색 시간** (Retrieval Time)
   - 문서 검색에 소요되는 시간 (초)
   
2. **총 응답 시간** (Total Response Time)
   - 검색 + 답변 생성 전체 시간 (초)
   
3. **정확도 점수** (Accuracy Score, 0-10)
   - 키워드 매칭 기반 간단한 정확도 평가
   - 답변 길이 및 관련성 고려

### 테스트 쿼리
- "생산현황 작성 시 주의사항 알려줘"
- "회사에서 브이로그 찍어 유튜브에 올리려고 하는데 회사 규정에 어긋나는지 알려줘"
- "직원의 연차휴가는 몇 일인가요?"
- "기록물 관리 절차에 대해 설명해주세요"
- 기타 회사 문서 관련 질문들

## 📈 결과 파일

테스트 실행 후 다음 파일들이 생성됩니다:

- `performance_results_YYYYMMDD_HHMMSS.json` - 원시 데이터
- `performance_chart_YYYYMMDD_HHMMSS.png` - 성능 시각화 차트

## 🔧 사용법

### 기본 사용
```python
from performance_test import RAGPerformanceTester

tester = RAGPerformanceTester()
results = tester.run_performance_test()

# 결과 저장
tester.save_results()

# 시각화 생성
tester.create_performance_chart(results)
```

### 커스텀 쿼리 테스트
```python
# 사용자 정의 쿼리로 개별 테스트
result = tester.test_single_query("내가 궁금한 질문")
```

## 📋 시각화 내용

생성되는 차트는 4개 섹션으로 구성됩니다:

1. **검색 시간** - 각 쿼리별 문서 검색 소요 시간
2. **총 응답 시간** - 각 쿼리별 전체 처리 시간
3. **정확도 점수** - 각 쿼리별 답변 품질 점수
4. **평균 성능 요약** - 전체 평균 성능 지표

📊 새로운 정확도 평가 시스템:
1. 키워드 기반 정확도 (30% 가중치)
   기존 방식 유지
   질문-답변 간 키워드 매칭 + 답변 길이 보정

2. LLM 기반 정확도 (50% 가중치)
   GPT-4o-mini를 사용한 답변 품질 평가
   정확성(40%) + 완성도(30%) + 관련성(20%) + 명확성(10%)

3. 임베딩 유사도 (20% 가중치)
   OpenAI text-embedding-3-small 사용
   질문-답변 간 의미적 코사인 유사도

4. 종합 정확도
   세 가지 평가의 가중 평균

## 🚨 문제 해결

### 자주 발생하는 오류

1. **ModuleNotFoundError: No module named 'internal_retrieve'**
   - RAG 서비스 모듈을 찾을 수 없음
   - 현재 디렉토리에서 실행하는지 확인

2. **matplotlib 관련 오류**
   ```bash
   pip install matplotlib
   ```

3. **한글 폰트 문제**
   - 시스템에 'Malgun Gothic' 폰트가 없는 경우
   - 차트는 생성되지만 한글이 깨질 수 있음

## 📝 참고사항

- 테스트는 실제 RAG 서비스를 사용하므로 OpenAI API 키와 Chroma Cloud 설정이 필요합니다
- 정확도 평가는 간단한 키워드 매칭 방식을 사용합니다
- 더 정교한 평가를 위해서는 별도의 Ground Truth 데이터셋이 필요합니다
