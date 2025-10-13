# 🚀 Caesar 빠른 시작 가이드

## 🤔 어떤 방법을 선택할까요?

### 🐳 **Docker 사용 (권장 - 설정 간단)**

```bash
# 1. Docker Desktop 설치
# https://www.docker.com/products/docker-desktop/

# 2. 이미지 빌드 & 실행
docker build -f Dockerfile.ocr -t caesar-ocr .
docker run -p 8000:8000 caesar-ocr

# ✅ 완료! 브라우저에서 http://localhost:8000 접속
```

```
개발 단계에서는 옵션 B (백그라운드 실행)를 권장:

# 첫 실행
docker run -d -p 8000:8000 --name caesar-backend caesar-ocr

# 서버 재시작 (서버 종료 후)
docker start caesar-backend

# 로그 확인
docker logs caesar-backend

# 완전 중지
docker stop caesar-backend
```

**장점:**
- ✅ 한 번 설정으로 모든 기능 사용
- ✅ 팀원 간 동일한 환경
- ✅ OCR 자동 포함
- ✅ setup_ocr.py 실행 불필요

**단점:**
- ❌ Docker Desktop 설치 필요
- ❌ 디버깅이 약간 복잡

---


## 🎯 **상황별 권장**

| 상황 | 권장 방법 | 이유 |
|------|-----------|------|
| **새 팀원 합류** | 🐳 Docker | 빠른 온보딩 |
| **프론트엔드 개발** | 💻 로컬 (OCR=false) | 가벼운 환경 |
| **RAG/AI 개발** | 🐳 Docker | 모든 기능 필요 |
| **배포/운영** | 🐳 Docker | 안정성 |

---

## ⚡ **초고속 시작 (30초)**

### Docker 있으면:
```bash
docker run -p 8000:8000 caesar-ocr
```

### Docker 없으면:
```bash
pip install -r requirements.txt
cp env.example .env
# .env에서 ENABLE_OCR=false 설정
uvicorn app.main:app --reload
```

---

## 🆘 **문제 해결**

### "Docker가 실행되지 않아요"
- Docker Desktop이 실행 중인지 확인
- Windows: WSL2 활성화 필요할 수 있음

### "OCR 오류가 발생해요"
```bash
# 로컬 개발에서만 해당
python scripts/setup_ocr.py
# 또는 OCR 비활성화
echo "ENABLE_OCR=false" >> .env
```

### "포트 8000이 사용 중이에요"
```bash
# 다른 포트 사용
docker run -p 8001:8000 caesar-ocr
# 또는
uvicorn app.main:app --port 8001
```

---

## 🎉 **성공 확인**

브라우저에서 접속:
- http://localhost:8000 (메인 페이지)
- http://localhost:8000/docs (API 문서)

✅ 페이지가 로드되면 성공!
