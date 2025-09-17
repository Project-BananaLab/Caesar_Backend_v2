from fastapi.middleware.cors import CORSMiddleware


# CORS 설정 추가
def add_cors_middleware(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 개발 환경에서만 사용, 프로덕션에서는 특정 도메인만 허용
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
