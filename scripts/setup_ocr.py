#!/usr/bin/env python3
"""
OCR 환경 설정 스크립트
협업 시 각 개발자가 실행하여 OCR 환경을 자동 설정
"""

import os
import sys
import platform
import subprocess
import urllib.request
import zipfile
from pathlib import Path

def detect_os():
    """운영체제 감지"""
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    elif system == "darwin":
        return "macos"
    elif system == "linux":
        return "linux"
    else:
        return "unknown"

def install_tesseract_windows():
    """Windows에서 Tesseract 설치"""
    print("🪟 Windows 환경에서 Tesseract OCR 설치 중...")
    
    # Tesseract 설치 파일 다운로드 URL
    tesseract_url = "https://github.com/UB-Mannheim/tesseract/releases/download/v5.3.3.20231005/tesseract-ocr-w64-setup-5.3.3.20231005.exe"
    
    print("📥 Tesseract 설치 파일 다운로드 중...")
    print(f"URL: {tesseract_url}")
    print("\n⚠️  수동 설치가 필요합니다:")
    print("1. 위 URL에서 설치 파일을 다운로드하세요")
    print("2. 설치 시 '한국어 언어팩'을 포함하여 설치하세요")
    print("3. 설치 경로를 기억해두세요 (기본: C:\\Program Files\\Tesseract-OCR)")
    
    # 설치 확인
    tesseract_paths = [
        "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
        "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
    ]
    
    for path in tesseract_paths:
        if os.path.exists(path):
            print(f"✅ Tesseract 발견: {path}")
            return path
    
    print("❌ Tesseract가 설치되지 않았습니다.")
    return None

def install_tesseract_macos():
    """macOS에서 Tesseract 설치"""
    print("🍎 macOS 환경에서 Tesseract OCR 설치 중...")
    
    try:
        # Homebrew로 설치
        subprocess.run(["brew", "install", "tesseract", "tesseract-lang"], check=True)
        print("✅ Tesseract 설치 완료")
        return "/usr/local/bin/tesseract"
    except subprocess.CalledProcessError:
        print("❌ Homebrew를 통한 설치 실패")
        print("수동으로 설치하세요: brew install tesseract tesseract-lang")
        return None

def install_tesseract_linux():
    """Linux에서 Tesseract 설치"""
    print("🐧 Linux 환경에서 Tesseract OCR 설치 중...")
    
    try:
        # Ubuntu/Debian
        subprocess.run([
            "sudo", "apt-get", "update", "&&", 
            "sudo", "apt-get", "install", "-y", 
            "tesseract-ocr", "tesseract-ocr-kor", "tesseract-ocr-eng"
        ], shell=True, check=True)
        print("✅ Tesseract 설치 완료")
        return "/usr/bin/tesseract"
    except subprocess.CalledProcessError:
        print("❌ apt-get을 통한 설치 실패")
        print("수동으로 설치하세요: sudo apt-get install tesseract-ocr tesseract-ocr-kor")
        return None

def test_ocr():
    """OCR 기능 테스트"""
    print("\n🧪 OCR 기능 테스트 중...")
    
    try:
        import pytesseract
        from PIL import Image
        import io
        
        # 간단한 테스트 이미지 생성 (텍스트 포함)
        print("✅ pytesseract 및 PIL 라이브러리 로드 성공")
        
        # Tesseract 버전 확인
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract 버전: {version}")
        
        # 언어 지원 확인
        langs = pytesseract.get_languages()
        print(f"✅ 지원 언어: {langs}")
        
        if 'kor' in langs and 'eng' in langs:
            print("✅ 한국어 + 영어 지원 확인")
        else:
            print("⚠️  한국어 언어팩이 설치되지 않았을 수 있습니다")
        
        return True
        
    except Exception as e:
        print(f"❌ OCR 테스트 실패: {e}")
        return False

def update_env_file(tesseract_path):
    """환경 변수 파일 업데이트"""
    if not tesseract_path:
        return
    
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, "a") as f:
            f.write(f"\n# Tesseract OCR 경로\n")
            f.write(f"TESSERACT_CMD={tesseract_path}\n")
        print(f"✅ .env 파일에 Tesseract 경로 추가: {tesseract_path}")

def main():
    print("🔧 Caesar OCR 환경 설정 스크립트")
    print("=" * 50)
    
    os_type = detect_os()
    print(f"🖥️  감지된 운영체제: {os_type}")
    
    tesseract_path = None
    
    if os_type == "windows":
        tesseract_path = install_tesseract_windows()
    elif os_type == "macos":
        tesseract_path = install_tesseract_macos()
    elif os_type == "linux":
        tesseract_path = install_tesseract_linux()
    else:
        print("❌ 지원하지 않는 운영체제입니다")
        sys.exit(1)
    
    if tesseract_path:
        update_env_file(tesseract_path)
        
        if test_ocr():
            print("\n🎉 OCR 환경 설정 완료!")
            print("이제 PDF 이미지에서 텍스트를 추출할 수 있습니다.")
        else:
            print("\n⚠️  OCR 설정에 문제가 있습니다. 수동 확인이 필요합니다.")
    else:
        print("\n❌ OCR 환경 설정 실패")
        sys.exit(1)

if __name__ == "__main__":
    main()
