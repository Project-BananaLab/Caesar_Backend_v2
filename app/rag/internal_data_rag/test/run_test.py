#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 성능 테스트 실행 스크립트
"""

import sys
from pathlib import Path

# 상위 디렉토리를 Python 경로에 추가
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from performance_test import main

if __name__ == "__main__":
    main()
