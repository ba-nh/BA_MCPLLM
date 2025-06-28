#!/bin/bash

# FastAPI 앱 실행 (최적화된 uvicorn 설정)
uvicorn rag_server:app --host 0.0.0.0 --port 5000 --workers 2 --timeout-keep-alive 5
