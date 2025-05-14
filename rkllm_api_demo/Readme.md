# 컴파일 방법

```bash
bash build-linux.sh
```
실행 시 build 폴더에 output 생성

build 폴더에 make 파일까지 생성시키고 make까지 실행함

->  llm_demo와 multimodel_demo 생성됨

## 실행 방법
./llm_demo


## 라이브러리 위치
linux/librkllm_api/include/rkllm.h

linux/librkllm_api/arm64-v8a/librkllmrt.so

# Translation 폴더
아래 프로세스 중 2, 4번 번역을 위함

1. 한국어 입력
   
3. 입력된 한국어를 영어로 번역
   
5. 영어 입력을 LLM에 입력 -> 영어로 출력
   
7. 출력된 영어를 한국어로 다시 번역
