# rag_server.py (FastAPI 기반, 약어 확장 + 유연한 매칭 + 일반/매뉴얼 질문 제어)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import json, os, time

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 경로 설정
persist_dir = "./chroma_manual"
chunks_path = os.path.join(persist_dir, "chunks.jsonl")

# 임베딩 모델 초기화
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# 벡터 저장소 로딩
vector_db = Chroma(persist_directory=persist_dir, embedding_function=embedding)

# chunks 불러오기 및 dict 캐싱
with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = [json.loads(line.strip()) for line in f if line.strip()]
    chunk_dict = {c["content"].strip(): c for c in chunks}

# 약어 확장 맵
ABBREVIATION_MAP = {
    "hda": "highway driving assist",
    "acc": "adaptive cruise control",
    "lka": "lane keeping assist",
    "fca": "forward collision-avoidance assist",
    "scc": "smart cruise control",
    "nscc": "navigation-based smart cruise control",
    "bca": "blind-spot collision-avoidance assist",
    "rcca": "rear cross-traffic collision-avoidance assist",
    "daw": "driver attention warning",
    "isla": "intelligent speed limit assist",
    "msla": "manual speed limit assist",
    "rvm": "rear view monitor",
    "sew": "safe exit warning",
    "epb": "electronic parking brake",
    "esc": "electronic stability control",
    "vsm": "vehicle stability management",
    "abs": "anti-lock brake system",
    "hac": "hill-start assist control"
}

def expand_abbreviation(query: str) -> str:
    for abbr, full in ABBREVIATION_MAP.items():
        if abbr in query.lower():
            query = query.replace(abbr, full)
    return query

# 유연한 매칭 함수
def find_match(text):
    text = text.strip()
    for content, chunk in chunk_dict.items():
        if content in text:
            return chunk
    return None

# 요청 모델 정의
class RAGRequest(BaseModel):
    query: str
    k: int = 2
    alpha: float = 0.5

@app.post("/rag")
async def rag(request: RAGRequest):
    query = expand_abbreviation(request.query)
    k = request.k

    t_rag_start = time.time()
    results = vector_db.similarity_search_with_score(query, k=k)
    t_rag_end = time.time()
    rag_search_time = t_rag_end - t_rag_start

    t_prompt_start = time.time()
    selected_chunks, scores = [], []

    for doc, score in results:
        text = doc.page_content.strip()
        match = find_match(text)
        if match:
            selected_chunks.append(match)
            scores.append(score)

    if not selected_chunks or (scores and scores[0] < 0.3):
        # 일반 질문 제어용 system 프롬프트 추가
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant in my car. Answer clearly and concisely. Do not continue the conversation or ask your own questions."
            " Avoid repeating or speculating. End with a complete sentence.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        )
        return {
            "query": query,
            "prompt": prompt,
            "score": scores[0] if scores else 0.0,
            "context_count": 0,
            "results": [],
            "is_manual_based": False,
            "rag_search_time": rag_search_time,
            "prompt_gen_time": time.time() - t_prompt_start
        }

    context = "\n\n".join([f"Content: {c['content']}" for c in selected_chunks])
    prompt = (
        "<|im_start|>system\n"
        "Answer clearly and accurately based on the provided context. Keep the answer short and self-contained. End with a complete sentence.\n"
        "<|im_end|>\n"
        f"<|im_start|>user\nBased on this information: {context}\n\nQuestion: {query}\n"
        "<|im_start|>assistant\n"
    )

    return {
        "query": query,
        "prompt": prompt,
        "score": scores[0],
        "context_count": len(selected_chunks),
        "results": selected_chunks,
        "is_manual_based": True,
        "rag_search_time": rag_search_time,
        "prompt_gen_time": time.time() - t_prompt_start
    }

@app.get("/health")
async def health():
    return {"status": "ok", "chunks": len(chunks)}
