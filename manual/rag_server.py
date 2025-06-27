# ✅ rag_server.py (Odroid 최적화 + TinyLlama 전용 프롬프트 + relevance 판단 포함)

from flask import Flask, request, jsonify
from flask_cors import CORS
import json, os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

app = Flask(__name__)
CORS(app)

# 📁 경로 설정
persist_dir = "./chroma_manual"
chunks_path = os.path.join(persist_dir, "chunks.jsonl")

# ✅ bge-small 모델 로딩
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# ✅ Chroma DB 로딩
vector_db = Chroma(persist_directory=persist_dir, embedding_function=embedding)

# ✅ chunk 메타 정보 로드
with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = [json.loads(line.strip()) for line in f if line.strip()]

@app.route("/rag", methods=["POST"])
def rag():
    data = request.get_json()
    query = data.get("query", "")
    k = int(data.get("k", 3))
    alpha = float(data.get("alpha", 0.5))

    if not query:
        return jsonify({"error": "Query is required"}), 400

    results = vector_db.similarity_search_with_score(query, k=k)
    selected_chunks = []
    scores = []

    for doc, score in results:
        text = doc.page_content.strip()
        match = next((c for c in chunks if c["content"] in text), None)
        if match:
            selected_chunks.append(match)
            scores.append(score)

    if not selected_chunks:
        # 연관된 청크가 없을 때 일반 지식으로 답변
        prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        return jsonify({
            "query": query,
            "prompt": prompt,
            "score": 0.0,
            "context_count": 0,
            "results": [],
            "is_manual_based": False
        })

    top_score = scores[0] if scores else 0.0

    if top_score < 0.3:
        prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        is_manual_based = False
    else:
        context = "\n\n".join([f"Content: {c['content']}" for c in selected_chunks])
        prompt = f"""<|im_start|>system
You are a car manual assistant. Based on the provided information, give a clear and accurate answer. Do not ask follow-up questions or continue the conversation.
<|im_end|>
<|im_start|>user
Based on this information: {context}

Question: {query}
<|im_start|>assistant
"""
        is_manual_based = True

    return jsonify({
        "query": query,
        "prompt": prompt,
        "score": top_score,
        "context_count": len(selected_chunks),
        "results": selected_chunks,
        "is_manual_based": is_manual_based
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok", "chunks": len(chunks)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)