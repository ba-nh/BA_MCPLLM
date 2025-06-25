import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import time

class HybridSearchEngine:
    def __init__(self, vector_db_path: str, bm25_path: str, metadata_path: str):
        """하이브리드 검색 엔진을 초기화합니다."""
        # 벡터 검색 초기화
        print("벡터 검색 모델 로드 중...")
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Annoy 인덱스 로드
        print("Annoy 인덱스 로드 중...")
        self.annoy_index = AnnoyIndex(self.embedding_dim, 'angular')
        self.annoy_index.load(vector_db_path)
        
        # BM25 인덱스 로드
        print("BM25 인덱스 로드 중...")
        with open(bm25_path, 'rb') as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data['bm25']
        self.chunks = bm25_data['chunks']
        
        # 메타데이터 로드
        print("메타데이터 로드 중...")
        self.metadata = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.metadata.append(json.loads(line.strip()))
        
        print(f"하이브리드 검색 엔진 초기화 완료 (총 {len(self.chunks)}개 문서)")
    
    def preprocess_text(self, text: str) -> List[str]:
        """텍스트를 전처리하고 토큰화합니다."""
        text = text.lower()
        text = re.sub(r'[^\w\s-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split()
        return tokens
    
    def vector_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """벡터 검색을 수행합니다."""
        query_embedding = self.model.encode([query])
        indices, distances = self.annoy_index.get_nns_by_vector(
            query_embedding[0], k, include_distances=True
        )
        
        # 거리를 유사도로 변환 (1 - 거리)
        similarities = [1 - dist for dist in distances]
        return list(zip(indices, similarities))
    
    def bm25_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """BM25 검색을 수행합니다."""
        query_tokens = self.preprocess_text(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # 상위 k개 결과 반환
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = [(idx, scores[idx]) for idx in top_indices]
        return results
    
    def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """하이브리드 검색을 수행합니다."""
        # 벡터 검색과 BM25 검색 수행
        vector_results = self.vector_search(query, k * 2)
        bm25_results = self.bm25_search(query, k * 2)
        
        # 결과 통합
        combined_scores = {}
        
        # 벡터 검색 결과 추가
        for idx, score in vector_results:
            combined_scores[idx] = {'vector_score': score, 'bm25_score': 0.0}
        
        # BM25 검색 결과 추가
        for idx, score in bm25_results:
            if idx in combined_scores:
                combined_scores[idx]['bm25_score'] = score
            else:
                combined_scores[idx] = {'vector_score': 0.0, 'bm25_score': score}
        
        # 정규화 및 결합
        max_vector_score = max([s['vector_score'] for s in combined_scores.values()]) if combined_scores else 1.0
        max_bm25_score = max([s['bm25_score'] for s in combined_scores.values()]) if combined_scores else 1.0
        
        # 정규화 및 가중 평균 계산
        final_results = []
        for idx, scores in combined_scores.items():
            if max_vector_score > 0:
                norm_vector_score = scores['vector_score'] / max_vector_score
            else:
                norm_vector_score = 0.0
            
            if max_bm25_score > 0:
                norm_bm25_score = scores['bm25_score'] / max_bm25_score
            else:
                norm_bm25_score = 0.0
            
            # 가중 평균
            final_score = alpha * norm_vector_score + (1 - alpha) * norm_bm25_score
            
            final_results.append({
                'chunk_id': idx,
                'final_score': final_score,
                'vector_score': norm_vector_score,
                'bm25_score': norm_bm25_score,
                'title': self.chunks[idx]['title'],
                'content': self.chunks[idx]['content'],
                'page': self.chunks[idx]['page'],
                'font_size': self.chunks[idx]['font_size']
            })
        
        # 최종 점수로 정렬
        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return final_results[:k]

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)

# 전역 검색 엔진 변수
search_engine = None

@app.route('/search', methods=['POST'])
def search():
    """검색 API 엔드포인트"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        k = data.get('k', 10)
        alpha = data.get('alpha', 0.5)  # 벡터 검색 가중치
        search_type = data.get('type', 'hybrid')  # 'vector', 'bm25', 'hybrid'
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        if search_type == 'vector':
            results = search_engine.vector_search(query, k)
            formatted_results = []
            for idx, score in results:
                chunk = search_engine.chunks[idx]
                formatted_results.append({
                    'chunk_id': idx,
                    'score': score,
                    'title': chunk['title'],
                    'content': chunk['content'],
                    'page': chunk['page'],
                    'font_size': chunk['font_size']
                })
        elif search_type == 'bm25':
            results = search_engine.bm25_search(query, k)
            formatted_results = []
            for idx, score in results:
                chunk = search_engine.chunks[idx]
                formatted_results.append({
                    'chunk_id': idx,
                    'score': score,
                    'title': chunk['title'],
                    'content': chunk['content'],
                    'page': chunk['page'],
                    'font_size': chunk['font_size']
                })
        else:  # hybrid
            formatted_results = search_engine.hybrid_search(query, k, alpha)
        
        return jsonify({
            'query': query,
            'type': search_type,
            'results': formatted_results,
            'total_results': len(formatted_results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """헬스 체크 엔드포인트"""
    return jsonify({'status': 'healthy', 'total_chunks': len(search_engine.chunks)})

@app.route('/info', methods=['GET'])
def info():
    """서버 정보 엔드포인트"""
    return jsonify({
        'name': 'Hybrid Search Server',
        'total_chunks': len(search_engine.chunks),
        'embedding_dim': search_engine.embedding_dim,
        'supported_search_types': ['vector', 'bm25', 'hybrid']
    })

@app.route('/rag', methods=['POST'])
def rag():
    try:
        data = request.get_json()
        query = data.get('query', '')
        k = data.get('k', 5)
        alpha = data.get('alpha', 0.5)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Hybrid search
        search_results = search_engine.hybrid_search(query, k, alpha)
        
        # If no results
        if not search_results:
            prompt = (
                "<|im_start|>system\n"
                "You are an in-car assistant who answers questions strictly based on the manual of the car the user is driving. "
                "Summarize your answer in 1-2 sentences. If your answer is based on the manual, clearly state that it is according to the manual. "
                "If the answer is not found in the manual, honestly say you do not know.\n"
                "<|im_end|>\n"
                f"<|im_start|>user\n{query}\n<|im_end|>\n"
                "<|im_start|>assistant\nI do not know.\n<|im_end|>"
            )
            return jsonify({
                'query': query,
                'prompt': prompt,
                'results': [],
                'context_length': 0
            })
        
        # Build context from top 3 results
        context_parts = []
        for i, result in enumerate(search_results[:3]):
            context_parts.append(f"Title: {result['title']}\nContent: {result['content']}")
        context = "\n\n".join(context_parts)
        
        # Qwen-style prompt
        system_prompt = (
        "You are a strict in-car assistant. Your only knowledge source is the car's official manual. "
        "Do not use your own knowledge. Always say 'According to the manual...' if the answer is found. "
        "If not found in the manual, say 'I do not know based on the manual.' "
        "Summarize your answer in 1-2 sentences."
        )   

        rag_prompt = (
            f"<|im_start|>system\n{system_prompt}\n\nManual Excerpt:\n{context}\n<|im_end|>\n"
            f"<|im_start|>user\n{query}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        return jsonify({
            'query': query,
            'prompt': rag_prompt,
            'results': search_results,
            'context_length': len(context)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def main():
    global search_engine
    
    # 검색 엔진 초기화
    print("하이브리드 검색 엔진 초기화 중...")
    search_engine = HybridSearchEngine(
        vector_db_path="vector_db.ann",
        bm25_path="bm25_index.pkl",
        metadata_path="chunk_meta.jsonl"
    )
    
    # Flask 서버 시작
    print("검색 서버 시작 중...")
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main() 