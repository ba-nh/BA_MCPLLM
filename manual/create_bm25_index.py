import json
import pickle
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import re

class BM25Indexer:
    def __init__(self):
        """BM25 인덱서를 초기화합니다."""
        self.bm25 = None
        self.chunks = []
        self.tokenized_chunks = []
    
    def load_chunks(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """JSONL 파일에서 chunk를 로드합니다."""
        chunks = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line.strip()))
        return chunks
    
    def preprocess_text(self, text: str) -> List[str]:
        """텍스트를 전처리하고 토큰화합니다."""
        # 소문자 변환
        text = text.lower()
        # 특수문자 제거 (하이픈과 공백은 유지)
        text = re.sub(r'[^\w\s-]', ' ', text)
        # 연속된 공백을 단일 공백으로 변환
        text = re.sub(r'\s+', ' ', text).strip()
        # 토큰화
        tokens = text.split()
        return tokens
    
    def create_bm25_index(self, chunks: List[Dict[str, Any]]) -> BM25Okapi:
        """BM25 인덱스를 생성합니다."""
        self.chunks = chunks
        self.tokenized_chunks = []
        
        print("텍스트 전처리 및 토큰화 중...")
        for chunk in chunks:
            # 제목과 내용을 결합
            text = f"{chunk['title']} {chunk['content']}".strip()
            tokens = self.preprocess_text(text)
            self.tokenized_chunks.append(tokens)
        
        print("BM25 인덱스 생성 중...")
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        print(f"BM25 인덱스 생성 완료 (총 {len(chunks)}개 문서)")
        
        return self.bm25
    
    def search(self, query: str, k: int = 5) -> List[tuple]:
        """BM25 검색을 수행합니다."""
        if self.bm25 is None:
            raise ValueError("BM25 인덱스가 생성되지 않았습니다.")
        
        # 쿼리 전처리
        query_tokens = self.preprocess_text(query)
        
        # 검색 수행
        scores = self.bm25.get_scores(query_tokens)
        
        # 상위 k개 결과 반환
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = []
        for idx in top_indices:
            results.append((idx, scores[idx]))
        
        return results
    
    def save_bm25_index(self, output_path: str):
        """BM25 인덱스를 저장합니다."""
        if self.bm25 is None:
            raise ValueError("BM25 인덱스가 생성되지 않았습니다.")
        
        # BM25 인덱스와 메타데이터를 함께 저장
        data = {
            'bm25': self.bm25,
            'chunks': self.chunks,
            'tokenized_chunks': self.tokenized_chunks
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"BM25 인덱스 저장 완료: {output_path}")
    
    def test_search(self, query: str = "engine oil", k: int = 5):
        """검색 테스트를 수행합니다."""
        print(f"\n=== BM25 검색 테스트: '{query}' ===")
        results = self.search(query, k)
        
        for i, (idx, score) in enumerate(results):
            chunk = self.chunks[idx]
            print(f"{i+1}. Chunk {idx} (점수: {score:.4f})")
            print(f"   제목: {chunk['title']}")
            print(f"   내용: {chunk['content'][:100]}...")
            print(f"   페이지: {chunk['page']}")
            print()

def main():
    # 설정
    jsonl_path = "Manual_EN_chunks.jsonl"
    bm25_path = "bm25_index.pkl"
    
    # BM25 인덱서 초기화
    indexer = BM25Indexer()
    
    # chunk 로드
    print("Chunk 로드 중...")
    chunks = indexer.load_chunks(jsonl_path)
    print(f"총 {len(chunks)}개의 chunk 로드 완료")
    
    # BM25 인덱스 생성
    bm25_index = indexer.create_bm25_index(chunks)
    
    # BM25 인덱스 저장
    indexer.save_bm25_index(bm25_path)
    
    # 검색 테스트
    indexer.test_search("engine oil")
    indexer.test_search("brake system")
    indexer.test_search("tire pressure")
    
    print("BM25 인덱스 구축 완료!")

if __name__ == "__main__":
    main() 