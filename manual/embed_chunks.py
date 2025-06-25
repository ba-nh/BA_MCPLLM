import json
import numpy as np
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import pickle
from typing import List, Dict, Any

class ChunkEmbedder:
    def __init__(self, model_name: str = 'paraphrase-MiniLM-L6-v2'):
        """임베딩 모델을 초기화합니다."""
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"모델 로드 완료: {model_name}")
        print(f"임베딩 차원: {self.embedding_dim}")
    
    def load_chunks(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """JSONL 파일에서 chunk를 로드합니다."""
        chunks = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line.strip()))
        return chunks
    
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """chunk 텍스트를 임베딩합니다."""
        texts = []
        for chunk in chunks:
            # 제목과 내용을 결합하여 임베딩
            text = f"{chunk['title']} {chunk['content']}".strip()
            texts.append(text)
        
        print(f"총 {len(texts)}개의 텍스트 임베딩 생성 중...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def build_annoy_index(self, embeddings: np.ndarray, n_trees: int = 100) -> AnnoyIndex:
        """Annoy 인덱스를 구축합니다."""
        print("Annoy 인덱스 구축 중...")
        annoy_index = AnnoyIndex(self.embedding_dim, 'angular')
        
        for i, embedding in enumerate(embeddings):
            annoy_index.add_item(i, embedding)
        
        annoy_index.build(n_trees)
        print(f"Annoy 인덱스 구축 완료 (trees: {n_trees})")
        return annoy_index
    
    def save_chunk_metadata(self, chunks: List[Dict[str, Any]], output_path: str):
        """chunk 메타데이터를 저장합니다."""
        metadata = []
        for i, chunk in enumerate(chunks):
            meta = {
                'id': i,
                'chunk_id': chunk.get('id', i),
                'page': chunk.get('page', 0),
                'title': chunk.get('title', ''),
                'content': chunk.get('content', ''),
                'font_size': chunk.get('font_size', 0)
            }
            metadata.append(meta)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for meta in metadata:
                f.write(json.dumps(meta, ensure_ascii=False) + '\n')
        
        print(f"메타데이터 저장 완료: {output_path}")
    
    def save_annoy_index(self, annoy_index: AnnoyIndex, output_path: str):
        """Annoy 인덱스를 저장합니다."""
        annoy_index.save(output_path)
        print(f"Annoy 인덱스 저장 완료: {output_path}")
    
    def test_search(self, annoy_index: AnnoyIndex, chunks: List[Dict[str, Any]], 
                   query: str = "engine oil", k: int = 5):
        """검색 테스트를 수행합니다."""
        print(f"\n=== 검색 테스트: '{query}' ===")
        query_embedding = self.model.encode([query])
        indices, distances = annoy_index.get_nns_by_vector(
            query_embedding[0], k, include_distances=True
        )
        
        for i, (idx, distance) in enumerate(zip(indices, distances)):
            chunk = chunks[idx]
            print(f"{i+1}. Chunk {idx} (거리: {distance:.4f})")
            print(f"   제목: {chunk['title']}")
            print(f"   내용: {chunk['content'][:100]}...")
            print(f"   페이지: {chunk['page']}")
            print()

def main():
    # 설정
    jsonl_path = "Manual_EN_chunks.jsonl"
    vector_db_path = "vector_db.ann"
    metadata_path = "chunk_meta.jsonl"
    
    # 임베더 초기화
    embedder = ChunkEmbedder()
    
    # chunk 로드
    print("Chunk 로드 중...")
    chunks = embedder.load_chunks(jsonl_path)
    print(f"총 {len(chunks)}개의 chunk 로드 완료")
    
    # 임베딩 생성
    embeddings = embedder.create_embeddings(chunks)
    
    # Annoy 인덱스 구축
    annoy_index = embedder.build_annoy_index(embeddings)
    
    # 파일 저장
    embedder.save_annoy_index(annoy_index, vector_db_path)
    embedder.save_chunk_metadata(chunks, metadata_path)
    
    # 검색 테스트
    embedder.test_search(annoy_index, chunks, "engine oil")
    embedder.test_search(annoy_index, chunks, "brake system")
    
    print("임베딩 및 벡터DB 구축 완료!")

if __name__ == "__main__":
    main() 