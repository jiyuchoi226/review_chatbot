from typing import List, Optional, Dict, Any
import numpy as np
import faiss
import pickle
from pathlib import Path
from dotenv import load_dotenv
import os
import re
from app.services.embedding import TextEmbedding

# .env 파일 로드
load_dotenv()

class VectorStore:
    def __init__(self, dimension: int = 1536):
        
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.embedder = TextEmbedding()
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        embeddings = self.embedder.embed_documents(texts)
        if len(embeddings) == 0:
            return
            
        self.index.add(embeddings.astype(np.float32))
        self.texts.extend(texts)
        
        if metadatas:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{} for _ in texts])
            
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        
        query_embedding = self.embedder.embed_documents(query)
        if len(query_embedding) == 0:
            return []
            
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                result = {
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx],
                    'distance': float(distances[0][i])
                }
                results.append(result)
                
        return results
        
    def save(self, directory: str) -> None:
        save_path = Path(directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # FAISS 인덱스 저장
        faiss.write_index(self.index, str(save_path / "review.faiss"))
        
        # 텍스트와 메타데이터 저장
        with open(save_path / "review.pkl", "wb") as f:
            pickle.dump({
                'texts': self.texts,
                'metadata': self.metadata,
                'dimension': self.dimension
            }, f)
            
    @classmethod
    def load(cls, directory: str) -> 'VectorStore':
        load_path = Path(directory)
        with open(load_path / "review.pkl", "rb") as f:
            data = pickle.load(f)
        instance = cls(dimension=data['dimension'])
        instance.texts = data['texts']
        instance.metadata = data['metadata']
        instance.index = faiss.read_index(str(load_path / "review.faiss"))
        
        return instance
        
    def extract_category(self, question: str) -> str:
        category_match = re.match(r'\[(.*?)\]', question)
        return category_match.group(1) if category_match else "일반"
        
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        
        # 대괄호로 둘러싸인 카테고리 제거
        text = re.sub(r'\[.*?\]', '', text)
        
        # 불필요한 문구 제거
        removals = [
            "위 도움말이 도움이 되었나요?",
            "별점",
            "관련 도움말/키워드"
        ]
        
        for phrase in removals:
            if phrase in text:
                text = text.split(phrase)[0]
        
        return text.strip()
        
    def load_faq_data(self, faq_path: str) -> None:
        try:
            with open(faq_path, 'rb') as f:
                faq_data = pickle.load(f)
                
            texts = []
            metadatas = []
            
            if isinstance(faq_data, dict):
                for question, answer in faq_data.items():
                    if isinstance(answer, str):
                        category = self.extract_category(question)
                        clean_question = self.clean_text(question)
                        clean_answer = self.clean_text(answer)
                        
                        if clean_question and clean_answer:
                            texts.append(clean_question)
                            metadatas.append({
                                'answer': clean_answer,
                                'category': category,
                                'type': 'faq'
                            })
            
            if texts:
                print(f"FAQ 데이터 로드 완료: {len(texts)}개 항목")
                self.add_texts(texts, metadatas)
            else:
                print("경고: 로드할 FAQ 데이터가 없습니다.")
                
        except Exception as e:
            print(f"FAQ 데이터 로드 중 오류 발생: {str(e)}")
