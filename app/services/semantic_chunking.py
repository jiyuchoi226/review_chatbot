from typing import List
from .tokenizer import TextTokenizer
from .embedding import TextEmbedding
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

class SemanticChunker:
    def __init__(
        self,
        tokenizer: TextTokenizer,
        max_chunk_size: int = 2000,  
        min_chunk_size: int = 200,
        overlap_size: int = 100,
        similarity_threshold: float = 0.8,
        batch_size: int = 50  # 배치 크기 추가
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.embedder = TextEmbedding()  

    def _merge_similar_sentences(self, sentences: List[str], embeddings: np.ndarray) -> List[List[str]]:
        if not sentences:
            return []

        token_counts = [self.tokenizer.count_tokens(sent) for sent in sentences]
        
        # 배치 단위로 유사도 계산
        sentence_groups = []
        current_group = []
        current_length = 0

        for i in range(len(sentences)):
            current_sent = sentences[i]
            current_tokens = token_counts[i]
            
            if not current_group:
                current_group.append(current_sent)
                current_length = current_tokens
            else:
                if i > 0:
                    prev_idx = i-1
                    # 현재 문장과 이전 문장의 임베딩만 사용하여 유사도 계산
                    similarity = cosine_similarity(
                        embeddings[prev_idx:prev_idx+1],
                        embeddings[i:i+1]
                    )[0][0]
                    
                    if similarity >= self.similarity_threshold and current_length + current_tokens <= self.max_chunk_size:
                        current_group.append(current_sent)
                        current_length += current_tokens
                    else:
                        if current_length < self.min_chunk_size and len(current_group) > 1:
                            last_sent = current_group.pop()
                            sentence_groups.append(current_group)
                            current_group = [last_sent, current_sent]
                            current_length = self.tokenizer.count_tokens(" ".join([last_sent, current_sent]))
                        else:
                            sentence_groups.append(current_group)
                            current_group = [current_sent]
                            current_length = current_tokens
        
        if current_group:
            sentence_groups.append(current_group)
            
        return sentence_groups

    def _optimize_chunk_size(self, sentence_groups: List[List[str]]) -> List[str]:
        chunks = []
        current_chunk = []
        current_length = 0
        
        for group in sentence_groups:
            group_text = " ".join(group)
            group_length = self.tokenizer.count_tokens(group_text)
            
            if current_length + group_length <= self.max_chunk_size:
                current_chunk.append(group_text)
                current_length += group_length
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [group_text]
                current_length = group_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def chunk_text(self, text: str) -> List[str]:
        # 문장 분리
        sentences = self.tokenizer.preprocess_and_split(text)
        if not sentences:
            return []
            
        # 배치 단위로 임베딩 생성
        all_embeddings = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]
            batch_embeddings = self.embedder.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            
        if not all_embeddings:
            return []
            
        # 임베딩을 numpy 배열로 변환
        embeddings = np.array(all_embeddings)
        
        # 문장 그룹화 및 청크 생성
        sentence_groups = self._merge_similar_sentences(sentences, embeddings)
        chunks = self._optimize_chunk_size(sentence_groups)
        return chunks

    def get_chunk_info(self, chunks: List[str]) -> List[dict]:
        chunk_info = []
        for i, chunk in enumerate(chunks):
            token_count = self.tokenizer.count_tokens(chunk)
            sentences = self.tokenizer.preprocess_and_split(chunk)
            
            info = {
                'chunk_id': i,
                'text': chunk,
                'token_count': token_count,
                'sentence_count': len(sentences),
                'overlap_prev': sentences[0] if i > 0 else "",
                'overlap_next': sentences[-1] if i < len(chunks)-1 else ""
            }
            chunk_info.append(info)
        return chunk_info