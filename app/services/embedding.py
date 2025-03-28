from typing import List
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class TextEmbedding:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model

    def embed_documents(self, texts: str | List[str]) -> np.ndarray:
        try:
            if isinstance(texts, str):
                texts = [texts]
                
            response = client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
            
        except Exception as e:
            print(f"임베딩 생성 중 오류 발생: {str(e)}")
            return np.array([])
