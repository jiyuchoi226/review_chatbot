import os
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional, Tuple
import openai
import numpy as np
from dotenv import load_dotenv
import faiss
from app.models.history import ConversationHistory
from app.services.embedding import TextEmbedding
from app.services.vectorstore import VectorStore
from app.services.tokenizer import TextTokenizer
from app.services.semantic_chunking import SemanticChunker


class Chatbot:
    DEFAULT_RESPONSE = "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다."
    ERROR_RESPONSE = "죄송합니다. 답변을 생성하는 데 문제가 발생했습니다."
    DEFAULT_FOLLOW_UP = "- 더 자세한 설명이 필요하신가요?"

    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # 서비스 초기화
        self.tokenizer = TextTokenizer()
        self.chunker = SemanticChunker(self.tokenizer)
        self.embedder = TextEmbedding()
        
        # VectorStore 초기화
        faiss_base_path = os.getenv("FAISS_BASE_PATH", "/CHATBOT/data/review")
        self.review_data = VectorStore.load(faiss_base_path)
        
        # ConversationHistory 초기화
        self.conversation_history = ConversationHistory(
            embeddings=self.embedder,
            text_splitter=self.chunker
        )
        
        # 설정값
        self.similarity_threshold = -0.5
        self.follow_up_threshold = 0.3
        self.context_window = 4

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        vec1_norm = np.linalg.norm(vec1)
        vec2_norm = np.linalg.norm(vec2)
        return 0.0 if vec1_norm == 0 or vec2_norm == 0 else np.dot(vec1, vec2) / (vec1_norm * vec2_norm)

    def get_best_matching_doc(self, query: str, top_k: int = 1) -> Tuple[Optional[Dict], float]:
        try:
            query_vector = self.embedder.embed_documents(query)
            result_docs = self.review_data.similarity_search(query, k=top_k)
            
            if not result_docs:
                return None, 0.0

            best_doc = result_docs[0]
            doc_vector = self.embedder.embed_documents(best_doc['text'])
            similarity = self.cosine_similarity(query_vector, doc_vector)
            
            print(f"문서: {best_doc['text'][:150]}... 유사도: {similarity:.4f}")
            return best_doc, similarity

        except Exception as e:
            print(f"❌ 문서 검색 중 오류 발생: {str(e)}")
            return None, 0.0

    def get_conversation_context(self, user_id: str, user_question: str) -> str:
        try:
            self.conversation_history.load_index(user_id)
            if not self.conversation_history.history_vectorstore:
                return ""

            relevant_docs = self.conversation_history.history_vectorstore.similarity_search(
                user_question, k=self.context_window
            )
            
            if not relevant_docs:
                return ""

            sorted_docs = sorted(
                relevant_docs,
                key=lambda x: x.metadata.get('timestamp', ''),
                reverse=True
            )
            context = "\n\n".join([doc.page_content for doc in sorted_docs])
            print(f"이전 대화 맥락: {context[:150]}...")
            return context

        except Exception as e:
            print(f"❌ 대화 맥락 로드 중 오류 발생: {str(e)}")
            return ""

    def generate_prompt(self, context: str) -> str:
        return f"""
            당신은 스마트스토어 전문 챗봇입니다. 다음 지침을 따라 답변해주세요:

            1. 답변 스타일:
            - 사용자의 질문에 정확한 답변을 제공하세요
            - 전문적이고 명확한 설명을 해주세요

            2. 답변 구조:
            - 먼저 사용자의 질문에 대한 직접적인 답변을 제공하세요
            - 답변 후에는 관련된 후속 질문을 1-2개 제시하세요.

            3. 스마트스토어 관련성:
            - 스마트스토어와 관련 없는 질문의 경우:
                "{self.DEFAULT_RESPONSE}"
                라고 답변하세요

            4. 맥락 활용:
            - 제공된 컨텍스트를 적극적으로 활용하세요
            - 이전 대화 내용을 고려하여 일관성 있는 답변을 제공하세요
            
            ---
            CONTEXT:
            {context}
        """

    def save_conversation(self, user_id: str, user_question: str, bot_answer: str) -> bool:
        try:
            print(f"대화 저장 시작 - 사용자: {user_id}")
            print(f"질문: {user_question}")
            print(f"답변: {bot_answer}")
            
            # 대화 저장
            self.conversation_history.add_conversation(
                user_id=user_id,
                user_question=user_question,
                bot_answer=bot_answer
            )
            
            # 저장 확인
            self.conversation_history.load_index(user_id)
            if self.conversation_history.history_vectorstore:
                print("✅ 대화 기록 저장 완료")
                return True
            else:
                print("⚠️ 대화 기록 저장 실패")
                return False

        except Exception as e:
            print(f"❌ 대화 저장 중 오류 발생: {str(e)}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            return False

    def generate_answer_with_similarity(self, user_input: str, user_id: str) -> str:
        try:
            # 프롬프트 생성 및 답변 생성
            prompt = self.generate_prompt(user_input)
            
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                n=1,
                temperature=0.7,
                stream=True
            )

            # 스트리밍 응답 처리
            answer = ""
            for chunk in response:
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    answer += content
                    print(content, end='', flush=True)
            print()

            # 대화 저장
            self.save_conversation(user_id, user_input, answer)

            # 후속 질문 생성
            follow_up_questions = self.suggest_follow_up_questions(user_input, answer)
            return answer.strip() + "\n" + "\n".join(follow_up_questions)

        except Exception as e:
            print(f"❌ 답변 생성 중 오류 발생: {str(e)}")
            return self.ERROR_RESPONSE

    def suggest_follow_up_questions(self, user_input: str, answer: str) -> List[str]:
        try:
            combined_text = f"{user_input} {answer}"
            best_doc, best_similarity = self.get_best_matching_doc(combined_text)
            
            if not best_doc or best_similarity <= self.follow_up_threshold:
                return [self.DEFAULT_FOLLOW_UP]

            follow_up_prompt = f"""
            사용자가 '{best_doc['text']}'에 대해 질문했습니다. 
            이 질문에 대한 후속 질문을 생성해 주세요. 
            다음 지침을 따라주세요:
            1. 사용자가 더 깊이 이해할 수 있도록 돕는 내용이어야 합니다.
            2. 현재 대화 맥락과 자연스럽게 연결되어야 합니다.
            3. 1-2개의 구체적인 질문을 생성해주세요.
            4. 각 질문은 한 문장으로 작성해주세요.
            5. 스마트스토어와 관련된 질문만 생성해주세요.
            """

            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": follow_up_prompt}],
                max_tokens=1000,
                n=1,
                temperature=0.7
            )
            
            follow_up_question = response.choices[0].message.content.strip()
            return [f"- {follow_up_question}"]

        except Exception as e:
            print(f"❌ 후속 질문 생성 중 오류 발생: {str(e)}")
            return [self.DEFAULT_FOLLOW_UP]