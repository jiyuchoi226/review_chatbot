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
import json


class Chatbot:
    DEFAULT_RESPONSE = "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다."
    ERROR_RESPONSE = "죄송합니다. 답변을 생성하는 데 문제가 발생했습니다."

    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.tokenizer = TextTokenizer()
        self.chunker = SemanticChunker(self.tokenizer)
        self.embedder = TextEmbedding()
        faiss_base_path = os.getenv("FAISS_BASE_PATH", "/CHATBOT/data/review")
        self.review_data = VectorStore.load(faiss_base_path)
        self.conversation_history = ConversationHistory(
            embeddings=self.embedder,
            text_splitter=self.chunker
        )

        self.similarity_threshold = -0.5
        self.follow_up_threshold = 0.3
        self.context_window = 5


    # 코사인 유사도 계산
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        vec1_norm = np.linalg.norm(vec1)
        vec2_norm = np.linalg.norm(vec2)
        return 0.0 if vec1_norm == 0 or vec2_norm == 0 else np.dot(vec1, vec2) / (vec1_norm * vec2_norm)


    # 가장 유사한 문서 검색
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
            return None, 0.0



    # 이전 대화 맥락 
    def get_conversation_context(self, user_id: str, user_question: str) -> str:
        try:
            self.conversation_history.load_index(user_id)
            if not self.conversation_history.history_vectorstore:
                return ""

            relevant_docs = self.conversation_history.history_vectorstore.similarity_search(user_question, k=self.context_window)
            if not relevant_docs:
                return ""

            sorted_docs = sorted(
                relevant_docs,
                key=lambda x: x.metadata.get('timestamp', ''),
                reverse=True
            )
            context = "\n\n".join([doc.page_content for doc in sorted_docs])
            return context

        except Exception as e:
            return ""


    # 프롬프트
    def generate_prompt(self, context: str, user_input: str) -> str:
        return f"""
            당신은 스마트스토어 전문 챗봇입니다. 다음 지침을 따라 답변해주세요:

            1. 답변 스타일:
            - 사용자의 질문에 정확하고 상세한 답변을 제공하세요
            - 네이버 스마트스토어 관련 질문에 대해서만 답변해주세요
            - 전문적이면서도 친근한 톤으로 답변해주세요
            - 필요한 경우 단계별로 설명해주세요

            2. 답변 구조:
            - 먼저 사용자의 질문에 대한 직접적인 답변을 제공하세요
            - 답변이 길어질 경우 단락을 나누어 가독성을 높여주세요
            - 중요한 정보는 강조하여 표시해주세요
            - 필요한 경우 예시를 들어 설명해주세요

            3. 스마트스토어 관련성 판단:
            - 이전 대화 맥락과 현재 질문을 함께 고려하여 판단하세요
            - 이전 대화가 스마트스토어 관련이었다면, 현재 질문도 관련된 것으로 간주하세요
            - 현재 질문이 이전 대화의 맥락을 이어가는 경우, 스마트스토어 관련 질문으로 판단하세요
            - 예시:
              * 이전: "미성년자도 상품 등록할 수 있나요?"
              * 현재: "부모님의 도움이 뭐가 필요한가요?"
              → 이전 대화의 맥락을 이어가는 질문이므로 스마트스토어 관련 질문으로 판단
            - 이전의 대화를 참고하더라도 스마트스토어와 전혀 관련 없는 질문의 경우에만:
                "{self.DEFAULT_RESPONSE}"
                라고 답변하세요

            4. 답변 품질:
            - 정확한 정보만을 제공하세요
            - 불확실한 정보는 제공하지 마세요

            5. 대화 맥락 활용:
            - 이전 대화 내용을 반드시 참고하여 답변하세요
            - 사용자가 이전에 물어본 내용과 관련된 정보를 자연스럽게 연결하세요
            - 이전 답변에서 언급된 내용을 반복하지 않고, 새로운 정보를 추가하세요
            - 이전 대화에서 언급된 내용을 참고하여 더 구체적인 답변을 제공하세요
            - 대화의 맥락을 고려하여 적절한 수준의 상세도를 유지하세요

            예시)
            [이전 대화]
            유저: 미성년자도 상품 등록할 수 있나요?
            챗봇: 미성년자가 스마트스토어에 상품을 등록하는 것은 법적으로 제한됩니다. 
                  스마트스토어를 운영하기 위해서는 사업자 등록증이 필요하며, 
                  이를 발급받기 위해서는 만 19세 이상이어야 합니다. 
                  또한, 미성년자가 스마트스토어 운영을 위해서는 부모님의 도움이 필요합니다.

            [현재 대화]
            유저: 부모님의 도움이 무엇이 필요한가요?
            챗봇: 미성년자가 스마트스토어를 운영하기 위해서는 다음과 같은 부모님의 도움이 필요합니다:
            1. 법정 대리인 자격으로 사업자 등록증 발급
            2. 계좌 개설 및 관리
            3. 계약서 작성 및 서명
            4. 세금 신고 및 관리

            ---
            이전 대화 맥락:
            {context}

            ---
            현재 질문:
            {user_input}
        """

    # 대화 저장
    def save_conversation(self, user_id: str, user_question: str, bot_answer: str, follow_up_questions: List[str] = None) -> bool:
        try:
            print(f"대화 저장 시작 - 사용자: {user_id}")
            print(f"질문: {user_question}")
            print(f"답변: {bot_answer}")
            if follow_up_questions:
                print(f"후속 질문: {follow_up_questions}")
            
            # 후속 질문이 있는 경우 답변에 추가
            full_answer = bot_answer
            if follow_up_questions:
                full_answer = f"{bot_answer}\n\n" + "\n".join(follow_up_questions)
            
            # 대화 저장
            self.conversation_history.add_conversation(
                user_id=user_id,
                user_question=user_question,
                bot_answer=full_answer
            )
            
            # 저장 확인
            self.conversation_history.load_index(user_id)
            if self.conversation_history.history_vectorstore:
                return True
            else:
                return False

        except Exception as e:
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            return False


    # 사용자 쿼리 재조정
    def adjust_user_query(self, user_input: str, context: str) -> str:
        try:
            if not context:
                return user_input

            prompt = f"""
            이전 대화 맥락과 현재 질문을 바탕으로, 사용자의 의도를 더 명확하게 파악하여 재조정된 질문을 생성해주세요.

            [규칙]
            1. 이전 대화 맥락을 고려하여 현재 질문의 맥락을 파악하세요
            2. 현재 질문이 이전 대화의 맥락을 이어가는 경우, 그 맥락을 반영하여 질문을 재조정하세요
            3. 현재 질문이 이전 대화와 관련이 없는 경우, 원래 질문을 그대로 유지하세요
            4. 재조정된 질문은 스마트스토어 FAQ 맥락에서 이해할 수 있도록 해주세요

            [예시]
            이전 대화:
            유저: 미성년자도 상품 등록할 수 있나요?
            챗봇: 미성년자가 스마트스토어에 상품을 등록하는 것은 법적으로 제한됩니다...

            현재 질문: 부모님의 도움이 무엇이 필요한가요?
            재조정된 질문: 미성년자가 스마트스토어를 운영하기 위해 필요한 부모님의 법적 도움이 무엇인가요?

            ---
            이전 대화 맥락:
            {context}

            ---
            현재 질문:
            {user_input}

            ---
            재조정된 질문을 생성해주세요:
            """

            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                n=1,
                temperature=0.3
            )
            
            adjusted_query = response.choices[0].message.content.strip()
            print(f"원본 질문: {user_input}")
            print(f"재조정된 질문: {adjusted_query}")
            return adjusted_query

        except Exception as e:
            print(f"❌ 쿼리 재조정 중 오류 발생: {str(e)}")
            return user_input

    # 답변 생성
    def generate_answer_with_similarity(self, user_input: str, user_id: str):
        try:
            context = self.get_conversation_context(user_id, user_input)
            adjusted_query = self.adjust_user_query(user_input, context)
            prompt = self.generate_prompt(context, adjusted_query)
            
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                n=1,
                temperature=0.7,
                stream=True
            )

            # 스트리밍 응답 처리
            answer = ""
            collected_chunk = ""  # 임시로 청크를 모으는 변수
            
            for chunk in response:
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    answer += content
                    collected_chunk += content
                    
                    # 완성된 단어나 문장 단위로 전송
                    if any(c in content for c in [" ", ".", ",", "!", "?", "\n"]) or len(collected_chunk) >= 10:
                        yield collected_chunk
                        collected_chunk = ""

            # 남은 청크가 있다면 전송
            if collected_chunk:
                yield collected_chunk

            # 후속 질문 생성
            follow_up_questions = self.suggest_follow_up_questions(user_input, answer)
            self.save_conversation(user_id, user_input, answer, follow_up_questions)

            # 후속 질문을 한 글자씩 스트리밍
            if follow_up_questions:
                yield "\n\n"
                
                for question in follow_up_questions:
                    # 각 글자를 개별적으로 전송
                    for char in question:
                        yield char
                    yield "\n"

        except Exception as e:
            print(f"❌ 답변 생성 중 오류 발생: {str(e)}")
            yield self.ERROR_RESPONSE

    # 키워드 추출
    def extract_keywords(self, text: str) -> List[str]:
        try:
            prompt = f"""
            다음 텍스트에서 가장 중요한 키워드 2-3개를 추출해주세요.
            추출 규칙:
            1. 질문에서 가장 중요한 키워드를 추출
            2. 키워드는 쉼표로 구분하여 나열
            3. 키워드는 한 문장으로 작성해주세요.

            텍스트: {text}
            """
            
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                n=1,
                temperature=0.3
            )
            
            keywords = response.choices[0].message.content.strip().split(',')
            return [k.strip() for k in keywords if k.strip()]
            
        except Exception as e:
            print(f"❌ 키워드 추출 중 오류 발생: {str(e)}")
            return []


    # 후속 질문문
    def suggest_follow_up_questions(self, user_input: str, answer: str) -> List[str]:
        try:
            keywords = self.extract_keywords(user_input)
            if not keywords:
                return []
            print(f"추출된 키워드: {keywords}")

            # 키워드로 관련 문서 검색
            related_docs = []
            for keyword in keywords:
                docs = self.review_data.similarity_search(keyword, k=3)
                related_docs.extend(docs)
            
            if not related_docs:
                return []

            # 중복 제거 및 유사도 기준 정렬
            unique_docs = {}
            for doc in related_docs:
                if doc['text'] not in unique_docs:
                    unique_docs[doc['text']] = doc
            
            sorted_docs = sorted(
                unique_docs.values(),
                key=lambda x: x['distance'],
                reverse=True
            )[:2]  
            
            # 후속 질문 생성
            follow_up_prompt = f"""
            사용자가 '{user_input}'에 대해 질문했습니다.
            추출된 키워드는 {', '.join(keywords)}입니다.
            
            다음 문서들을 참고하여 관련된 후속 질문을 생성해주세요:
            {[doc['text'] for doc in sorted_docs]}
            
            다음 지침을 따라주세요:

            1. 스마트스토어 FAQ 관련 질문인 경우:
               - 사용자의 질문과 직접적으로 관련된 후속 질문을 생성하세요
               - FAQ 데이터 내의 관련 질문을 참고하세요

            2. 스마트스토어 FAQ와 관련 없는 질문인 경우:
               - 사용자의 질문에서 추출된 키워드를 스마트스토어 맥락으로 변환하세요
               - 예시:
                 * "맛집 추천" → "스토어 등록", "판매 상품"
                 * "날씨" → "배송", "상품 보관"
                 * "운동" → "스포츠용품 판매", "운동복 스토어"
               - 변환된 키워드를 바탕으로 스마트스토어 관련 후속 질문을 생성하세요
               - FAQ 데이터 내의 가장 관련성 높은 질문을 참고하세요

            [공통 지침]
            1. 후속 질문은 1-2개로 생성해주세요
            2. 각 질문은 한 문장으로 작성해주세요
            3. 모든 후속 질문은 반드시 스마트스토어와 관련되어야 합니다
            4. 질문은 FAQ 데이터 내의 질문 형식을 따르세요

            예시1)
            - 음식점 스토어 등록이 가능한지 궁금하신가요?
            - 식품 판매를 위한 스토어 등록 절차가 궁금하신가요?

            예시2)
            - 우산 판매 스토어 등록이 궁금하신가요?
            - 비옷 판매를 위한 스토어 등록 절차가 궁금하신가요?
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
            return []