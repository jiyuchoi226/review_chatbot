import os
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional, Tuple
import openai
import numpy as np
from dotenv import load_dotenv
import faiss
from app.models.history import ConversationHistory
from app.models.embedding import TextEmbedding
from app.services.vectorstore import VectorStore
from app.services.tokenizer import TextTokenizer
from app.services.semantic_chunking import SemanticChunker


class Chatbot:
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        faiss_base_path = os.getenv("FAISS_BASE_PATH", "/CHATBOT/data/review")
        self.review_data = VectorStore.load(faiss_base_path)
        self.tokenizer = TextTokenizer()
        self.chunker = SemanticChunker(self.tokenizer)
        self.embedder = TextEmbedding()
        self.similarity_threshold = 0.3  
        self.follow_up_threshold = 0.5 
        self.context_window = 4  

    def l2_similarity(self, vec1, vec2):
        """L2 거리를 유사도 점수로 변환 (거리가 작을수록 유사도가 높음)"""
        l2_distance = np.linalg.norm(vec1 - vec2)
        # L2 거리를 0~1 사이의 유사도 점수로 변환
        similarity = 1 / (1 + l2_distance)
        return similarity

    def retrieve_documents_with_similarity(self, query, top_k=5):
        try:
            query_vector = self.embedder.embed_documents(query)
            result_docs = self.review_data.similarity_search(query, k=top_k)
            results_with_similarity = []

            for doc in result_docs:
                doc_vector = self.embedder.embed_documents(doc['text'])
                similarity = self.l2_similarity(query_vector, doc_vector)
                results_with_similarity.append((doc, similarity))
                print(f"문서: {doc['text'][:150]}... 유사도: {similarity:.4f}")

            results_with_similarity = sorted(results_with_similarity, key=lambda x: x[1], reverse=True)
            return results_with_similarity
        except Exception as e:
            print(f"❌ 문서 검색 중 오류 발생: {str(e)}")
            return []

    def contextualized_retrieval_with_similarity(self, user_question, conversation_history, user_id, top_k=1):
        try:
            conversation_history.load_index(user_id)
            context = ""
            if conversation_history.history_vectorstore:
                relevant_docs = conversation_history.history_vectorstore.similarity_search(
                    user_question, 
                    k=self.context_window
                )
                if relevant_docs:
                    sorted_docs = sorted(
                        relevant_docs,
                        key=lambda x: x.metadata.get('timestamp', ''),
                        reverse=True
                    )
                    context = "\n\n".join([doc.page_content for doc in sorted_docs])
                    print(f"이전 대화 맥락: {context[:150]}...")

            # 질문과 맥락을 결합하여 검색
            enhanced_question = f"{context} {user_question}" if context else user_question
            main_results_with_similarity = self.retrieve_documents_with_similarity(enhanced_question, top_k)
            return main_results_with_similarity
        except Exception as e:
            print(f"❌ 맥락 기반 검색 중 오류 발생: {str(e)}")
            return []



    def generate_prompt_with_similarity(self, user_input, conversation_history, user_id):
        try:
            results_with_similarity = self.contextualized_retrieval_with_similarity(
                user_input, conversation_history, user_id, top_k=3
            )
            
            print("\n--- 유사도---")
            for doc, similarity in results_with_similarity:
                print(f"유사도: {similarity:.4f}")
            print("-------------------")

            context = "\n\n".join([
                f"{doc['text']} (유사도: {similarity:.4f})" 
                for doc, similarity in results_with_similarity
            ])

            prompt = f"""
                당신은 스마트스토어 전문 챗봇입니다.
                다음 지침을 따라 답변해주세요:
                1. 제공된 컨텍스트를 기반으로 정확하고 친절하게 답변해주세요.
                2. 답변은 간단명료하게 해주세요.
                3. 필요한 경우 예시를 들어 설명해주세요.
                4. 답변 후에는 사용자가 더 궁금해할 만한 후속 질문을 제안해주세요.
                
                스마트스토어와 관련 없는 질문에는 
                "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다."
                라는 메시지를 반환한 뒤, 사용자의 질문과 연관있는 질문을 제시해주세요.
                
                ---
                CONTEXT:
                {context}
            """
            return prompt, context
        except Exception as e:
            print(f"❌ 프롬프트 생성 중 오류 발생: {str(e)}")
            return "", ""


    def generate_answer_with_similarity(self, user_input, conversation_history, user_id):
        try:
            results_with_similarity = self.retrieve_documents_with_similarity(user_input, top_k=3)
            average_similarity = np.mean([similarity for _, similarity in results_with_similarity])

            if average_similarity < self.similarity_threshold:  
                return "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다."

            conversation_history.add_conversation(
                user_id=user_id,
                user_question=user_input
            )

            prompt, context = self.generate_prompt_with_similarity(user_input, conversation_history, user_id)
            if not prompt or not context:
                return "죄송합니다. 답변을 생성하는 데 문제가 발생했습니다."

            response = openai.Completion.create(
                engine="gpt-4o-mini",
                prompt=prompt,
                max_tokens=150,
                n=1,
                stop=None,
                temperature=0.7,
                stream=True  # 스트리밍 활성화
            )
            answer = ""
            for chunk in response:
                if 'choices' in chunk:
                    choice = chunk['choices'][0]
                    if 'text' in choice:
                        answer += choice['text']
                        print(choice['text'], end='')

            # 챗봇 답변 저장
            conversation_history.add_bot_response(
                user_id=user_id,
                bot_answer=answer
            )

            # 후속 질문
            follow_up_questions = self.suggest_follow_up_questions(user_input, answer)
            full_response = answer.strip() + "\n" + "\n".join(follow_up_questions)
            return full_response

        except Exception as e:
            print(f"❌ 답변 생성 중 오류 발생: {str(e)}")
            return f"죄송합니다. 답변을 생성하는 데 문제가 발생했습니다: {str(e)}"



    def suggest_follow_up_questions(self, user_input, answer):
        try:
            follow_up_questions = []
            combined_text = f"{user_input} {answer}"
            similar_questions = self.retrieve_documents_with_similarity(combined_text, top_k=3)
            
            high_similarity_docs = [
                doc for doc, similarity in similar_questions 
                if similarity > self.follow_up_threshold
            ]
            
            if high_similarity_docs:
                best_doc = high_similarity_docs[0]
                follow_up_prompt = f"""
                사용자가 '{best_doc['text']}'에 대해 질문했습니다. 
                이 질문에 대한 후속 질문을 생성해 주세요. 
                다음 지침을 따라주세요:
                1. 사용자가 더 깊이 이해할 수 있도록 돕는 내용이어야 합니다.
                2. 현재 대화 맥락과 자연스럽게 연결되어야 합니다.
                3. 1-2개의 구체적인 질문을 생성해주세요.
                4. 각 질문은 한 문장으로 작성해주세요.
                """
                try:
                    response = openai.Completion.create(
                        engine="gpt-4o-mini",
                        prompt=follow_up_prompt,
                        max_tokens=50,
                        n=1,
                        stop=None,
                        temperature=0.7
                    )
                    follow_up_question = response.choices[0].text.strip()
                    follow_up_questions.append(f"- {follow_up_question}")
                except Exception as e:
                    print(f"❌ 후속 질문 생성 중 오류 발생: {str(e)}")

            return follow_up_questions if follow_up_questions else ["- 더 자세한 설명이 필요하신가요?"]
        except Exception as e:
            print(f"❌ 후속 질문 생성 중 오류 발생: {str(e)}")
            return ["- 더 자세한 설명이 필요하신가요?"] 