import os
import json
import shutil
import traceback
import pickle
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict
from app.services.tokenizer import TextTokenizer
from app.services.semantic_chunking import SemanticChunker
from app.services.embedding import TextEmbedding
import faiss
import numpy as np
import logging

class ConversationHistory:
    def __init__(self, embeddings, text_splitter):
        self.embeddings = embeddings
        self.text_splitter = text_splitter
        self.history = {}
        self.history_vectorstore = None
        self.base_index_path = os.getenv("HISTORY_PATH", "data/conversation")
        self.texts: List[str] = []
        self.metadata: List[Dict] = []
        
        # 로깅 설정
        self._setup_logging()
    
    def _setup_logging(self):
        # 로그 디렉토리 생성
        log_dir = os.path.join(self.base_index_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 로그 파일명 설정 (날짜별)
        log_file = os.path.join(log_dir, f"conversation_{datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d')}.log")
        
        # 로거 설정
        self.logger = logging.getLogger('ConversationHistory')
        self.logger.setLevel(logging.DEBUG)
        
        # 파일 핸들러
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        
        # 포맷 설정
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 핸들러 추가
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_history(self, user_id: str):
        history_entries = self.history.get(user_id, [])
        return " ".join([f"Bot: {entry['bot_question']} User: {entry['user_answer']}" for entry in history_entries])
    
    # 날짜별 대화 저장 경로 생성
    def _get_user_history_path(self, user_id: str) -> str:
        date = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
        user_path = os.path.join(self.base_index_path, user_id)
        os.makedirs(user_path, exist_ok=True)
        return user_path
    
    # 현재 날짜의 대화 인덱스 저장
    def save_index(self, user_id: str):
        try:
            if self.history_vectorstore:
                user_path = self._get_user_history_path(user_id)
                date = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
                
                # FAISS 인덱스 저장
                index_path = os.path.join(user_path, f"{date}.faiss")
                self.logger.info(f"FAISS 인덱스 저장 경로: {index_path}")
                faiss.write_index(self.history_vectorstore, index_path)
                
                # 텍스트와 메타데이터 저장
                pkl_path = os.path.join(user_path, f"{date}.pkl")
                self.logger.info(f"대화 기록 저장 경로: {pkl_path}")
                with open(pkl_path, "wb") as f:
                    pickle.dump({
                        'texts': self.texts,
                        'metadata': self.metadata,
                        'dimension': self.history_vectorstore.d
                    }, f)
                
                self.logger.info(f"✅ 대화 기록 저장 완료: {index_path}, {pkl_path}")
            else:
                self.logger.warning("⚠️ 저장할 FAISS 인덱스가 없습니다.")
    
        except Exception as e:
            self.logger.error(f"❌ 대화 기록 저장 중 오류 발생: {str(e)}")
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
   
    # 현재 날짜의 대화 인덱스 로드
    def load_index(self, user_id: str):
        try:
            user_path = self._get_user_history_path(user_id)
            date = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
            
            # FAISS 인덱스 로드
            index_file = os.path.join(user_path, f"{date}.faiss")
            pkl_file = os.path.join(user_path, f"{date}.pkl")
            
            self.logger.info(f"대화 기록 로드 경로: {index_file}, {pkl_file}")
            
            if os.path.exists(index_file) and os.path.exists(pkl_file):
                # FAISS 인덱스 로드
                self.history_vectorstore = faiss.read_index(index_file)
                
                # 텍스트와 메타데이터 로드
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)
                    self.texts = data['texts']
                    self.metadata = data['metadata']
                
                self.logger.info(f"✅ 대화 기록 로드 완료: {len(self.texts)}개 항목")
            else:
                self.logger.warning(f"⚠️ 대화 기록 파일이 존재하지 않습니다: {index_file}, {pkl_file}")
                self.history_vectorstore = None
                self.texts = []
                self.metadata = []
                
        except Exception as e:
            self.logger.error(f"❌ 대화 기록 로드 중 오류 발생: {str(e)}")
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            self.history_vectorstore = None
            self.texts = []
            self.metadata = []

    def add_conversation(self, user_id: str, user_question: str, bot_answer: str = None, faq_category: str = None, event_info: dict = None, emotion_info: dict = None):
        try:
            self.logger.info(f"ConversationHistory - 대화 저장 시작")
            
            if user_id not in self.history:
                self.history[user_id] = []
            
            current_time = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
            
            try:
                # 대화 내용을 리스트로 준비
                conversation_texts = []
                conversation_metadatas = []
                
                # 사용자 질문 추가
                conversation_texts.append(user_question)
                conversation_metadatas.append({
                    'user_id': user_id,
                    'timestamp': current_time,
                    'type': 'user_question',
                    'faq_category': faq_category,
                    'event_info': event_info,
                    'emotion_info': emotion_info
                })
                
                # 챗봇 답변 추가
                if bot_answer:
                    conversation_texts.append(bot_answer)
                    conversation_metadatas.append({
                        'user_id': user_id,
                        'timestamp': current_time,
                        'type': 'bot_answer',
                        'faq_category': faq_category,
                        'event_info': event_info,
                        'emotion_info': emotion_info
                    })
                
                self.logger.info(f"대화 텍스트 생성: {len(conversation_texts)}개 항목")
                
                # 임베딩 생성
                embeddings = self.embeddings.embed_documents(conversation_texts)
                self.logger.info(f"임베딩 생성 완료: shape={embeddings.shape}")

                # FAISS 인덱스 초기화 또는 업데이트
                if self.history_vectorstore is None:
                    d = embeddings.shape[1]
                    self.history_vectorstore = faiss.IndexFlatL2(d)
                    self.logger.info(f"새로운 FAISS 인덱스 생성: dimension={d}")
                
                # 임베딩 추가
                self.history_vectorstore.add(embeddings)
                
                # 텍스트와 메타데이터 추가
                self.texts.extend(conversation_texts)
                self.metadata.extend(conversation_metadatas)
                
                self.logger.info(f"대화 기록 추가 완료: 현재 총 {len(self.texts)}개 항목")
                
                # 인덱스 저장 (비동기로 처리)
                self.save_index(user_id)
                self.logger.info("인덱스 저장 완료")
                
            except Exception as e:
                self.logger.error(f"⚠️ FAISS 인덱스 업데이트 중 오류 발생: {str(e)}")
                self.logger.error(f"상세 오류: {traceback.format_exc()}")
                            
        except Exception as e:
            self.logger.error(f"❌ 대화 기록 저장 중 오류 발생: {str(e)}")
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            raise e

    def delete_conversation_history(self, user_id: str):
        try:
            user_path = self._get_user_history_path(user_id)
            date = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
            index_file = os.path.join(user_path, f"{date}.faiss")
            pkl_file = os.path.join(user_path, f"{date}.pkl")
            
            if os.path.exists(index_file):
                os.remove(index_file)
            if os.path.exists(pkl_file):
                os.remove(pkl_file)
                
            self.history_vectorstore = None
            self.texts = []
            self.metadata = []
            self.logger.info(f"사용자 {user_id}의 오늘 대화 기록이 삭제되었습니다.")
                
        except Exception as e:
            self.logger.error(f"대화 기록 삭제 중 오류 발생: {str(e)}")
            raise e
