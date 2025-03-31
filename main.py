from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from app.core.chat import Chatbot

app = FastAPI(title="Smart Store FAQ Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chatbot 인스턴스 생성
chatbot = Chatbot()

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    try:
        if not chat_request.message:
            raise ValueError("사용자 메시지가 비어있습니다.")
            
        async def generate():
            # 챗봇의 스트리밍 응답을 직접 전달
            for chunk in chatbot.generate_answer_with_similarity(
                user_input=chat_request.message,
                user_id=chat_request.user_id
            ):
                if chunk:  # 빈 청크 건너뛰기
                    yield chunk
        
        return StreamingResponse(
            generate(),
            media_type="text/plain"  
        )

    except Exception as e:
        print(f"Chat error: {str(e)}")  # 디버그 로그 추가
        raise HTTPException(
            status_code=400,
            detail=f"챗봇 응답 생성 실패: {str(e)}"
        )
