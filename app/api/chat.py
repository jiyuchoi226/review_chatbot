from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from app.core.chat import Chatbot
from app.models.history import ConversationHistory
from app.services.embedding import TextEmbedding
from app.services.semantic_chunking import SemanticChunker
from app.services.tokenizer import TextTokenizer

router = APIRouter()

# 전역 변수로 초기화
tokenizer = TextTokenizer()
chunker = SemanticChunker(tokenizer)
embedder = TextEmbedding()
conversation_history = ConversationHistory(embedder, chunker)
chatbot = Chatbot()

class ChatRequest(BaseModel):
    message: str
    user_id: str = "test_user"

@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        async def generate():
            response = chatbot.generate_answer_with_similarity(
                request.message,
                conversation_history,
                request.user_id
            )
            
            # 스트리밍 응답을 SSE 형식으로 전송
            for chunk in response:
                yield f"data: {json.dumps({'message': chunk})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 