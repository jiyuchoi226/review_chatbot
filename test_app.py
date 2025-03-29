import streamlit as st
import requests
import json
import sseclient

# FastAPI 서버 URL
API_URL = "http://localhost:8000"

def chat_with_bot(message, user_id="test_user"):
    try:
        # SSE 클라이언트 설정
        headers = {'Accept': 'text/event-stream'}
        response = requests.post(
            f"{API_URL}/chat",
            json={"message": message, "user_id": user_id},
            headers=headers,
            stream=True
        )
        response.raise_for_status()
        
        client = sseclient.SSEClient(response)
        full_response = ""
        message_placeholder = st.empty()
        
        # 스트리밍 응답 처리
        for event in client.events():
            if event.data:
                try:
                    data = json.loads(event.data)
                    if "message" in data:
                        chunk = data["message"]
                        full_response += chunk
                        message_placeholder.markdown(full_response)
                except json.JSONDecodeError:
                    continue
        
        return full_response
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return f"Error: {str(e)}"

# Streamlit UI
st.title("스마트 스토어 FAQ 챗봇 테스트")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("질문을 입력하세요"):
    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 챗봇 응답
    with st.chat_message("assistant"):
        response = chat_with_bot(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response}) 