import streamlit as st
import requests
import json
import time

# FastAPI 서버 URL
# API_URL = "http://localhost:8080"
API_URL = "http://localhost:8000"

def chat_with_bot(message, user_id="test_user"):
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"message": message, "user_id": user_id},
            stream=True
        )
        response.raise_for_status()
        full_response = ""
        message_placeholder = st.empty()
        
        # 응답 출력
        for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
            if chunk:
                full_response += chunk
                message_placeholder.markdown(full_response + "|")
                if chunk in [".", "!", "?", "\n"]:
                    time.sleep(0.1)  # 문장 끝에서 좀 더 긴 딜레이
                elif chunk == " ":
                    time.sleep(0.05)  # 단어 사이에서 중간 딜레이
                else:
                    time.sleep(0.02)  # 일반 문자에서 짧은 딜레이
        
        # 최종 응답 표시 
        message_placeholder.markdown(full_response)
        return full_response
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return f"Error: {str(e)}"

# Streamlit UI
st.title("스마트 스토어 FAQ 챗봇 테스트")
# 커스텀 CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@300;400;500;600&display=swap');
    * { font-family: 'IBM Plex Sans KR', sans-serif !important;}
 
    .e1nzilvr1{
        font-size: 30px;
    }
</style>
""", unsafe_allow_html=True)

# 소개 메시지
st.markdown("""
<div style='padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem; margin-bottom: 2rem;'>
    <span style='margin: 0px; font-weight: 600;'>안녕하세요! 스마트 스토어 FAQ 챗봇입니다 👋</span>
    <p style='margin: 0px;'>스마트 스토어에 대해 궁금하신 점을 자유롭게 물어보세요.</p>
</div>
""", unsafe_allow_html=True)
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
