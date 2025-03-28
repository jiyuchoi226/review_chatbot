from app.services.vectorstore import VectorStore
import pickle
import json
import re

def extract_category(question: str) -> str:
    """질문 텍스트에서 카테고리를 추출합니다."""
    category_match = re.match(r'\[(.*?)\]', question)
    return category_match.group(1) if category_match else "일반"

def clean_text(text: str) -> str:
    """텍스트에서 불필요한 부분을 제거합니다."""
    if not text:
        return ""
    
    # 대괄호로 둘러싸인 카테고리 제거
    text = re.sub(r'\[.*?\]', '', text)
    
    # 불필요한 문구 제거
    removals = [
        "위 도움말이 도움이 되었나요?",
        "별점",
        "관련 도움말/키워드"
    ]
    
    for phrase in removals:
        if phrase in text:
            text = text.split(phrase)[0]
    
    return text.strip()

def store_faq_data():
    print("1. FAQ 데이터 로드 중...")
    with open("data/final_result.pkl", "rb") as f:
        faq_data = pickle.load(f)
    print(f"- 총 FAQ 항목 수: {len(faq_data)}")
    
    print("\n2. FAQ 데이터 전처리 중...")
    faq_list = []
    categories = set()
    
    for question, answer in faq_data.items():
        if isinstance(answer, str):
            category = extract_category(question)
            clean_question = clean_text(question)
            clean_answer = clean_text(answer)
            
            if clean_question and clean_answer:
                categories.add(category)
                faq_list.append({
                    'question': clean_question,
                    'answer': clean_answer,
                    'category': category
                })
    
    print(f"- 처리된 FAQ 항목 수: {len(faq_list)}")
    print(f"- 카테고리 목록: {sorted(list(categories))}")
    
    if faq_list:
        print("\n3. 데이터 샘플:")
        sample = faq_list[0]
        print(f"- 카테고리: {sample['category']}")
        print(f"- 질문: {sample['question']}")
        print(f"- 답변: {sample['answer'][:200]}...")
    
    print("\n4. 벡터 저장소 초기화 중...")
    vector_store = VectorStore()
    
    print("\n5. FAQ 데이터 벡터화 및 저장 중...")
    if faq_list:
        questions = [item['question'] for item in faq_list]
        metadatas = [{
            'answer': item['answer'],
            'category': item['category'],
            'type': 'faq'
        } for item in faq_list]
        
        # 배치 크기를 작게 설정하여 처리
        batch_size = 100
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            vector_store.add_texts(
                texts=batch_questions,
                metadatas=batch_metadatas
            )
            print(f"- 진행 상황: {min(i + batch_size, len(questions))}/{len(questions)} 처리됨")
        
        print(f"\n- 저장된 텍스트 수: {len(vector_store.texts)}")
        print(f"- FAISS 인덱스 크기: {vector_store.index.ntotal}")
        
        print("\n6. 벡터 저장소를 data/review 디렉토리에 저장 중...")
        vector_store.save("data/review")
        print("- 저장 완료!")
        
        print("\n7. 테스트 검색 실행...")
        results = vector_store.similarity_search("상품 등록 방법에 대해 알고 싶어요", k=3)
        
        print("\n8. 검색 결과:")
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n결과 {i}:")
                print(f"카테고리: {result['metadata']['category']}")
                print(f"질문: {result['text']}")
                print(f"답변: {result['metadata']['answer']}")
                print(f"유사도 거리: {result['distance']}")
        else:
            print("검색 결과가 없습니다.")
    else:
        print("\n경고: 벡터 저장소에 데이터가 없습니다!")

if __name__ == "__main__":
    store_faq_data()