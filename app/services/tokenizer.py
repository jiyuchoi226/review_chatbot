from typing import List
import re
import tiktoken
import emoji

class TextTokenizer:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.sentence_endings = [
            r'[.!?]',
            r'[.!?][\'")\]]',
            r'[.!?]\s*[A-Z]',
            r'[.!?]\s*[가-힣]',
            r'[.!?]\s*[0-9]',
            r'[.!?]\s*[ㄱ-ㅎㅏ-ㅣ]',
            r'[.!?]\s*[^a-zA-Z0-9가-힣]',
        ]
        self.sentence_pattern = '|'.join(self.sentence_endings)
    
    #전처리
    def preprocess_and_split(self, text: str) -> List[str]:
        if not text or not isinstance(text, str):
            return []
            
        try:
            text = emoji.demojize(text)  
            text = re.sub(r"\s+", " ", text).strip()
            
            if not text:
                return []
            sentences = re.split(f'(?<=[{self.sentence_pattern}])\s+', text)
            cleaned_sentences = []
            for sent in sentences:
                sent = sent.strip()
                if not sent or len(sent) < 2:  # 너무 짧은 문장 제외
                    continue
                if not re.search(f'[{self.sentence_pattern}]$', sent):
                    if not cleaned_sentences or not re.search(f'[{self.sentence_pattern}]$', cleaned_sentences[-1]):
                        cleaned_sentences.append(sent)
                    else:
                        cleaned_sentences[-1] = cleaned_sentences[-1] + " " + sent
                else:
                    cleaned_sentences.append(sent)
            return cleaned_sentences
            
        except Exception as e:
            print(f"텍스트 전처리 중 오류 발생: {str(e)}")
            return []

    #토큰화
    def process_text(self, text: str) -> List[List[int]]:
        if not text or not isinstance(text, str):
            return []
            
        try:
            sentences = self.preprocess_and_split(text)
            if not sentences:
                return []
                
            result = []
            for sentence in sentences:
                try:
                    tokens = self.tokenizer.encode(sentence)
                    if tokens:  
                        result.append(tokens)
                except Exception as e:
                    print(f"문장 토큰화 중 오류 발생: {str(e)}")
                    continue
            return result
        except Exception as e:
            print(f"텍스트 처리 중 오류 발생: {str(e)}")
            return []
    
    #토큰 수 계산   
    def count_tokens(self, text: str) -> int:
        if not text or not isinstance(text, str):
            return 0

        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            print(f"토큰 수 계산 중 오류 발생: {str(e)}")
            return 0

