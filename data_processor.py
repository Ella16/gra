# ocr free pdf processor
import sys
sys.path.append('./youme')
from invoice_recog.file_processor import FileProcessor
from invoice_recog.tool_layer_manager import Tokenizer
from invoice_recog.utils import utils
import keys
from openai import OpenAI

import os

openai_client = OpenAI(api_key=keys.openai.api_key, organization=keys.openai.organization)
GPT_MODEL = "gpt-4o-mini"
EMBEDDING_ENGINE = "text-embedding-ada-002"
model_name =  GPT_MODEL
temperature = 0.7


class OCRFreeFileProcessor(FileProcessor):
    def calc_cost(self, model):
        self.num_pixels.extend(self.pdf_reader.num_pixels)
        self.num_input_tokens.extend(self.pdf_reader.num_input_tokens)
        self.num_input_tokens.extend(self.invoice_recognizer.num_input_tokens)
        self.num_output_tokens.extend(self.pdf_reader.num_output_tokens)
        self.num_output_tokens.extend(self.invoice_recognizer.num_output_tokens)

        cost = utils.calculate_api_call_cost(
            model,
            len(self.num_pixels),
            sum(self.num_input_tokens),
            sum(self.num_output_tokens),
        )
        return cost
    
    def process_file(self, file: str, client, model) -> tuple[dict, tuple[float]]:
        # remove temporary files
        self.pdf_reader.reset()
        self.invoice_recognizer.reset()
        
        # generate markdown
        result = self._generate_markdown(file, client, model) # {"file":file_basename,"contents": markdown_text, "num_pages": len(base64_images) }
        result['cost'] = self.calc_cost(model)

        return result

def make_markdown():
    import os
    import time
    import json 
    
    tokenizer = Tokenizer(tiktoken_cache_dir = "./youme/data/tiktoken_cache")
    file_proc = OCRFreeFileProcessor(tokenizer)
    data_dir = './data/250415/pdfs'
    result_dir = './data/250415/processed'
    
    os.makedirs(result_dir, exist_ok=True)
    processed = [os.path.basename(pdf).replace('.md', '.pdf').lower() for pdf in os.listdir(result_dir) 
            if pdf.endswith('.md')]
    pdfs = []
    for pdf in os.listdir(data_dir):
        basename = os.path.basename(pdf).lower()
        if basename.endswith('.pdf'):
            if basename not in processed:
                pdfs.append(os.path.join(data_dir, pdf))
            else:
                print(f'already processed', pdf)
    
    # pdfs = pdfs[:1] # # 224-5. 5. 보툴리눔 독 소제제의 국가검정제도 소개.pdf 만 하기 
    # pdfs = pdfs
    num = len(pdfs)
    res = []
    print('num pdfs', num)
    for i, file in enumerate(pdfs):
        try:
            basename = os.path.basename(file)
            print(f'run_ocr_free_proc: processing {i}/{num} {basename}')
            st = time.time()
            result = file_proc.process_file(file, openai_client, model_name) 
            took = time.time() - st
            res.append(result)        
            print(f'run_ocr_free_proc: end {i}/{num} {basename}', result['file'], result['cost'], f'{took:.0f} s')
            
            f = os.path.join(result_dir, basename.replace('.pdf', '.md'))
            json.dump(result, open(f, 'w', encoding='utf-8'), ensure_ascii=False )
            time.sleep(10)
        except Exception as e:
            print(file, e)
            time.sleep(10)
            continue 
    import pandas as pd
    pd.DataFrame(res)[['file', 'num_pages', 'cost', 'contents']].to_csv('result.csv')
    

import re

def clean_text(text):
    # 1. '2023.', '1999년' → @YEAR@
    text = re.sub(r'\b(19|20)\d{2}(년|\.)\b?', '@YEAR@', text)

    # 2. 괄호 안 숫자로 끝나는 경우 삭제: 예) (3), (12)
    text = re.sub(r'\(\d+\)$', '', text)

    # 3. 앞뒤 특수문자 제거 (단, 「」는 보존)
    text = re.sub(r'^[^\w가-힣「」]+|[^\w가-힣」]+$', '', text)

    # 4. 중간 특수문자 제거 (숫자, 영문, 한글, 「」만 유지)
    text = re.sub(r'[^\w가-힣「」]', '', text)

    return text

def preprocess_qa():
    # src, ref, q, a
    # 안내서-0194-02+「의약품+국제공통기술문서(CTD)+해설서」[민원인+안내서].pdf
    # 의약품+국제공통기술문서+작성을+위한+질의응답집(품질)(민원인안내서) 얘는 QA 다시 만들어야 할 것 같다
    # 배포용, 최종본, 붙임, 최종, .pdf 
    
    # src
    # 질의응답집, 민원인안내서, 년도 표기 등은 기호로 대체
    
    # a: new line이 이걸로 시작할 경우 리스트로 표현해야 할듯? v, 답변: ❍ , O, ■, ○, (답변 1), ◉, 〇 , ▶ , -
    # Q 3.을 참조하시기 바랍니다. 는 삭제하기 
    dir = './data/250415/'
    file = os.path.join(dir, )
     
    
if __name__=="__main__":
    # make_markdown()
    