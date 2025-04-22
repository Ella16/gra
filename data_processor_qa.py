import os
import pandas as pd
import re

def keep_first_occurrence(text, substring):
    found = False

    def replacer(match):
        nonlocal found
        if not found:
            found = True
            return match.group(0)  # 첫 번째는 유지
        else:
            return ''  # 이후는 제거

    # re.escape: substring에 특수문자 있을 경우 대비
    pattern = re.compile(re.escape(substring))
    return pattern.sub(replacer, text)

def clean_text(text):
    try:
        text = text.strip()
        for comment in ['배포용', '최종본', '붙임', '최종', '제정', '개정', 
                        '.pdf']:
            text = text.replace(comment, '')
        text = re.sub(r'(19|20)(\d{2})(년\s|년|\.|)((|\s|_|)(\d{1,2})(월|\.|\s|)?((\d{1,2})+(일|\.|))?)?', '@YEAR@', text) # 2024년 2024. 20240922
        
        text = re.sub(r'(^\d+|\d+$)+(-(\d{1,2}))?', ' ', text) # 앞뒤 숫자 삭제
        text = re.sub(r'\(\d+\)', ' ', text) # 괄호 안 숫자 삭제
        text = re.sub(r'^[^\w가-힣「」@YEAR@]+|[^\w가-힣」@YEAR@]+$', ' ', text) # 특수문자 삭제제
        text = re.sub(r'[^\w가-힣「」@YEAR@]', ' ', text)
        text = text.replace('_', ' ')
        text = text.replace('  ', '')
        for faq in ['자주하는질의응답집', '자주묻는질의집', '민원질의응답모음집', '자주 하는', '자주하는', '자주 묻는', '자주묻는', 
                    '질문집', '질의 응답집', '질의응답집', '질의 응답', '질의응답', 'FAQ',  'QNA',
                    '가이드라인','민원인 안내서',  '민원인안내서','안내서', '민원', '이백문이백답'] :
            text = text.replace(faq, '@faq@') 
        
        text = keep_first_occurrence(text, '@YEAR@')
        text = keep_first_occurrence(text, '@faq@')
        text = text.replace('@faq@', '@FAQ@') 
        text = text.replace('@@', '@ @') 
        text = text.strip()
        # text = re.sub(r'^\d+|\d+$', ' ', text)
        
    except Exception as e:
        print('failed', text, e)
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
    data_dir = './data/250415/'
    file = 'qa-250422.csv'
    df = pd.read_csv(os.path.join(data_dir, 'qa-250422.csv')) 
    df = df[df.valid!='X']
    df = df[['valid', 'src', 'ref', 'Q', 'A']]
    df = df.fillna('')
    
    src = []
    for text in df.src.unique():
        src.append(clean_text(text))
    
    df = df.merge(pd.DataFrame({"src":df.src.unique(), "src_processed":src}),
                  on='src', how='left')
    
    bullets = ['v', '답변:', '❍' , 'O', '■', '○', '(답변 1)', '◉', '〇' , '▶' , '-', "ü", '∘','*' ]
    bullets = [[b, len(b)] for b in bullets]
    remove_list = ['(중략 )', '(중략)', '(생략 )', '(생략)']
    df['A_processed'] = None
    for i, r in df.iterrows():
        lines = r['A'].split('\n')
        for j in range(len(lines)-1, -1, -1):
            l = lines[j].strip()
            for b, lenb in bullets:
                if l.startswith(b):
                    l = l[lenb:]
            for r in remove_list:
                l = l.replace(r, '')
            l = l.strip()
            if not l:
                del lines[j]
            else:
                lines[j] = l
            
        df.at[i, 'A_processed'] = lines[::-1]
        if (i+1)%100 == 0:
            print(i, 'processed')
    
    df.to_csv(os.path.join(data_dir, file.replace('.csv', '-processed.csv')))
        

if __name__=="__main__":
    preprocess_qa()
    