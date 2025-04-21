# import os
# import keys
# os.environ["OPENAI_API_KEY"] = keys.openai.api_key
# OPENAI_MODEL = "gpt-4o-mini"

# from langchain_openai import OpenAIEmbeddings
# embeddings= OpenAIEmbeddings(model=OPENAI_MODEL)
# from openai import OpenAI
# client = OpenAI(api_key=keys.openai.api_key, organization=keys.openai.organization)

import pandas as pd
import re 
import os
# from app import generate_response, fillin_messages, segment_by_paragraph

def gen_queries(doc_path):
    with open(doc_path, 'r', encoding='utf-8') as f:
        doc = f.read()
    segs = segment_by_paragraph(doc)
    print(doc_path, len(segs))

    system_prompt = """You're a QA agent, expert in Medicine and its regulation.
    According to the doc contents: generate 3 pairs of questions user could ask and their answers.
    The QA must be clear and detailed.
    The output should be in Korean."""
    user_prompt = """Doc: {0}"""
    
    res = []
    for sid, seg in enumerate(segs):
        _user_prompt = user_prompt.format(seg)
        msg = fillin_messages(system_prompt, _user_prompt)
        resp = generate_response(msg)
        res.append([sid, resp, seg])
    return res

def doc_to_querys():
    data_dir = './data/processed/250314/안내서.지침-의약품'
    files = os.listdir(data_dir)[:4]
    # 0-3.csv
    #./data/processed/250314/안내서.지침-의약품\1. 니자티딘+의약품+중+NDMA+분석법_공개.txt 3       
    # ./data/processed/250314/안내서.지침-의약품\10. 시판+후+의약품+위해성+관리계획+수립+시+약물유전체+활용+가이드라인.txt 21
    # ./data/processed/250314/안내서.지침-의약품\11. 의약품+임상시험+대조군+설정+가이드라인.txt 42  
    # ./data/processed/250314/안내서.지침-의약품\12. 의약품+임상시험+시+성별+고려사항+가이드라인.txt 13
    df_docs = []
    for doc_id, doc_name in enumerate(files):
        doc_path = os.path.join(data_dir, doc_name)
        doc_qa = gen_queries(doc_path)
        df = pd.DataFrame(doc_qa, columns=['seg_id', 'response', 'segment'])

        df["doc_id"] = doc_id
        df["doc_name"] = doc_name

        df_docs.append(df)

    final = pd.concat(df_docs)
    final.to_csv(f'{len(files)}.csv', index=False)

# TODO gen_query, data_processor 이것들 정리하기 
def parse_qa_doc():
    data_dir = './data/processed/250314/안내서.지침-의약품'
    qadocs = [
    # "27. 의료제품+임상통계+상담사례집.txt", # 이미지네..
"30. 2021+임상시험+관련+자주묻는+질의응답.txt",
"15. 항암제+비임상시험+가이드라인+질의응답집.txt", 
"53-2. 의약품+허가+후+제조방법+변경관리+질의응답집(민원인안내서).txt",
    ]
    res = []
    for doc_id, doc_name in enumerate(qadocs):
        doc_path = os.path.join(data_dir, doc_name)
        if not os.path.exists(doc_path): continue
        with open(doc_path, 'r', encoding='utf-8') as f:
            doc = f.readlines()
        
        q = "" # 한 질문에 
        a = [] # 답이 여러 체크포인트로 이뤄짐 
        isq = False
        for line in doc:
            line = line.lower()
            if line[0] in ['v', 'a', '○']:
                a.append(line) 
            elif line.startswith('q'):                
                if a:
                    res.append([doc_name, q, len(a), a])
                    a = []
                    q = ""
                q = line
            else:
                if q and a:
                    a[-1] += " " + line 
                elif q and not a:
                    q += " " + line 
    y = pd.DataFrame(res, columns=['doc_name', 'query', 'num_answer', 'answer'])    
    y.to_csv('qa-doc.csv', index=False)            

def postprocess():
    csvfile = '0-3.csv'
    df = pd.read_csv(csvfile)
    df2 = []
    columns = ['doc_id', 'seg_id', 'doc_name','segment']
    for i, r in df.iterrows():
        lines = r.response.split('\n')
        lines = [line.strip() for line in lines if len(line.strip()) > 0]
        r2 = r[columns]
        
        for qid, x in enumerate(lines):
            r2['name'] = f'{r.doc_id}-{r.seg_id}-{qid//2}'
            if '질문' in x[:10] and '답변' in x[:10]:
                continue 
            elif '질문' in x[:10]:
                r2['query'] = x
            elif '답변' in x[:10]:
                r2['answer'] = x                
                df2.append(r2)
                r2=r[columns]
            else:
                continue 

    y = pd.concat(df2, axis=1).transpose()
    y = y[['name', 'doc_id', 'seg_id', 'doc_name', 'query', 'answer', 'segment']]
    y.to_csv('0-3-postprocess.csv', index=False)
    print()

if __name__=='__main__':
    # postprocess()
    parse_qa_doc()
