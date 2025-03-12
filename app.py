import os
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import ast
from openai import OpenAI
import streamlit as st
from streamlit_chat import message

import keys # Set OpenAI API key 
client = OpenAI(api_key=keys.openai.api_key, organization=keys.openai.organization)

DOC_EMB = None

def get_embedding(text, engine):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=engine).data[0].embedding

def load_doc_embedding():
    emb_dir = './embedding'
    emb_name = 'embedding.csv'
    emb_path = os.path.join(emb_dir, emb_name)

    if os.path.isfile(emb_path):
        df = pd.read_csv(emb_path)
        df['embedding'] = df['embedding'].apply(ast.literal_eval)
    else:
        data_dir = './data'
        doc_names = [file for file in os.listdir(data_dir) if file.endswith('.txt')]

        docs = []
        for file in doc_names:
            doc_path = os.path.join(data_dir, file)
            with open(doc_path, 'r', encoding='utf-8') as f:
                single_doc = f.read()
                docs.append([doc_path, single_doc])
        
        # TODO 벡터디비로 바꿔야 할듯 
        context_len = 2000
        window = context_len // 10 # 10% 를 윈도우로 잡는다. 

        embedding = []
        for docid, (_, single_doc) in enumerate(docs):
            doclen = len(single_doc)
            if doclen > context_len:
                num_seg = doclen // context_len 
                seg_len = doclen // num_seg + 1
                for sid in range(seg_len):
                    doc_seg = single_doc[max(0, sid*context_len - window) : min(doclen, (sid+1)*context_len)]
                    doc_seg = doc_seg.strip()
                    if len(doc_seg) == 0: continue
                    emb_seg = get_embedding(doc_seg, engine="text-embedding-ada-002") 
                    embedding.append([docid, sid, emb_seg, doc_seg])

            else:
                emb = get_embedding(single_doc, engine="text-embedding-ada-002") # context length 8k
                embedding.append([docid, 0, emb, single_doc])

        df = pd.DataFrame(embedding, columns=['docid', 'segid', 'embedding', 'text'])        
        df.to_csv(emb_path, index=False, encoding='utf-8')
    
    return df 

# 두 임베딩간 유사도 계산
def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

# 질문을 임베딩하고, 유사도 높은 탑3 자료
def return_answer_candidate(query):
    query_embedding = get_embedding(
        query,
        engine="text-embedding-ada-002"
    )
    # 입력된 질문과 각 문서의 유사도
    DOC_EMB['similarity'] = DOC_EMB['embedding'].apply(lambda x: cos_sim(np.array(query_embedding), np.array(x)))
    # 유사도 높은 순으로 정렬
    top3 = DOC_EMB.sort_values("similarity", ascending=False).head(3)
    return top3


def create_prompt(query, ref_docs):
    # 질문과 가장 유사한 문서 3개 가져오기
    
    system_message = f"""
    너는 주어진 문서를 참고해서, 자세하게 대답해줘.
    문서내용:
    문서1: """ + str(ref_docs.iloc[0]['text']) + """
    문서2: """ + str(ref_docs.iloc[1]['text']) + """
    문서3: """ + str(ref_docs.iloc[2]['text']) + """
    한국어로 답변해주고, 문서에 기반에서 정확한 답을 해줘
    """

    user_message = f"""User question: "{str(query)}". """

    messages =[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    return messages

# 완성된 질문에 대한 답변 생성
def generate_response(messages):
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.4,
        max_tokens=500)
    print(result.choices[0].message.content)
    return result.choices[0].message.content 


def main():
    if "past" not in st.session_state:
        st.session_state["past"] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []    
    if 'reference' not in st.session_state:
        st.session_state['reference'] = []



    with st.form('form', clear_on_submit=True): # streamlit 구성 
        user_input = st.text_input('물어보세요!', '', key='input')
        submitted = st.form_submit_button('Send')

    if submitted and user_input:
        query = user_input # TODO query rephrase 
        ref_docs = return_answer_candidate(user_input)
        prompt = create_prompt(query, ref_docs)
        chatbot_response = generate_response(prompt)
        st.session_state["past"].append(user_input)
        st.session_state["generated"].append(chatbot_response)
        st.session_state["reference"].append(ref_docs)

    if st.session_state["generated"]:
        for i in reversed(range(len(st.session_state["generated"]))):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i) + '_gpt')            
            if len(st.session_state["reference"][i]) > 0:
                message(st.session_state["reference"][i], key=str(i) + '_ref')            


if __name__=="__main__":
    # vscode 실행시 lanuch.json 확인할 것. steramlit debug로 돌려야 함. 
    DOC_EMB = load_doc_embedding()
    main()
