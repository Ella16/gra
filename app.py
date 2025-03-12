import os
import pandas as pd
import numpy as np
import ast
from openai import OpenAI
import streamlit as st
import streamlit_chat as chat
import time 

import keys # Set OpenAI API key 
client = OpenAI(api_key=keys.openai.api_key, organization=keys.openai.organization)

DOC_EMB = pd.DataFrame(columns=['docid', 'segid', 'embedding', 'text'])        
GPT_MODEL = "gpt-4o-mini"
EMBEDDING_ENGINE = "text-embedding-ada-002"

def make_embedding(text):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=EMBEDDING_ENGINE).data[0].embedding

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
                    emb_seg = make_embedding(doc_seg) 
                    embedding.append([docid, sid, emb_seg, doc_seg])

            else:
                emb = make_embedding(single_doc) # context length 8k
                embedding.append([docid, 0, emb, single_doc])

        df = pd.DataFrame(embedding, columns=['docid', 'segid', 'embedding', 'text'])        
        df.to_csv(emb_path, index=False, encoding='utf-8')
    
    return df 

def cos_sim(A, B): # TODO 다른 로직? 
    return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))

def return_answer_candidate(query, th_similarity=0.89):
    query_embedding = make_embedding(query)
    DOC_EMB['similarity'] = DOC_EMB['embedding'].apply(lambda x: cos_sim(np.array(query_embedding), np.array(x)))
    top3 = DOC_EMB[DOC_EMB.similarity > th_similarity].sort_values("similarity", ascending=False).head(3)
    return top3

def create_prompt(query, ref_docs):
    if len(ref_docs) > 0:
        system_message = f"""
        너는 주어진 문서를 참고해서, 자세하게 대답해줘.
        문서내용:\n"""
        for doc_id, doc in enumerate(ref_docs):
            system_message += f'''문서{doc_id}: """{str(doc['text'])}"""\n'''
        system_message +="\n한국어로 답변해주고, 문서에 기반에서 정확한 답을 해줘"

    else:
        system_message = f"""사용자의 질문에 자세하게 대답해줘."""

    user_message = f"""User question: "{str(query)}". """
    messages =[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    return messages

# 완성된 질문에 대한 답변 생성
def generate_response(messages):
    result = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=500)
    print(result.choices[0].message.content)
    return result.choices[0].message.content 

def fake_response():
    user_input = '자 질문을 해보자'
    chat_response = '챗봇이 답변을 했어여'
    ref_docs = DOC_EMB[:3]
    
    st.session_state["past"].append(user_input)
    st.session_state["generated"].append(chat_response)
    st.session_state["reference"].append(ref_docs)



def main(ONLINE=True):

    if "past" not in st.session_state:
        st.session_state["past"] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []    
    if 'reference' not in st.session_state:
        st.session_state['reference'] = []
    
    with st.form('form', clear_on_submit=True): # streamlit 구성 
        user_input = st.text_input('물어보세요!', '', key='input')
        submitted = st.form_submit_button('Send')
    if not ONLINE:    
        if submitted:
            with st.status("답변 작성 중...", expanded=True) as status:
                time.sleep(3)
                fake_response()
                status.update(label="완료!", state="complete", expanded=False)
    else:
        if submitted and user_input:
            query = user_input # TODO query rephrase 
            ref_docs = return_answer_candidate(user_input)
            prompt = create_prompt(query, ref_docs)
            chat_response = generate_response(prompt)
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(chat_response)
            st.session_state["reference"].append(ref_docs)
        

    if st.session_state["generated"]:
        for i in reversed(range(len(st.session_state["generated"]))):
            chat.message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            chat.message(st.session_state["generated"][i], key=str(i) + '_gpt')            
            if len(st.session_state["reference"][i]) > 0:
                refs = ""    
                for ref_id, r in st.session_state["reference"][i].iterrows():
                    refs += f"**REFRENCE DOC #{ref_id+1}**\n" + r["text"].replace('\n\n', '\n').replace('\n\n', '\n') +"\n\n"
                chat.message(refs, key=str(i) + '_ref')            


if __name__=="__main__":
    # vscode 실행시 lanuch.json 확인할 것. steramlit debug로 돌려야 함. 
    DOC_EMB = load_doc_embedding()
    ONLINE=False
    main(ONLINE)
