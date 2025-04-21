import os
import pandas as pd
import numpy as np
import ast
import time 

from openai import OpenAI

import sys 
sys.append('../')
import keys as keys # Set OpenAI API key 
client = OpenAI(api_key=keys.openai.api_key, organization=keys.openai.organization)

import streamlit as st
import streamlit_chat as chat

class Bubble: # bubble types 
    query_user = "query_user"
    query_rephrased = "query_rephrased"
    response_agent = "response_agent"
    reference = "reference"

sess = st.session_state  # init bubbles
sess[Bubble.query_user] = []
sess[Bubble.query_rephrased] = []
sess[Bubble.response_agent] = []
sess[Bubble.reference] = []
QA_ID = 0

# offline 일때 대비
off_qa = pd.read_csv('demo-qa.csv')
off_ref = pd.read_csv('demo-ref.csv')

from rag import load_doc_embedding, make_embedding 
data_dir = './data/processed/250324_demo/'
emb_path = './embedding/embedding-250324_demo.csv'
DOC_EMB = load_doc_embedding(data_dir, emb_path)

GPT_MODEL = "gpt-4o-mini"
EMBEDDING_ENGINE = "text-embedding-ada-002"
SIMILARITY_THRESHOLD = 0.4 #0.81

def return_answer_candidate(query):
    query_embedding = make_embedding(query)
    DOC_EMB['similarity'] = DOC_EMB['embedding'].apply(lambda x: cos_sim(np.array(query_embedding), np.array(x)))
    top3 = DOC_EMB[DOC_EMB.similarity > SIMILARITY_THRESHOLD].sort_values("similarity", ascending=False).head(3)
    return top3

def main(ONLINE=True):
    global QA_ID
    print(f'#{QA_ID} QA')
    query_user = None
    # chain = create_chain()
    with st.form('form', clear_on_submit=True): # streamlit 구성 
        query_user = st.text_input('물어보세요!', '', key='input')
        submitted = st.form_submit_button('Send')
    
    if submitted:
        with st.status("답변 작성 중...", expanded=True) as status:
            if not ONLINE or query_user == '':    
                time.sleep(1)

                if query_user != "":
                    qa_id = np.random.random_integers(0, 4) # 데모 상황에서 임의의 것을 입력했을 때 순서 벗어나지 않게 하려고 
                else:
                    qa_id = QA_ID
                    QA_ID += 1
                row_qa = off_qa[off_qa.qa_id == qa_id]
                query_user = row_qa['query'].iloc[0]
                query_re = row_qa.reph.iloc[0]

                response_agent = row_qa.agent.iloc[0]
                ref_docs = off_ref[off_ref.qa_id == qa_id]
                
            elif query_user:
                query_re = rephrase_query(query_user)
                ref_docs = return_answer_candidate(query_re)
                prompt = create_prompt(query_re, ref_docs)
                response_agent = generate_response(prompt)
                QA_ID += 1
                # response_agent = chain.run(query_user)

            sess[Bubble.query_user].append(query_user)
            sess[Bubble.query_rephrased].append(query_re)
            sess[Bubble.response_agent].append(response_agent)
            sess[Bubble.reference].append(ref_docs)
            
            status.update(label="완료!", state="complete", expanded=False)
       

    if sess[Bubble.response_agent]:

        for i in reversed(range(len(sess[Bubble.response_agent] ))): # 마지막 꺼만 쓰는게 아니라 매번 history를 다시 씀
            brief = "내부 DB에 없는 내용이네요. 하지만 최선을 다해 도와드릴게요!"
            refs = ""    
            if len(sess[Bubble.reference][i]) > 0:
                brief = f"질문과 관련된 문서를 {len(sess[Bubble.reference][i])}개 찾았어요!"
                avg_score = []
                doc_names = []
                for doc_id, doc in sess[Bubble.reference][i].iterrows():
                    avg_score.append(doc['similarity'])
                    doc_names.append(doc['doc_path'].split('/')[-1])
                    refs += f"**DOC #{doc['doc_id']}: {doc['similarity']:.4f} {doc['doc_path'].split('/')[-1]}**\n"
                    refs += doc["text"].replace('\n\n', '\n').replace('\n\n', '\n') +"\n\n"
                    refs += '**=========================================**\n\n'
                avg_score = sum(avg_score)/len(avg_score)
                if not ONLINE and avg_score < SIMILARITY_THRESHOLD:
                    brief = "내부 DB에 없는 내용이네요. 하지만 최선을 다해 도와드릴게요!"
                    sess[Bubble.response_agent][i] = off_qa.loc[off_qa.qa_id==QA_ID-1].agent_general.iloc[0]
                    refs = ""
                else:
                    doc_names = '\n'.join(doc_names)
                    brief += f"\n\n**평균 점수: {avg_score * 100:.2f}**"
                    brief += f"\n{doc_names}"

            chat.message(sess[Bubble.query_user][i], is_user=True, key=str(i) + '_user', avatar_style="croodles")
            chat.message(sess[Bubble.query_rephrased][i], is_user=True,  key=str(i) + '_rephrased', avatar_style="croodles")
            chat.message(brief, key=str(i) + '_ref_brief', avatar_style="thumbs")      
            chat.message(sess[Bubble.response_agent][i], key=str(i) + '_agent',avatar_style="thumbs")            
            if len(refs)>0:
                chat.message(refs, key=str(i) + '_ref', avatar_style="icons")      
                      
            

# if __name__=="__main__":
#     main(ONLINE=False)  #여기서 돌리면 계속 main을 로딩하는 이슈 있음.. 아마 스트림릿 세팅때문에 그런가봄? run.py로 돌려야함. 
#     # main(ONLINE=True)
#     print()
    