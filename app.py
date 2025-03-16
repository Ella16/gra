import os
import pandas as pd
import numpy as np
import ast
import time 

from openai import OpenAI
import keys # Set OpenAI API key 
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


DOC_EMB = pd.DataFrame(columns=['docid', 'segid', 'embedding', 'text'])        
GPT_MODEL = "gpt-4o-mini"
EMBEDDING_ENGINE = "text-embedding-ada-002"
SIMILARITY_THRESHOLD = 0.85

def generate_response(messages):
    result = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=500)
    print(result.choices[0].message.content)
    return result.choices[0].message.content 


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

def rephrase_query(query):
    system_prompt = """I want you to act as an expert in Query Rephrasing and Query Intent Analysis and perform the task While strictly following the principles and requirements provided below: 
 [Principles]
 1. Anaphora Resolution: Improve sentence clarity by replacing pronouns or demonstratives with more specific words that refer to previously mentioned entities.
 2. Phrase Rephrasing: Enhance the sentence by rephrasing specific phrases, rearranging or substituting them with semantically meaningful synonyms to enhance impact.
 3. Sentence Restructuring for Cause-and-Effect: Emphasize the cause-and-effect relationship by restructuring the sentence, rearranging or highlighting elements to express the    relationship more clearly.
 4. Paraphrasing Complex Sentences: Simplify complex sentences by summarizing and transforming them into a more comprehensible form.
 5. Synonym Replacement: Enhance the sentence by replacing specific words with appropriate synonyms to add diversity and richness in expression.
 6. Context Understanding: Grasp the context from the conversation history to better understand the question and provide accurate answers. Seek clarity and gather necessary       information when sentences are ambiguous.
 7. Clear Question Formulation: Formulate clear and unambiguous questions by including specific information and requirements to effectively convey the intention.
 8. Semantic Consistency: Preserve the original meaning and intention of the query when rephrasing, ensuring that the query's semantic integrity remains intact.
Output must be in Korean.
"""
    user_prompt = f"""Query: "{str(query)}". """
    messages =[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    query = generate_response(messages)
    return query

def return_answer_candidate(query):
    query_embedding = make_embedding(query)
    DOC_EMB['similarity'] = DOC_EMB['embedding'].apply(lambda x: cos_sim(np.array(query_embedding), np.array(x)))
    top3 = DOC_EMB[DOC_EMB.similarity > SIMILARITY_THRESHOLD].sort_values("similarity", ascending=False).head(3)
    return top3


def create_prompt(query, ref_docs):
    if len(ref_docs) > 0:
        system_prompt = f"""
            You're an expert for medicine regulations and legislation.
            Yor're very fluent in Korean and English.
            According to following docs, answer the user query.:\n"""
        for doc_id, doc in ref_docs.iterrows():
            system_prompt += f'''doc#{doc_id}: """{str(doc['text'])}"""\n'''
        system_prompt +="\nAnswer user query in detail; it must be in Korean and must have rationale w.r.t the given docs."
       
    else:
        system_prompt = f"""Answer user query in detail"""

    user_prompt = f"""User Query: "{str(query)}". """
    messages =[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return messages

def main(ONLINE=True):
    query_user = None
    with st.form('form', clear_on_submit=True): # streamlit 구성 
        query_user = st.text_input('물어보세요!', '', key='input')
        submitted = st.form_submit_button('Send')
    
    if submitted:
        with st.status("답변 작성 중...", expanded=True) as status:
            if not ONLINE or query_user == '':    
                time.sleep(3)
                query_user = '자 질문을 해보자'
                query_re = '질문을 시작해 보겠습니다.' # rephrase_query(query_user)
                response_agent = '챗봇이 답변을 했어여'
                
                ref_docs = DOC_EMB.sample(n=1) # sample random doc 
                ref_docs.loc[:, 'similarity'] = 0.9
            
            elif query_user:
                query_re = rephrase_query(query_user)
                ref_docs = return_answer_candidate(query_re)
                prompt = create_prompt(query_re, ref_docs)
                response_agent = generate_response(prompt)

            sess[Bubble.query_user].append(query_user)
            sess[Bubble.query_rephrased].append(query_re)
            sess[Bubble.response_agent].append(response_agent)
            sess[Bubble.reference].append(ref_docs)
            
            status.update(label="완료!", state="complete", expanded=False)

    if sess[Bubble.response_agent]:
        for i in reversed(range(len(sess[Bubble.response_agent]))):
            chat.message(sess[Bubble.query_user][i], is_user=True, key=str(i) + '_user')
            chat.message(sess[Bubble.query_rephrased][i], is_user=True, key=str(i) + '_rephrased')
            chat.message(sess[Bubble.response_agent][i], key=str(i) + '_agent')            
            if len(sess[Bubble.reference][i]) > 0:
                refs = ""    
                for doc_id, doc in sess[Bubble.reference][i].iterrows():
                    refs += f"**REF DOC #{doc_id+1}: {doc['similarity']}**\n"
                    refs += doc["text"].replace('\n\n', '\n').replace('\n\n', '\n') +"\n\n"
                chat.message(refs, key=str(i) + '_ref')            


if __name__=="__main__":
    # vscode 실행시 lanuch.json 확인할 것. steramlit debug로 돌려야 함. 
    DOC_EMB = load_doc_embedding()
    # ONLINE=False
    ONLINE=True
    main(ONLINE)
