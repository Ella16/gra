import os
import pandas as pd
import numpy as np
import ast
import time 

from openai import OpenAI
import keys # Set OpenAI API key 
client = OpenAI(api_key=keys.openai.api_key, organization=keys.openai.organization)

GPT_MODEL = "gpt-4o-mini"
EMBEDDING_ENGINE = "text-embedding-ada-002"
SIMILARITY_THRESHOLD = 0.85

def domain_classifier(query):
    retriever_dict = {
        "form":None,
        "qa": None,
        "docs": None,
    }
    
    if True:
        domain = "docs"
        retriever = retriever_dict[domain]
        return domain, retriever
    else:
        from transformers import pipeline

        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        result = classifier("What are the symptoms of COVID?", candidate_labels=["medical", "legal", "technical"])
        print(result['labels'][0])  # Best guess


        query = "How do I appeal a court decision?"

        domain = domain_classifier(query)

        
        docs = retriever.retrieve(query)

        response = generator.generate(query, docs)



def load_offline_embedding():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('jhgan/ko-sbert-sts')

def make_embedding(text):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=EMBEDDING_ENGINE).data[0].embedding

def segment_by_paragraph(doc, context_length=1000, tolerance=0.1, stride_para=2):
    paras = doc.strip().split('\n')
    paras = [p.strip() for p in paras if len(p.strip())>0]
    segments = []
    current_segment = []
    current_length = 0
    max_len = context_length * (1 + tolerance)
    for i, signle_para in enumerate(paras):
        paragraph_length = len(signle_para)
        if current_length + paragraph_length <= max_len:
            current_segment.append(signle_para)
            current_length += paragraph_length
        else:
            segments.append("\n".join(current_segment)) 
            current_segment = current_segment[-stride_para:] + [signle_para] # get last 2 para
            current_length = paragraph_length    
    if current_segment: # 마지막 segment 
        segments.append("\n".join(current_segment))
    return segments

def segment(doc, context_len=1000, window_percent=10):
    window = context_len // window_percent # 1/10 를 윈도우로 잡는다. 
    segs = []
    doclen = len(doc)

    if doclen > context_len:
        num_seg = doclen // context_len 
        seg_len = doclen // num_seg + 1
        for sid in range(seg_len):
            doc_seg = doc[max(0, sid*context_len - window) : min(doclen, (sid+1)*context_len)]
            doc_seg = doc_seg.strip()
            if len(doc_seg) == 0: continue
            segs.append([sid, doc_seg])
    else:
        segs.append([0, doc])
    return segs

def make_doc_embedding(doc):
    embedding = []
    segs = segment_by_paragraph(doc)
    for sid, doc_seg in enumerate(segs):
        emb_seg = make_embedding(doc_seg) 
        embedding.append([sid, emb_seg, doc_seg])
    
    return embedding

def load_doc_embedding(data_dir, emb_path):
    doc_names = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.txt')]
    added = doc_names 

    df = pd.DataFrame(columns=['docid', 'segid', 'embedding', 'text', 'doc_path'])
    df_added = pd.DataFrame(columns=['docid', 'segid', 'embedding', 'text', 'doc_path'])
    if os.path.isfile(emb_path):
        df = pd.read_csv(emb_path)
        df['embedding'] = df['embedding'].apply(ast.literal_eval)
        added = set(doc_names) - set(df.doc_path.unique())

    if len(added) > 0:
        doc_names = list(added)
        print(f"add embeddidng for {len(doc_names)} docs", data_dir, emb_path)
        docs = []
        for file in doc_names:
            with open(doc_path, 'r', encoding='utf-8') as f:
                single_doc = f.read()
                docs.append([doc_path, single_doc])

        embeddings = []
        for doc_id, (doc_path, single_doc) in enumerate(docs):
            print('processing...', doc_id, doc_path)
            _doc_emb = make_doc_embedding(single_doc, doc_id)
            _df = pd.DataFrame(_doc_emb, columns =['seg_id', 'embedding', 'text'])
            _df['doc_id'] = doc_id
            _df['doc_path'] = doc_path
            embeddings.append(_df)
            df_added = pd.concat(embeddings) 
            df = pd.concat([df, df_added])            
    
        df.to_csv(emb_path, index=False, encoding='utf-8')
    
    print(f"{__name__} LOAD doc embedding done! num_segs/num_docs = {len(df)}/{len(doc_names)}")
    return df 


def generate_response(messages):
    result = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=500)
    return result.choices[0].message.content 

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


def create_prompt(query, ref_docs:list):
    if len(ref_docs) > 0:
        system_prompt = f"""
You're an expert for medicine regulations and legislation.
Yor're very fluent in Korean and English.
According to following docs, answer the user query.:\n"""
        for doc_id, doc in ref_docs:
            system_prompt += f'''doc#{doc_id}: """{str(doc)}"""\n'''
        system_prompt +="\nAnswer user query in detail; it must be in Korean and must have rationale w.r.t the given docs."
       
    else:
        system_prompt = f"""Answer user query in detail"""

    user_prompt = f"""User Query: "{str(query)}". """
    messages =[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return messages
