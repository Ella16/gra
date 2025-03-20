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
