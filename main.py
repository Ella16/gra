import faiss
import numpy as np
import rag
faiss_gpu_resource = faiss.StandardGpuResources() 
faiss_db_dir = 'faiss/'
def update_db(data:list, index_name:str='abc_news.faiss'):
    dim = 1536 # openai, 768 for the other offline models 
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    
    encoded_list= []
    for doc in data:
        # domain, retriver = rag.domain_classifier(query) # TODO
        encoded = rag.make_embedding(doc)
        encoded_list.append(encoded)
        
    faiss.normalize_L2(encoded_list)
    index.add_with_ids(encoded_list, np.array(range(0, len(encoded_list))))
    faiss.write_index(index, os.path.join(faiss_db_dir, index_name))
    
def infer(index_name='abc_news.faiss'):
    index = faiss.read_index(index_name)
    index = faiss.index_cpu_to_gpu(faiss_gpu_resource, 0, index) # single gpu
    k = 5
    docs = ['ask me: hi',
'suask me: nyou',
'task me: his is test',
'forask me: ',
'fask me: or',]
    while True:
        query_user = input("ask me: ")
        # domain, retriver = rag.domain_classifier(query_user)  # TODO
        query_re = rag.rephrase_query(query_user)
        encoded = rag.make_embedding(query_re)

        distances, indices = index.search(np.array([encoded]).astype(np.float32), k)
        # ref_docs = rag.return_answer_candidate(query_re)
        ref_docs = [[i, docs[i]] for i in indices[0]]
        prompt = rag.create_prompt(query_re, ref_docs)
        response_agent = rag.generate_response(prompt)
        print(response_agent)
            

if __name__=="__main__":
    import pandas as pd
    import os
    data_dir = './data/250415/'
    file = 'qa-250422-processed.csv'
    qa = pd.read_csv(os.path.join(data_dir, file))
    for col in ['src_processed', 'ref', 'Q', 'A_processed']:
        update_db(qa[col].unique().tolist(), index_name=f'qa-{col}.faiss')
        print(col)
    # update_db()
    # infer()

# # FAISS로 top-k 검색
# index = faiss.IndexFlatL2(query_embedding.shape[1])
# index.add(docs_embedding)
# _, I = index.search(query_embedding, k=5)
# top_docs = [docs[i] for i in I[0]]

# # 3. 생성 모델 호출 (Ko-LLM 등)
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")
# model = AutoModelForCausalLM.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")

# prompt = f"질문: {query}\n참고 문서: {' '.join(top_docs)}\n답변:"
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids
# output = model.generate(input_ids, max_new_tokens=256)
# print(tokenizer.decode(output[0], skip_special_tokens=True))

