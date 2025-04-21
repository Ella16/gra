import faiss
import numpy as np
import rag



# def add_faiss(index:faiss.swigfaiss.IndexIDMap, encoded_data, data):
#     if not isinstance(encoded_data, np.array):
#         encoded_data = np.array(encoded_data)
#     index.add_with_ids(encoded_data, np.array(range(0, len(data))))
    
def update_db():
    dim = 1536 # openai, 768 for the other offline models 
    index = faiss.IndexFlatIP(dim)
    
    encoded_list= []
    query_list = []
    for _ in range(5): # 이게 chunked docs이면 됨 
        query = input("ask me: ") 
        domain, retriver = rag.domain_classifier(query) 
        encoded = rag.make_embedding(query)
        encoded_list.append(encoded)
        query_list.append(query)
        
    index.add(np.array(encoded_list).astype(np.float32))
    
    faiss.write_index(index, 'main.faiss')
    
def infer(index_name='abc_news.faiss'):
    index = faiss.read_index(index_name)
    k = 5
    docs = ['ask me: hi',
'suask me: nyou',
'task me: his is test',
'forask me: ',
'fask me: or',]
    while True:
        query_user = input("ask me: ")
        domain, retriver = rag.domain_classifier(query_user) 
        query_re = rag.rephrase_query(query_user)
        encoded = rag.make_embedding(query_re)

        distances, indices = index.search(np.array([encoded]).astype(np.float32), k)
        # ref_docs = rag.return_answer_candidate(query_re)
        ref_docs = [[i, docs[i]] for i in indices[0]]
        prompt = rag.create_prompt(query_re, ref_docs)
        response_agent = rag.generate_response(prompt)
        print(response_agent)
            

if __name__=="__main__":
    # update_db()
    infer()

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

