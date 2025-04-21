
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

query = input("ask me: ")
domain, retriver = domain_classifier(query)  # 위에서 설명한 대로


# 2. 임베딩 + 검색
def load_offline_embedding():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('jhgan/ko-sbert-sts')

import faiss
import numpy as np
emb_model = 
query_embedding = model.encode([query])
docs_embedding = ...  # 문서들 임베딩 미리 저장

# FAISS로 top-k 검색
index = faiss.IndexFlatL2(query_embedding.shape[1])
index.add(docs_embedding)
_, I = index.search(query_embedding, k=5)
top_docs = [docs[i] for i in I[0]]

# 3. 생성 모델 호출 (Ko-LLM 등)
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")
model = AutoModelForCausalLM.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")

prompt = f"질문: {query}\n참고 문서: {' '.join(top_docs)}\n답변:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
output = model.generate(input_ids, max_new_tokens=256)
print(tokenizer.decode(output[0], skip_special_tokens=True))

