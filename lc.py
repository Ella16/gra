import os
import keys
os.environ["OPENAI_API_KEY"] = keys.openai.api_key

from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# HF_MODEL = "lmms-lab/llama3-llava-next-8b", #??? 읽어지지가 않음.  # https://github.com/kongds/E5-V
# HF_MODEL = 'jhgan/ko-sroberta-nli'
# HF_MODEL = 'davidkim205/komt-mistral-7b-v1'
# HF_MODEL = 'arnir0/Tiny-LLM' # 20MB ! 진짜 작네! 영어만됨. 전체 돌아가는거 확인하는 용도로 쓸만함. pip install sentencepiece
HF_MODEL = 'beomi/llama-2-ko-7b' # 15GB - 이정도만 되도 cpu에 띄우기 무거움  korean available 
OPENAI_MODEL = "gpt-4o-mini"
hf = True
def load_models():

    from langchain.chat_models import init_chat_model
    if not hf:
        from langchain_openai import OpenAIEmbeddings
        llm = init_chat_model(OPENAI_MODEL, model_provider="openai")
        embeddings= OpenAIEmbeddings(model=OPENAI_MODEL)
    else:
        # 이런건들 띄우려면 GPU머신으로 하는게 나을듯. 너무 느림. 일단 API 쓴다. 
        # from langchain_ollama import OllamaEmbeddings
        # embeddings = OllamaEmbeddings(model="llama3") # 뭘 귀찮게 받아야함 ㅋㅋㅋ 안받기로 결정 
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # )
        # model = AutoModelForCausalLM.from_pretrained(HF_MODEL, quantization_config=bnb_config)
        model = AutoModelForCausalLM.from_pretrained(HF_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

        llm = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=500,
        )
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            cache_folder='./ckpts',
            model_name=HF_MODEL,
            multi_process=True,
            # model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity 
            )
                                        

    from langchain_chroma import Chroma
    vector_store = Chroma(embedding_function=embeddings,
                        persist_directory="./chroma_langchain_db")

    return llm, embeddings, vector_store
llm, embeddings, vector_store = load_models()


# pip install langchain langgraph langchain_community  
# pip install unstructured python-magic python-magic-binfrom langchain import hub
from langchain_community.document_loaders import DirectoryLoader, TextLoader #WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import os
def add_doc_to_db():
    dir = 'data'
    loader = DirectoryLoader(dir, glob="**/*.txt", 
                            show_progress=True, #use_multithreading=True,
                            loader_kwargs={'encoding':'utf-8'})
    docs = loader.load() #  TODO 엄청 오래걸림. 왜? 

    chunk_size = 1000
    chunk_overlap = chunk_size//10

    from transformers import AutoTokenizer
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                    AutoTokenizer.from_pretrained(HF_MODEL), # use the same tokenizer as embeeding model
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap,
                    add_start_index=True,
                    strip_whitespace=True, )
    all_splits = text_splitter.split_documents(docs)
    # Document(metadata={'source': 'data\\디지털의료제품법시행규칙.txt'},
    #  page_content='디지털의료제품법 시행규칙 ...')

    vector_store.add_documents(documents=all_splits)
    return vector_store

vector_store = add_doc_to_db()
# -------------------------------------------------------------------
from langchain import hub
prompt = hub.pull("rlm/rag-prompt") # 가장 최신 버전의 rag용 프람트 받는 function 
# input_variables=['context', 'question']
#  metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 
# 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'} 
# messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'],
#  template="You are an assistant for question-answering tasks. 
# Use the following pieces of retrieved context to answer the question.
#  If you don't know the answer, just say that you don't know. 
# Use three sentences maximum and keep the answer concise.
# \nQuestion: {question} \nContext: {context} \nAnswer:"))]


class State(TypedDict): # chat state
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "What is Task Decomposition?"})
for step in graph.stream(
    {"question": "What is Task Decomposition?"}, stream_mode="updates"
    ):
    print(f"{step}\n\n----------------\n")

print(response["answer"])
print()