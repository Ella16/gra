import os
import gra.keys as keys
os.environ["OPENAI_API_KEY"] = keys.openai.api_key

# pip install langchain langgraph langchain_community  
# pip install unstructured python-magic python-magic-binfrom langchain import hub
from langchain_community.document_loaders import DirectoryLoader#, TextLoader #WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from transformers import AutoTokenizer
import os
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# HF_MODEL = "lmms-lab/llama3-llava-next-8b", #??? 읽어지지가 않음.  # https://github.com/kongds/E5-V
# HF_MODEL = 'jhgan/ko-sroberta-nli'
# HF_MODEL = 'davidkim205/komt-mistral-7b-v1'
# HF_MODEL = 'arnir0/Tiny-LLM' # 20MB ! 진짜 작네! 영어만됨. 전체 돌아가는거 확인하는 용도로 쓸만함. pip install sentencepiece
HF_MODEL = 'beomi/llama-2-ko-7b' # 15GB - 이정도만 되도 cpu에 띄우기 무거움  korean available 
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TICKTOK_EMB_MODEL = 'text-embedding-ada-002' # 얘는 gpt4 tokenzier라서 허깅페이스에서 다운이 안됨 => 돈내고 쓸수 밖에 없구만만
hf = False
llm, embeddings, vector_store = None, None, None

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


# ------------------------------
def load_models(vertor_db_directory):

    from langchain.chat_models import init_chat_model
    if not hf:
        from langchain_openai import OpenAIEmbeddings
        llm = init_chat_model(OPENAI_MODEL, model_provider="openai")
        embeddings= OpenAIEmbeddings(model=OPENAI_TICKTOK_EMB_MODEL)       
        # => 얘 직접받아서 huf 로 읽어도 됨. 그럼 꽁짜. 

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
    vector_store = Chroma(embedding_function=embeddings, persist_directory=vertor_db_directory)
    
    return llm, embeddings, vector_store

def add_doc_to_db(vector_store, dir):    
    loader = DirectoryLoader(dir, glob="**/*.txt", 
                            show_progress=True, #use_multithreading=True,
                            loader_kwargs={'encoding':'utf-8'})
    docs = loader.load() #  TODO 엄청 오래걸림. 왜? 

    
    chunk_size = 1000
    chunk_overlap = chunk_size//10   

    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size, 
                        chunk_overlap=chunk_overlap,
                        add_start_index=True,
                        strip_whitespace=True, )
    all_splits = text_splitter.split_documents(docs)
    # Document(metadata={'source': 'data\\디지털의료제품법시행규칙.txt'},
    #  page_content='디지털의료제품법 시행규칙 ...')
    vector_store.add_documents(documents=all_splits)
    return vector_store



from langchain.chains import RetrievalQA
def create_chain():
    OPENAI_MODEL = 'gpt-4o-mini'
    data_dir = './data/processed/250324_demo/'
    vertor_db_directory =f"./chroma_langchain_db/{OPENAI_MODEL}/250324_demo/"
    llm, embeddings, vector_store = load_models(vertor_db_directory)
    vector_store = add_doc_to_db(vector_store, data_dir)

    retriever = vector_store.as_retriever(search_type='similarity',
                                           search_kwargs={
                                               "k": 3, # Select top k search results
                                            } 
                                        )

    rag_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)
    
    return rag_chain

# -------------------------------------------------------------------
class State(TypedDict): # chat state
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State, K=3):
    retrieved_docs = vector_store.similarity_search(state["question"], k=K)
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


if __name__=='__main__':
    
    # data_dir = './data/processed/250324_demo/'
    # if not hf:
    #     # vertor_db_directory =f"./chroma_langchain_db/{OPENAI_MODEL}/"
    #     vertor_db_directory =f"./chroma_langchain_db/{OPENAI_MODEL}/250324_demo/"
    # else:
    #     vertor_db_directory =f"./chroma_langchain_db/{HF_MODEL.replace('/', '_')}/"

    # llm, embeddings, vector_store = load_models(vertor_db_directory)
    # vector_store = add_doc_to_db(data_dir)

    # graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    # graph_builder.add_edge(START, "retrieve")
    # graph = graph_builder.compile()

    # response = graph.invoke({"question": "What is Task Decomposition?"})
    # for step in graph.stream(
    #     {"question": "What is Task Decomposition?"}, stream_mode="updates"
    #     ):
    #     print(f"{step}\n\n----------------\n")

    # print(response["answer"])
    # print()

    chain = create_chain()
    a = create_chain()
    query = '니자티딘 이 뭐야 '
    result = a.invoke({"question": query})
    print()