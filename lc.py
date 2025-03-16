import os
import keys
os.environ["OPENAI_API_KEY"] = keys.openai.api_key

EMBEDDING_MODEL = 'jhgan/ko-sroberta-nli'
OPENAI_MODEL = "gpt-4o-mini"
def load_models():
    from langchain.chat_models import init_chat_model
    llm = init_chat_model(OPENAI_MODEL, model_provider="openai")

    from langchain_openai import OpenAIEmbeddings
    embeddings= OpenAIEmbeddings(model=OPENAI_MODEL)

    # 이런건들 띄우려면 GPU머신으로 하는게 나을듯. 너무 느림. 일단 API 쓴다. 
    # from langchain_ollama import OllamaEmbeddings
    # embeddings = OllamaEmbeddings(model="llama3") # 뭘 귀찮게 받아야함 ㅋㅋㅋ 안받기로 결정 
    # from langchain_huggingface import HuggingFaceEmbeddings
    # embeddings = HuggingFaceEmbeddings(``
    #     cache_folder='./ckpts',
    #     model_name=EMBEDDING_MODEL,
    #     #model_name="lmms-lab/llama3-llava-next-8b", #??? 읽어지지가 않음.  # https://github.com/kongds/E5-V
    #     multi_process=True,
    #     # model_kwargs={"device": "cuda"},
    #     encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity 
    #     )
                                    

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
                    AutoTokenizer.from_pretrained(EMBEDDING_MODEL), # use the same tokenizer as embeeding model
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