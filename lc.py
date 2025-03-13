def load_models():
    import os
    import keys
    os.environ["OPENAI_API_KEY"] = keys.openai.api_key

    from langchain.chat_models import init_chat_model
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")

    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(model="llama3")

    from langchain_chroma import Chroma

    vector_store = Chroma(embedding_function=embeddings)

    return llm, embeddings, vector_store

# pip install langchain langgraph langchain_community  
# pip install unstructured python-magic python-magic-bin
from langchain import hub
from langchain_community.document_loaders import DirectoryLoader, TextLoader #WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

import os

# Load and chunk contents of the blog
dir = 'data'
loader = DirectoryLoader(dir, glob="**/*.txt", 
                         show_progress=True, #use_multithreading=True,
                         loader_kwargs={'encoding':'utf-8'})
docs = loader.load()


chunk_size = 1000
chunk_overlap = chunk_size//10
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
all_splits = text_splitter.split_documents(docs)


# -------------------------------------------------------------------

llm, embeddings, vector_store = load_models()


# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
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