import gradio as gr
# from langchain.chains import ConversationalRetrievalChain
# from langchain_chroma import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.document_loaders import TextLoader
# from langchain.agents import initialize_agent, Tool, AgentType


# def load_documents():
#     loader = TextLoader("sample_documents.txt")  # 텍스트 파일 경로
#     documents = loader.load()
#     return documents

# def create_index(vertor_db_directory):
#     embeddings = OpenAIEmbeddings()    
#     vector_store = Chroma(embedding_function=embeddings, persist_directory=vertor_db_directory)
#     return vector_store

# from utils import load_models, add_doc_to_db

# def create_chain():
#     OPENAI_MODEL = 'gpt-40-mini'
#     data_dir = './data/processed/250324_demo/'
#     vertor_db_directory =f"./chroma_langchain_db/{OPENAI_MODEL}/250324_demo/"
#     llm, embeddings, vector_store = load_models(vertor_db_directory)
#     vector_store = add_doc_to_db(data_dir)

#     retriever = vector_store.as_retriever()

#     rag_chain = ConversationalRetrievalChain.from_llm(llm, retriever)
    
#     return rag_chain

# # Langchain RAG 에이전트 실행 함수
# def rag_agent(query):
#     chain = create_chain()
#     response = chain.run(query)
#     return response

def echo(text):
    return text

# # Gradio 인터페이스 설정
# def gradio_interface():
#     with gr.Blocks() as demo:
#         gr.Markdown("## RAG Agent with Langchain & Gradio")
        
#         with gr.Row():
#             with gr.Column():
#                 query_input = gr.Textbox(label="Ask me anything", placeholder="Type your query here...", lines=2)
#             with gr.Column():
#                 submit =gr.Button(value='Submit')

#         response_output = gr.Textbox(label="Response", interactive=False)
        
#         query_input.submit(echo, inputs=query_input, outputs=response_output) # submit = ctrl + enter
#         submit.click(echo, inputs=query_input, outputs=response_output) # ㄴ 위랑 같은 동작 

#     return demo

# 대화 기록을 계속 누적시키는 함수
def chat_with_history(user_input, history):
    # 사용자 입력과 응답을 기록에 추가
    history.append(f"User: {user_input}")
    
    # 간단한 에이전트로 응답 생성 (여기서는 "I'm thinking about..." 형태로 응답)
    response = f"I'm thinking about: {user_input}"
    history.append(f"Bot: {response}")
    
    # 대화 기록 반환
    return "\n".join(history), history

# Gradio 인터페이스 구성
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## 대화 기록을 계속 누적하는 챗봇")
        
        # 사용자 입력
        user_input = gr.Textbox(label="Your message", placeholder="Type your message...", lines=2)
        
        # 대화 기록을 출력할 텍스트박스
        chat_history = gr.Textbox(label="Conversation History", interactive=False, lines=10)
        
        # 상태 변수 (대화 기록)
        state = gr.State([])  # 빈 리스트로 초기화 (대화 기록 리스트)

        # 사용자가 메시지를 입력할 때마다 기록을 누적하고 출력
        user_input.submit(chat_with_history, inputs=[user_input, state], outputs=[chat_history, state])
    
    return demo

# 실행
if __name__ == "__main__":
    app = gradio_interface()
    app.launch(share=True)
