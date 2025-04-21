# langchain # - 뭔가 성능이 떨어짐? 왠지 모르겠음. 
https://python.langchain.com/docs/tutorials/rag/
pip install -qU "langchain[openai]" # chatmodel - openai
pip install -qU langchain-huggingface
pip install -qU langchain-chroma # vector store 
<!-- pip install -qU langchain-ollama # embedding model - llama3  -->

# docker run 
docker run --gpus all -it --shm-size=13g -v ./:/app {image name} /bin/bash