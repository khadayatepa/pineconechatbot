import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
import os

# UI Setup
st.title("Chat with Your Document (Pinecone + LangChain + OpenAI)")

# Upload the file
uploaded_file = st.file_uploader("Upload a .txt file", type="txt")

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-pinecone-api-key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "docchat-index"

if uploaded_file:
    # Save uploaded file temporarily
    with open("uploaded.txt", "wb") as f:
        f.write(uploaded_file.read())

    # Load & Split Document
    loader = TextLoader("uploaded.txt", autodetect_encoding=True)  # 🛠️ fixed encoding issue
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)

    # Create embeddings & upload to Pinecone
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    vectorstore = Pinecone.from_documents(split_docs, embeddings, index_name=index_name)

    # Chat input
    user_query = st.text_input("Ask a question about the uploaded document:")

    if user_query:
        retriever = vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(user_query)

        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
        response = llm.invoke(f"Answer this based on the document: {user_query}")

        st.markdown("### Response")
        st.write(response.content)
