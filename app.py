import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# UI Setup
st.title("🔍 Chat with Your Docs using Pinecone + LangChain")

# Input fields
openai_api_key = st.text_input("🔑 OpenAI API Key", type="password")
pinecone_api_key = st.text_input("🌲 Pinecone API Key", type="password")
index_name = st.text_input("📦 Pinecone Index Name (e.g. openaitext-embedding-3-large)", value="openaitext-embedding-3-large")
uploaded_file = st.file_uploader("📁 Upload a text file")

if openai_api_key and pinecone_api_key and uploaded_file:
    st.success("✅ All credentials and file uploaded. Processing...")

    # Ensure LangChain sees the Pinecone API key
    os.environ["PINECONE_API_KEY"] = pinecone_api_key

    # Save file temporarily
    with open("uploaded.txt", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load & Split Document
    loader = TextLoader("uploaded.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)

    # OpenAI Embeddings (text-embedding-3-large = 3072 dimensions)
    embed_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_api_key
    )

    # Pinecone initialization
    pc = Pinecone(api_key=pinecone_api_key)

    # Create index if not exists
    if index_name not in pc.list_indexes().names():
        st.info(f"Creating index: {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        st.success(f"Index `{index_name}` created.")

    # Embed and store documents in Pinecone
    st.info("Embedding and uploading chunks to Pinecone...")
    vectorstore = Pinecone.from_documents(
        documents=split_docs,
        embedding=embed_model,
        index_name=index_name
    )
    st.success("✅ Documents embedded and uploaded to Pinecone!")

    # Optional: show sample metadata
    st.write("📚 Example chunk:", split_docs[0].page_content[:300])
else:
    st.warning("Please upload a file and enter all required keys.")
