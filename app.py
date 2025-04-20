import os
import chardet
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from pinecone.core.serverless import ServerlessSpec

# UI Setup
st.title("🔍 Chat with Your Docs using Pinecone + LangChain")

# Input fields
openai_api_key = st.text_input("🔑 OpenAI API Key", type="password")
pinecone_api_key = st.text_input("🌲 Pinecone API Key", type="password")
index_name = st.text_input("📦 Pinecone Index Name (e.g. openaitext-embedding-3-large)", value="openaitext-embedding-3-large")
uploaded_file = st.file_uploader("📁 Upload a text file")

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result['encoding']

if openai_api_key and pinecone_api_key and uploaded_file:
    st.success("✅ All credentials and file uploaded. Processing...")

    os.environ["PINECONE_API_KEY"] = pinecone_api_key

    # Save uploaded file temporarily
    file_path = "uploaded.txt"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Detect encoding and load content with error handling
    try:
        encoding = detect_encoding(file_path)
        st.info(f"Detected encoding: {encoding}")
        
        # Try opening with detected encoding first
        try:
            with open(file_path, "r", encoding=encoding, errors='replace') as f:
                text = f.read()
        except UnicodeDecodeError:
            # If the detected encoding fails, attempt with 'ISO-8859-1' or 'latin1'
            st.warning(f"❌ Failed to decode using {encoding}. Trying 'ISO-8859-1' encoding.")
            with open(file_path, "r", encoding='ISO-8859-1', errors='replace') as f:
                text = f.read()
        
        documents = [Document(page_content=text)]
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        st.stop()

    # Split the document
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)

    # Ensure no chunk exceeds 1000 characters
    for i, doc in enumerate(split_docs):
        if len(doc.page_content) > 1000:
            st.warning(f"Chunk {i} exceeds 1000 characters, reducing...")
            doc.page_content = doc.page_content[:1000]
    
    # OpenAI Embeddings
    embed_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_api_key
    )

    # Pinecone Initialization (Correctly initializing with LangChain's Pinecone wrapper)
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

    # Upload to Pinecone
    st.info("Embedding and uploading chunks to Pinecone...")
    vectorstore = LangChainPinecone.from_documents(
        documents=split_docs,
        embedding=embed_model,
        index_name=index_name
    )
    st.success("✅ Documents embedded and uploaded to Pinecone!")

    # Show a sample chunk
    st.write("📚 Example chunk:", split_docs[0].page_content[:300])
else:
    st.warning("Please upload a file and enter all required keys.")
