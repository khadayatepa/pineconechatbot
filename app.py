import streamlit as st
import os
import pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeStore
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# App title
st.title("📄🤖 Chat with Your PDF (Pinecone Edition)")

# API Key Inputs
openai_api_key = st.text_input("🔐 Enter your OpenAI API Key", type="password")
pinecone_api_key = st.text_input("🌲 Enter your Pinecone API Key", type="password")
pinecone_env = st.text_input("📍 Pinecone Environment (e.g. 'gcp-starter')")

if not (openai_api_key and pinecone_api_key and pinecone_env):
    st.warning("Please enter all API keys and environment.")
    st.stop()

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
index_name = "doc-chat-index"

# PDF Upload
uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")

if uploaded_file:
    # Save locally
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"✅ Uploaded: {uploaded_file.name}")

    # Load & Split
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Create Embeddings and Index in Pinecone
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Check if index exists
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=1536)

    vectorstore = PineconeStore.from_documents(chunks, embeddings, index_name=index_name)

    # Ask Questions
    query = st.text_input("💬 Ask a question about your document")

    if query:
        with st.spinner("🔍 Searching..."):
            docs_similar = vectorstore.similarity_search(query, k=3)
            llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=docs_similar, question=query)
            st.success("💡 Answer:")
            st.write(answer)
