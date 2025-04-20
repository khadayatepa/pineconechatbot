import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from pinecone import Pinecone

# --- UI Setup ---
st.title("📚 Chat with your PDF (Pinecone + OpenAI)")

# --- API Keys ---
openai_api_key = st.text_input("🔐 Enter your OpenAI API Key", type="password")
pinecone_api_key = st.text_input("🌲 Enter your Pinecone API Key", type="password")

if not openai_api_key or not pinecone_api_key:
    st.warning("Please enter both API keys to continue.")
    st.stop()

# --- Pinecone Setup ---
pc = Pinecone(api_key=pinecone_api_key)
index_name = "openaitext-embedding-3-large"

try:
    index = pc.Index(index_name)
except Exception as e:
    st.error(f"❌ Could not connect to Pinecone index: {e}")
    st.stop()

# --- Upload PDF ---
uploaded_file = st.file_uploader("📄 Upload a PDF file", type="pdf")
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # --- Load + Split PDF ---
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(pages)

    # --- Create Embeddings ---
    embed_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_api_key
    )

    # --- Store Vectors in Pinecone ---
    vectorstore = LangchainPinecone.from_documents(
        docs,
        embed_model,
        index_name=index_name
    )

    st.success("✅ PDF processed and embedded into Pinecone!")

    # --- Ask Questions ---
    query = st.text_input("💬 Ask a question about the PDF")
    if query:
        with st.spinner("🔍 Searching..."):
            docs_found = vectorstore.similarity_search(query)
            llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=docs_found, question=query)

        st.subheader("💡 Answer")
        st.write(answer)
