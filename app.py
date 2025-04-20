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
st.title("📚 Chat with your PDF using OpenAI + Pinecone")

# --- API Key Inputs ---
openai_api_key = st.text_input("🔐 OpenAI API Key", type="password")
pinecone_api_key = st.text_input("🌲 Pinecone API Key", type="password")

# --- Index Configuration ---
index_name = "openaitext-embedding-3-large"  # Your Pinecone index name
pinecone_region = "us-east-1"  # Match your Pinecone dashboard region

# --- Start if Keys are Present ---
if openai_api_key and pinecone_api_key:

    # --- Setup Pinecone Client ---
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    # --- Upload and Process PDF ---
    uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=openai_api_key
        )

        # --- Vectorstore with Pinecone ---
        vectorstore = LangchainPinecone.from_documents(
            documents=split_docs,
            embedding=embeddings,
            index_name=index_name
        )

        st.success("✅ PDF uploaded and indexed!")

        # --- Ask Question ---
        query = st.text_input("💬 Ask a question about your PDF")
        if query:
            with st.spinner("💡 Generating answer..."):
                results = vectorstore.similarity_search(query)
                llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
                chain = load_qa_chain(llm, chain_type="stuff")
                answer = chain.run(input_documents=results, question=query)

            st.subheader("🔍 Answer")
            st.write(answer)

else:
    st.info("Please enter both API keys above to get started.")
