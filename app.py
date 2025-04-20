import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Streamlit Title
st.title("🤖 Pinecone + PDF Chatbot")

# Input for API Keys
openai_key = st.text_input("Enter your OpenAI API key", type="password")
pinecone_key = st.text_input("Enter your Pinecone API key", type="password")

if not openai_key or not pinecone_key:
    st.warning("🔑 Please provide both OpenAI and Pinecone API keys.")
    st.stop()

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_key)
index_name = "pdf-chatbot-index"
dimension = 1536  # for OpenAI embeddings

# Create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )

index = pc.Index(index_name)

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

if uploaded_file:
    file_path = os.path.join("data", uploaded_file.name)
    os.makedirs("data", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"Uploaded {uploaded_file.name}")

    # Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embed and add to Pinecone
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = LangchainPinecone(index, embeddings.embed_query, "text")
    vectorstore.add_documents(chunks)
    st.success("✅ Document embedded and stored in Pinecone.")

# Question
query = st.text_input("💬 Ask something from the document")

if query:
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = LangchainPinecone(index, embeddings.embed_query, "text")
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_key)
    docs = vectorstore.similarity_search(query)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=query)
    st.success("💡 Answer:")
    st.write(answer)
