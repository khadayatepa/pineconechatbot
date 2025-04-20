import os
import streamlit as st
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env (if present)
load_dotenv()

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")

# Set the new index name (3072-dimensional)
index_name = "openaitext-embedding-3-large"

# Initialize the Pinecone Index
index = pinecone.Index(index_name)

# Initialize the OpenAI embeddings model (ensure it matches your index dimensions, e.g., text-embedding-ada-003)
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Set up the Langchain Pinecone vector store
vectorstore = LangchainPinecone(index, embeddings.embed_query, "text")

# Streamlit UI to upload the PDF document
st.title("Document Q&A Chatbot")

uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save uploaded file
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"✅ {uploaded_file.name} uploaded successfully.")

    # Load and split the document using PyPDFLoader
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Split the document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Add the chunks to Pinecone (index) via the vector store
    vectorstore.add_documents(chunks)

    # User input for a query
    query = st.text_input("💬 Ask a question about the document")

    if query:
        with st.spinner("🧠 Thinking..."):
            matched_docs = vectorstore.similarity_search(query)
            llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=matched_docs, question=query)
            st.success("💡 Answer:")
            st.write(answer)
