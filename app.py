import pinecone
from langchain_community.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Pinecone as LangchainPinecone

# Initialize Pinecone
pinecone.init(api_key="your_api_key", environment="us-west1-gcp")

# Load and process the uploaded file
file_path = "uploaded.txt"
try:
    with open(file_path, encoding='ISO-8859-1') as f:  # Handle potential encoding issues
        text = f.read()
except Exception as e:
    print(f"Error loading file: {e}")

# Split the text into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents([text])

# Create embeddings using OpenAI's embeddings model (or your own model)
embed_model = OpenAIEmbeddings()

# Create or connect to the Pinecone index
index_name = "your_index_name"
index = pinecone.Index(index_name)

# Initialize the Langchain Pinecone vectorstore with the correct Pinecone index
vectorstore = LangchainPinecone.from_documents(
    documents=split_docs,
    embedding=embed_model,
    index_name=index_name
)

# Add texts to the Pinecone index
pinecone_index = vectorstore._index
pinecone_index.add_texts([doc['text'] for doc in split_docs])  # Assuming your docs have a 'text' key

print("Documents have been indexed successfully!")
