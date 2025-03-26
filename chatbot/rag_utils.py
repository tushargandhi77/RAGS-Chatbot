import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
ASTRA_TOKEN = "AstraCS:iknYNcTDbZfiAtDTglfhUiBT:3738b6178b39ba6f985e79e0499248d3b18f86cfbe4112ad4c60071c47c47060"
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")

# Debug: Print the token and endpoint to verify
print(f"DEBUG: Hugging Face token loaded: {HF_TOKEN}")
print(f"DEBUG: Token loaded: {ASTRA_TOKEN}")
print(f"DEBUG: Endpoint loaded: {ASTRA_DB_API_ENDPOINT}")

# Check if tokens are loaded correctly
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN is not set in .env")
if not ASTRA_TOKEN or "AstraCS" not in ASTRA_TOKEN:
    raise ValueError("ASTRA_DB_APPLICATION_TOKEN is not set correctly in .env")

# Initialize Hugging Face models
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token=HF_TOKEN
)

# Initialize AstraDB Vector Store
vector_store = AstraDBVectorStore(
    embedding=embedding_model,
    collection_name="documents",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_TOKEN,
)
print("DEBUG: Successfully connected to AstraDB Vector Store")

# Function to clear the AstraDB collection
def clear_collection():
    try:
        # Declare global at the start of the function before any use
        global vector_store
        vector_store.delete_collection()
        print("DEBUG: Collection 'documents' cleared")
        # Reinitialize the vector store
        vector_store = AstraDBVectorStore(
            embedding=embedding_model,
            collection_name="documents",
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_TOKEN,
        )
        print("DEBUG: Collection 'documents' recreated")
    except Exception as e:
        print(f"Error clearing collection: {e}")
        raise

# Function to process and upload document
def process_document(file_path):
    try:
        # Load document
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"DEBUG: Loaded {len(documents)} pages from PDF at {file_path}")

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"DEBUG: Split into {len(chunks)} chunks")

        # Add chunks to AstraDB Vector Store
        texts = [chunk.page_content for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        vector_store.add_texts(texts=texts, ids=ids)
        print(f"DEBUG: Inserted {len(chunks)} chunks into AstraDB")
    except Exception as e:
        print(f"Error processing document: {e}")
        raise

# Function to retrieve and generate response
def get_response(query):
    try:
        results = vector_store.similarity_search(query, k=3)
        print(f"DEBUG: Query: {query}")
        print(f"DEBUG: Retrieved {len(results)} documents")


        for i, doc in enumerate(results):
            print(f"DEBUG: Doc {i}: {doc.page_content[:100]}...")

        context = " ".join([doc.page_content for doc in results])


        if not context.strip():
            print("DEBUG: No relevant context found")
            return "No Information Available"

        # Enhanced prompt to enforce context-only responses:
        prompt = (
            f"<|system|>You are an assistant. Answer strictly based on the provided context. "
            f"If the query cannot be answered using the context, respond with 'No Information Available'. "
            f"<|user|>Context: {context}\n\nQuestion: {query}\nAnswer:<|assistant|>"
        )
        # print(f"DEBUG: Prompt sent to LLM: {prompt[:200]}...")

        # Generate response
        response = llm.invoke(prompt).strip()
        # print(f"DEBUG: Raw LLM response: {response}")

        return response if response and "No Information Available" not in response else "No Information Available"

    except Exception as e:
        print(f"Error generating response: {e}")
        return "No Information Available"