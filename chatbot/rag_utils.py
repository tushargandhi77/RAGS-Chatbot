import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
ASTRA_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")

# ✅ Load embeddings once (small model)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Lazy-load heavy LLM
def get_llm():
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        huggingfacehub_api_token=HF_TOKEN
    )

# ✅ Lazy-load vector store
def get_vector_store():
    return AstraDBVectorStore(
        embedding=embedding_model,
        collection_name="documents",
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_TOKEN,
    )

def clear_collection():
    try:
        vector_store = get_vector_store()
        vector_store.delete_collection()
        print("DEBUG: Collection 'documents' cleared")
    except Exception as e:
        print(f"Error clearing collection: {e}")
        raise

def process_document(file_path):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"DEBUG: Loaded {len(documents)} pages")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"DEBUG: Split into {len(chunks)} chunks")

        texts = [chunk.page_content for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        vector_store = get_vector_store()
        vector_store.add_texts(texts=texts, ids=ids)
        print(f"DEBUG: Inserted {len(chunks)} chunks into AstraDB")
    except Exception as e:
        print(f"Error processing document: {e}")
        raise

def get_response(query):
    try:
        vector_store = get_vector_store()
        llm = get_llm()

        results = vector_store.similarity_search(query, k=3)
        context = " ".join([doc.page_content for doc in results])

        if not context.strip():
            return "No Information Available"

        prompt = (
            f"<|system|>You are an assistant. Answer strictly based on the provided context. "
            f"If the query cannot be answered using the context, respond with 'No Information Available'. "
            f"<|user|>Context: {context}\n\nQuestion: {query}\nAnswer:<|assistant|>"
        )

        response = llm.invoke(prompt).strip()
        return response if response and "No Information Available" not in response else "No Information Available"

    except Exception as e:
        print(f"Error generating response: {e}")
        return "No Information Available"
