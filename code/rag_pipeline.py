"""
RAG pipeline implementation.
Handles model loading, embedding generation, and retrieval-augmented generation.
"""
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import os

# Configuration
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Path to your GGUF model. Update this after downloading a model.
MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" 

def load_llm():
    """
    Loads the quantized Llama model using llama-cpp-python.
    Returns:
        LlamaCpp: The loaded LLM instance.
    """
    # Ensure model exists or handle error
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please download a GGUF model.")

    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=2048,
        temperature=0.5,
        max_tokens=512,
        top_p=1,
        verbose=True,
        f16_kv=True  # specific to llama-cpp-python for performance
    )
    return llm

def get_rag_chain():
    """
    Creates and returns the RetrievalQA chain.
    Returns:
        RetrievalQA: The configured RAG chain.
    """
    # Load embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
                                       model_kwargs={'device': 'cpu'})
    
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(f"Vector store not found at {DB_FAISS_PATH}. Run ingest.py first.")
        
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    llm = load_llm()
    
    # Custom prompt for Sanskrit context
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Answer in Sanskrit or English as requested.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    return qa_chain
