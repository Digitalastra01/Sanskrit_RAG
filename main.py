import streamlit as st
from rag_pipeline import get_rag_chain
import os

st.set_page_config(page_title="Sanskrit RAG", layout="wide")

st.title("📜 Sanskrit Document RAG System")
st.markdown("Query your Sanskrit documents using a local CPU-based LLM.")

# Sidebar for setup
with st.sidebar:
    st.header("System Status")
    if os.path.exists("vectorstore/db_faiss"):
        st.success("Vector Store Found")
    else:
        st.error("Vector Store Not Found. Run ingest.py")
        
    model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" # Should match rag_pipeline.py
    if os.path.exists(model_path):
        st.success(f"Model Found: {os.path.basename(model_path)}")
    else:
        st.error("Model Not Found. Please download a GGUF model to 'models/' folder.")

# Main Interface
query = st.text_input("Enter your question (Sanskrit/English):", "कुरुक्षेत्रे के समवेताः आसन्?")

if st.button("Get Answer"):
    if not query:
        st.warning("Please enter a query.")
    else:
        try:
            with st.spinner("Retrieving and Generating..."):
                chain = get_rag_chain()
                response = chain(query)
                
                st.subheader("Answer:")
                st.write(response['result'])
                
                st.subheader("Source Documents:")
                for i, doc in enumerate(response['source_documents']):
                    with st.expander(f"Source {i+1}"):
                        st.write(doc.page_content)
                        st.caption(f"Source: {doc.metadata['source']}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

st.markdown("---")
st.caption("Powered by LangChain, FAISS, and LlamaCpp")
