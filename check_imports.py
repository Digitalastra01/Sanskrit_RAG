try:
    from langchain.chains import RetrievalQA
    print("RetrievalQA imported successfully")
except ImportError as e:
    print(f"Error importing RetrievalQA: {e}")

try:
    from langchain_community.llms import LlamaCpp
    print("LlamaCpp imported successfully")
except ImportError as e:
    print(f"Error importing LlamaCpp: {e}")

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("HuggingFaceEmbeddings imported successfully")
except ImportError as e:
    print(f"Error importing HuggingFaceEmbeddings: {e}")
