from rag_pipeline import get_rag_chain
import sys

try:
    print("Initializing RAG chain...")
    chain = get_rag_chain()
    
    query = "कुरुक्षेत्रे के समवेताः आसन्?"
    print(f"Testing query: {query}")
    
    response = chain(query)
    print("Response received:")
    print(response['result'])
    
    print("\nSource Documents:")
    for doc in response['source_documents']:
        print(f"- {doc.page_content[:100]}...")
        
    print("\nVerification Successful!")
except Exception as e:
    print(f"Verification Failed: {e}")
    sys.exit(1)
