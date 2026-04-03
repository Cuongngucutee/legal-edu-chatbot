import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from book_index import BookIndex
from retriever import BookRAGRetriever

def main():
    print("Loading index...")
    t0 = time.time()
    index = BookIndex("d:/legal-edu-chatbot/data/final", "d:/legal-edu-chatbot/data/entity_graph.json")
    index.load_index()
    print(f"Index loaded in {time.time() - t0:.2f}s")
    
    retriever = BookRAGRetriever(index)
    
    query = "Điều kiện đình chỉ hoạt động đào tạo của cơ sở giáo dục đại học"
    print(f"\nQuery: {query}")
    print("Retrieving...")
    
    t1 = time.time()
    context = retriever.retrieve(query, top_k=3, max_hops=1)
    print(f"Retrieved in {time.time() - t1:.2f}s")
    
    print("\n--- CONTEXT ---")
    print(context)

if __name__ == "__main__":
    main()
