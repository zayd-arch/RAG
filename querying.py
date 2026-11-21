#!/usr/bin/env python3
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(
    collection_name="homl",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

print("count:", db._collection.count())

results = db.similarity_search("What is gradient descent?", k=4)
for r in results:
    print(f"--- page {r.metadata['page']} ---")
    print(r.page_content[:300])
