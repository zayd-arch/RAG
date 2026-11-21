#!/usr/bin/env python3
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

TEST_PAGES = None   # set to None to index the whole book

# 1. Ingestion
reader = PdfReader("HOML.pdf")
pages = reader.pages[:TEST_PAGES] if TEST_PAGES else reader.pages
docs = [
    Document(page_content=p.extract_text() or "", metadata={"page": i + 1})
    for i, p in enumerate(pages)
]
docs = [d for d in docs if d.page_content.strip()]

# 2. Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print("chunks:", len(chunks))

# 3. Embedding model + vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    collection_name="homl",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# 4. Store in batches, with progress
BATCH = 128
for i in range(0, len(chunks), BATCH):
    vector_store.add_documents(chunks[i:i + BATCH])
    print(f"stored {min(i + BATCH, len(chunks))}/{len(chunks)}")

print("count in db:", vector_store._collection.count())
