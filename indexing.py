
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def trust_the_process(pdf):

    # 1. Extract PDF text
    pdf.seek(0)
    reader = PdfReader(pdf)

    docs = [
        Document(
            page_content=p.extract_text() or "",
            metadata={"page": i + 1}
        )
        for i, p in enumerate(reader.pages)
    ]

    docs = [d for d in docs if d.page_content.strip()]

    if not docs:
        raise ValueError("No extractable text found.")

    # 2. Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    # 3. Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # 4. ChromaDB
    db = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )

    # 5. Store embeddings
    BATCH = 128

    for i in range(0, len(chunks), BATCH):
        db.add_documents(chunks[i:i+BATCH])

    return len(chunks)
