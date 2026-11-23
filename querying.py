
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma




def query(question):

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = Chroma(
        collection_name='my_app_docs',
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )

    results = db.similarity_search(
        question,
        k=4
    )

    return results
