#!/usr/bin/env python3

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from indexing import trust_the_process
from querying import query

import os

if "GROQ_API_KEY" in st.secrets and "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# 1. LLM, prompt, and chain
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer only using the provided context. If unsure, say you don't know.\n\nContext:\n{context}",
    ),
    ("user", "{question}"),
])

chain = prompt | llm | StrOutputParser()


def generate_answer(question: str, docs: list) -> str:
    context = "\n\n".join(doc.page_content for doc in docs)
    return chain.invoke({"context": context, "question": question})


st.title("RAG just RAG RJR")

# 1. User inputs
pdf = st.file_uploader("Upload your PDF", type=["pdf"])
name_col = st.text_input("Collection name", placeholder="e.g. homl")
question = st.text_input("Ask a question about your PDF")

# 2. Process PDF
if st.button("Index PDF"):
    if pdf is None or not name_col:
        st.warning("Upload a PDF and provide a collection name.")
    else:
        with st.spinner("Indexing PDF..."):
            try:
                chunks = trust_the_process(pdf, name_col)
                st.success(f"Indexed {chunks} chunks!")
            except Exception as e:
                st.error(str(e))

# 3. Query PDF + generate answer
if st.button("Ask question"):
    if not name_col or not question:
        st.warning("Enter a collection name and a question.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            try:
                docs = query(question, name_col)
                answer = generate_answer(question, docs)

                st.subheader("Answer")
                st.write(answer)

                with st.expander("Retrieved chunks"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Result {i + 1}** (Page: {doc.metadata.get('page', 'Unknown')})")
                        st.write(doc.page_content)

            except Exception as e:
                st.error(str(e))
