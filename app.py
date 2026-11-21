#!/usr/bin/env python3


import streamlit as st

from indexing import trust_the_process
from querying import query

st.title("My RAG Application")

# 1. User inputs
pdf = st.file_uploader("Upload your PDF", type=["pdf"])

name_col = st.text_input(
    "Collection name",
    placeholder="e.g. homl"
)

question = st.text_input(
    "Ask a question about your PDF"
)

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


# 3. Query PDF
if st.button("Ask question"):

    if not name_col or not question:
        st.warning("Enter a collection name and a question.")

    else:
        with st.spinner("Searching your documents..."):

            try:
                results = query(question, name_col)

                st.subheader("Retrieved information")

                for i, doc in enumerate(results):

                    st.markdown(f"### Result {i+1}")
                    st.caption(
                        f"Page: {doc.metadata.get('page', 'Unknown')}"
                    )
                    st.write(doc.page_content)

            except Exception as e:
                st.error(str(e))
