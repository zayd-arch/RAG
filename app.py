import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from indexing import trust_the_process
from querying import query   # rename to your function

st.title("RECALL your AI study buddy")

if "GROQ_API_KEY" in st.secrets and "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer only using the provided context. If unsure, say you don't know.\n\nContext:\n{context}"),
    ("user", "{question}"),
])
chain = qa_prompt | llm | StrOutputParser()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    st.session_state.indexed = set()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("HELLO!!", accept_file=True, file_type=["pdf"])


if user_input:
    # 1) index a new PDF
    if user_input.files:
        f = user_input.files[0]
        if f.name not in st.session_state.indexed:
            with st.spinner("Indexing..."):
                trust_the_process(f)
            st.session_state.indexed.add(f.name)
            st.toast("PDF indexed")

    # 2) answer a question
    if user_input.text:
        st.session_state.messages.append({"role": "user", "content": user_input.text})
        with st.chat_message("user"):
            st.markdown(user_input.text)

        docs = query(user_input.text)
        context = "\n\n".join(d.page_content for d in docs)
        answer = chain.invoke({"context": context, "question": user_input.text})

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
