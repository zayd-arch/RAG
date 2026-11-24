import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser



from langchain_groq import ChatGroq


from indexing import trust_the_process
from querying import query

st.title("RECALL your AI study buddy")

if "GROQ_API_KEY" in st.secrets and "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)



rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's last message as a standalone question, using the chat history "
     "to resolve references like 'that' or 'the thing'. "
     "Keep the original language. If it is already standalone, return it unchanged. "
     "Return only the question, nothing else."),
    MessagesPlaceholder("history"),
    ("user", "{question}"),
])
rewriter = rewrite_prompt | llm | StrOutputParser()

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a study assistant.\n"
     "- For questions about the documents, answer using only the Context below.\n"
     "- For questions about the conversation itself (what was said or asked earlier), "
     "use the chat history, not the Context.\n"
     "- If you can't find the answer in the right source, say you don't know.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("history"),
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

user_input = st.chat_input("", accept_file=True, file_type=["pdf"])

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
        standalone = rewriter.invoke({
            "history": st.session_state.messages[:-1],
            "question": user_input.text,
        })
        docs = query(standalone)
        context = "\n\n".join(d.page_content for d in docs)
        answer = chain.invoke({"context": context,
                               "history":st.session_state.messages[:-1],
                                "question": user_input.text},
                              )

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
