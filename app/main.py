import os
from document_loader import DocumentLoader
from index_manager import IndexManager
from query_engine_manager import QueryEngineManager
from chat_agent import ChatAgent
import streamlit as st
from llama_index.core.storage.chat_store import SimpleChatStore
from session import Session



# Vérifie si l'assistant est déjà en session_state
if "assistant" not in st.session_state:


    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../key.json"
    years = [2022, 2021, 2020, 2019]
    data_path = "../data/UBER"

    document_loader = DocumentLoader(years, data_path)
    doc_set, all_docs = document_loader.load_documents()

    index_manager = IndexManager(years)
    # index_set = index_manager.create_indices(doc_set)
    index_set = index_manager.load_indices()
    print("-------------index loaded-------------")

    query_engine_manager = QueryEngineManager(index_set, credentials_path="../key.json")
    tools = query_engine_manager.create_query_tools()

    persist_path = "chat_store.json"
    chat_store = SimpleChatStore.from_persist_path(persist_path)

    session = Session(user_id="test_user", chat_store=chat_store)
    assistant = ChatAgent(tools)

    st.session_state.session = session
    st.session_state.assistant = assistant
else:
    assistant = st.session_state.assistant
    session = st.session_state.session


st.title("QA Application")


for message in session.get_messages():
    with st.chat_message(message.role):
        st.markdown(message.content)

if prompt := st.chat_input("What is up?"):

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = assistant.chat(query=prompt, chat_history=session.get_messages())

        st.write(response)
    session.set_messages(assistant.agent.memory.chat_store.store["chat_history"])

