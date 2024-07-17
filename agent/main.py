
from llama_index.core.agent import AgentRunner
from llama_index.llms.vertex import Vertex
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage
from google.oauth2 import service_account
from session import Session
from llama_index.core.storage.chat_store import SimpleChatStore
import time


filename = "../key.json"
credentials: service_account.Credentials = (
    service_account.Credentials.from_service_account_file(filename)
)
llm = Vertex(
    model="gemini-1.5-flash-latest", project=credentials.project_id, credentials=credentials
)
#
Settings.llm = llm

# chat_store = {}

persist_path = "chat_store.json"
chat_store = SimpleChatStore.from_persist_path(persist_path)

agent = AgentRunner.from_llm(llm=llm, tools=[], verbose=True)
print(agent.get_prompts())


def get_response(query, chat_history):

    global agent

    response = agent.chat(query, chat_history=chat_history)
    chat_history = agent.memory.chat_store.store["chat_history"]

    return response, chat_history


def ask(query: str, user_id: str) -> str:
    global chat_store

    session = Session(user_id=user_id, chat_store=chat_store, persist_path=persist_path)

    chat_history = session.get_messages()

    response, chat_history = get_response(query, chat_history)
    session.set_messages(chat_history)

    return response


t1 = time.time()
print("asking: ", t1)
print(ask('Merci pour l explication?', 'user_4'))

t2 = time.time()
print("execution time: ", t2 - t1)
#print(chat_store['sitraka'])

"""messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
print(llm.chat(messages))
"""