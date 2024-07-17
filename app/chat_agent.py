from llama_index.core.agent import AgentRunner
from llama_index.core import Settings
from llama_index.llms.vertex import Vertex
from google.oauth2 import service_account


class ChatAgent:
    def __init__(self, tools, credentials_path: str = "../key.json"):

        credentials: service_account.Credentials = (
            service_account.Credentials.from_service_account_file(credentials_path)
        )
        llm = Vertex(
            model="gemini-1.5-flash", project=credentials.project_id, credentials=credentials
        )

        Settings.llm = llm
        self.agent = AgentRunner.from_llm(llm=llm, tools=tools, verbose=True)

    def chat(self, query, chat_history):
        response = self.agent.chat(query, chat_history=chat_history)
        return str(response)
