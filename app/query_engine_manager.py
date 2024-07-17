
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from google.oauth2 import service_account
from llama_index.llms.vertex import Vertex


class QueryEngineManager:
    def __init__(self, index_set, credentials_path):
        self.index_set = index_set
        self.tools = []
        credentials: service_account.Credentials = (
            service_account.Credentials.from_service_account_file(credentials_path)
        )
        self.llm = Vertex(
            model="gemini-1.5-flash", project=credentials.project_id, credentials=credentials
        )

    def create_query_tools(self):
        individual_query_engine_tools = [
            QueryEngineTool(
                query_engine=self.index_set[year].as_query_engine(llm=self.llm),
                metadata=ToolMetadata(
                    name=f"vector_index_{year}",
                    description=(
                        "useful for when you want to answer queries about the"
                        f" {year} SEC 10-K for Uber"
                    ),
                ),
            )
            for year in self.index_set.keys()
        ]
        query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=individual_query_engine_tools, llm=self.llm
        )
        query_engine_tool = QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="sub_question_query_engine",
                description=(
                    "useful for when you want to answer queries that require analyzing"
                    " multiple SEC 10-K documents for Uber"
                ),
            ),
        )
        self.tools = individual_query_engine_tools + [query_engine_tool]
        return self.tools
