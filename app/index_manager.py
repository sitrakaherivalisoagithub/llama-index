from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.embeddings.google import GooglePaLMEmbedding


class IndexManager:
    def __init__(self, years, chunk_size=512, chunk_overlap=64, llm_model="gpt-3.5-turbo", embed_model="models/embedding-gecko-001"):
        self.years = years
        self.index_set = {}
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        # Settings.embed_model = OpenAIEmbedding(model=embed_model)

        api_key = "AIzaSyDyKc04UhM-QGyhXo02Tn"
        Settings.embed_model = GooglePaLMEmbedding(model_name=embed_model, api_key=api_key)

    def create_indices(self, doc_set):
        for year in self.years:
            storage_context = StorageContext.from_defaults()
            cur_index = VectorStoreIndex.from_documents(
                doc_set[year],
                storage_context=storage_context,
            )
            self.index_set[year] = cur_index
            storage_context.persist(persist_dir=f"./storage/{year}")

        return self.index_set

    def load_indices(self):
        for year in self.years:
            storage_context = StorageContext.from_defaults(
                persist_dir=f"./storage/{year}"
            )
            cur_index = load_index_from_storage(
                storage_context,
            )
            self.index_set[year] = cur_index
        return self.index_set
