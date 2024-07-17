from llama_index.readers.file import UnstructuredReader
from pathlib import Path


class DocumentLoader:
    def __init__(self, years, data_path):
        self.years = years
        self.data_path = data_path
        self.loader = UnstructuredReader()
        self.doc_set = {}
        self.all_docs = []

    def load_documents(self):
        for year in self.years:
            year_docs = self.loader.load_data(
                file=Path(f"{self.data_path}/UBER_{year}.html"), split_documents=False
            )
            for d in year_docs:
                d.metadata = {"year": year}
            self.doc_set[year] = year_docs
            self.all_docs.extend(year_docs)
        return self.doc_set, self.all_docs



