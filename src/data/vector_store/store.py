from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

class LLMvector_store:
    def __init__(self, embeddings:OpenAIEmbeddings) -> None:
        self.embeddings = embeddings
        self.vector_store = self._init_store()
    def _init_store(self) -> InMemoryVectorStore:
        return(
            InMemoryVectorStore(self.embeddings)
        )
