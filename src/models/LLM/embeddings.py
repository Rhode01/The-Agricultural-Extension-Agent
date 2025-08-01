from typing import List, Union
from langchain_openai import OpenAIEmbeddings

class LLMEmbedding:
    def __init__(self, model_name: str = 'text-embedding-3-small') -> None:
        self.model_name = model_name
        self.client = OpenAIEmbeddings(model=self.model_name)

    def embed_text(self, text: str) -> List[float]:
        return self.client.embed_query(text)

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        return self.client.embed_documents(docs)

    def embed(self,texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if isinstance(texts, str):
            return self.embed_text(texts)
        return self.embed_documents(texts)
