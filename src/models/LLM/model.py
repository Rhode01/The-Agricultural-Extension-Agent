from langchain_openai import ChatOpenAI

class LLM:
    def __init__(self, model_name:str ='gpt-3.5', max_tokens:int=256, temp:float=0.7) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temp
        self.client = self._init_model()

    def _init_model(self)->ChatOpenAI:
        return ChatOpenAI(
            max_completion_tokens=self.max_tokens,
            temperature = self.temperature,
            model=self.model_name
        )