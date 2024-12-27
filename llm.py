import openai
import json
from typing import List, Tuple

from config import Config

class LLMService:
    def __init__(self):
        openai.api_key = Config.OPENAI_KEY
        openai.api_base = Config.ENDPOINT

    def get_preferences(self, user_input: str) -> str:
        return openai.ChatCompletion.create(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": Config.SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0,
            max_tokens=2048
        ).choices[0].message.content

    def get_rag_response(self, user_preferences: str, documents: List[dict]) -> Tuple[str, str]:
        history = [
            {'role': 'system', 'content': Config.GROUNDED_SYSTEM_PROMPT},
            {'role': 'documents', 'content': json.dumps(documents, ensure_ascii=False)},
            {'role': 'user', 'content': user_preferences}
        ]
        
        relevant_docs = openai.ChatCompletion.create(
            model=Config.LLM_MODEL,
            messages=history,
            temperature=0.0,
            max_tokens=2048
        ).choices[0].message.content

        final_answer = openai.ChatCompletion.create(
            model=Config.LLM_MODEL,
            messages=history + [{'role': 'assistant', 'content': relevant_docs}],
            temperature=0.3,
            max_tokens=2048
        ).choices[0].message.content

        return relevant_docs, final_answer
