from openai import AsyncOpenAI
import asyncio
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

from config import Config

class LLMService:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=Config.OPENAI_KEY,
            base_url=Config.ENDPOINT
        )

    async def extract_chunk_preferences(self, chunk: str) -> str:
        messages = [
            {"role": "system", "content": "Extract key tourist information from the text: attractions, weather, prices, accommodations, transport. Be brief."},
            {"role": "user", "content": chunk}
        ]

        response = await self.client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=512
        )
        return response.choices[0].message.content

    async def get_preferences(self, user_input: str) -> str:
        response = await self.client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": Config.SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0,
            max_tokens=2048
        )
        return response.choices[0].message.content

    async def process_chunks(self, chunks: List[str]) -> List[str]:
        tasks = [self.extract_chunk_preferences(chunk) for chunk in chunks]
        return await asyncio.gather(*tasks)

    async def get_rag_response(self, user_preferences: str, documents: List[dict]) -> Tuple[str, str]:
        messages = [
            {'role': 'system', 'content': Config.GROUNDED_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_preferences}
        ]

        # Send documents as separate message
        doc_message = {'role': 'user', 'content': f"Available information:\n{json.dumps(documents, ensure_ascii=False)}"}

        response = await self.client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=messages + [doc_message],
            temperature=0.0,
            max_tokens=2048
        )
        relevant_docs = response.choices[0].message.content

        final_response = await self.client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=messages + [doc_message, {'role': 'assistant', 'content': relevant_docs}],
            temperature=0.3,
            max_tokens=2048
        )
        return relevant_docs, final_response.choices[0].message.content

