from openai import AsyncOpenAI
from typing import List, Dict, Tuple, Optional
import tiktoken
import json
import asyncio

from config import Config


class ContextManager:
    def __init__(self, model_context_length: int = 10000):
        self.model_context_length = model_context_length
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")  # or your specific model
        self.system_prompt_tokens = len(self.tokenizer.encode(Config.SYSTEM_PROMPT))
        self.rag_prompt_tokens = len(self.tokenizer.encode(Config.GROUNDED_SYSTEM_PROMPT))
        self.expected_output_tokens = 2048

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def get_available_tokens(self, user_preferences: str, is_rag: bool = False) -> int:
        base_prompt = self.rag_prompt_tokens if is_rag else self.system_prompt_tokens
        user_tokens = self.count_tokens(user_preferences)
        return self.model_context_length - base_prompt - user_tokens - self.expected_output_tokens

class LLMService:
    def __init__(self, model_context_length: int = 10000):
        self.client = AsyncOpenAI(
            api_key=Config.OPENAI_KEY,
            base_url=Config.ENDPOINT
        )
        self.context_manager = ContextManager(model_context_length)

    async def compress_chunk(self, chunk: str) -> str:
        messages = [
            {"role": "system", "content": "Summarize the key tourist information in 2-3 sentences."},
            {"role": "user", "content": chunk}
        ]

        response = await self.client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=256
        )
        return response.choices[0].message.content

    async def merge_summaries(self, summaries: List[str], city: str) -> str:
        combined = f"Information about {city}:\n" + "\n".join(summaries)
        messages = [
            {"role": "system", "content": "Merge the information into a coherent summary about the location. Focus on tourist-relevant details."},
            {"role": "user", "content": combined}
        ]

        response = await self.client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=512
        )
        return response.choices[0].message.content

    async def prepare_rag_documents(self, 
                                  cities_chunks: Dict[str, List[str]], 
                                  user_preferences: str) -> List[dict]:
        available_tokens = self.context_manager.get_available_tokens(user_preferences, is_rag=True)
        tokens_per_city = available_tokens // len(cities_chunks)

        final_documents = []
        doc_id = 0

        for city, chunks in cities_chunks.items():
            # First stage: compress individual chunks
            compressed_chunks = await asyncio.gather(
                *[self.compress_chunk(chunk) for chunk in chunks]
            )

            # Second stage: merge compressed chunks for each city
            city_summary = await self.merge_summaries(compressed_chunks, city)

            # Check if we need to further compress
            while self.context_manager.count_tokens(city_summary) > tokens_per_city:
                city_summary = await self.compress_chunk(city_summary)

            final_documents.append({


                "doc_id": doc_id,
                "title": city,
                "content": city_summary
            })
            doc_id += 1

        return final_documents

    async def get_preferences(self, user_input: str) -> str:
        response = await self.client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": Config.SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0,
            max_tokens=512
        )
        return response.choices[0].message.content

    async def get_rag_response(self, user_preferences: str, documents: List[dict]) -> Tuple[str, str]:
        # Verify total context length
        docs_content = json.dumps(documents, ensure_ascii=False)
        available_tokens = self.context_manager.get_available_tokens(user_preferences, is_rag=True)

        if self.context_manager.count_tokens(docs_content) > available_tokens:
            # Emergency compression if still too long
            compressed_docs = []
            tokens_per_doc = available_tokens // len(documents)

            for doc in documents:
                content = doc["content"]
                while self.context_manager.count_tokens(content) > tokens_per_doc:
                    content = await self.compress_chunk(content)
                compressed_docs.append({
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "content": content
                })
            documents = compressed_docs

        messages = [
            {'role': 'system', 'content': Config.GROUNDED_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_preferences}
        ]

        doc_message = {'role': 'user', 'content': f"Available information:\n{json.dumps(documents, ensure_ascii=False)}"}

        response = await self.client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=messages + [doc_message],
            temperature=0.0,
            max_tokens=512
        )
        relevant_docs = response.choices[0].message.content

        final_response = await self.client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=messages + [doc_message, {'role': 'assistant', 'content': relevant_docs}],
            temperature=0.3,
            max_tokens=1024
        )
        return relevant_docs, final_response.choices[0].message.content

