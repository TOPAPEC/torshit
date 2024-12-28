from openai import AsyncOpenAI
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional
import tiktoken
import json
import asyncio

from config import Config



class ContextManager:
    def __init__(self, model_context_length: int = 10000):
        self.model_context_length = model_context_length
        self.tokenizer = AutoTokenizer.from_pretrained("Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24")
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
        self.max_summary_tokens = 512
        self.max_final_response_tokens = 1024

    async def compress_chunk(self, chunk: str, max_tokens: int) -> str:
        while self.context_manager.count_tokens(chunk) > max_tokens:
            messages = [
            {"role": "system", "content": """Кратко обобщите ключевую туристическую информацию, включая:
- климат и погодные условия
- экологическую обстановку
- температуру воды (если есть водоемы)
- основные достопримечательности
- исторические объекты
- транспортную доступность
Сохраняйте только самую важную информацию для туристов."""},
            {"role": "user", "content": chunk}
        ]
            response = await self.client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens
            )
            chunk = response.choices[0].message.content
        return chunk

    async def merge_summaries(self, summaries: List[str], city: str, max_tokens: int) -> str:
        combined = f"Информация о {city}:\n" + "\n".join(summaries)

        if self.context_manager.count_tokens(combined) <= max_tokens:
            return combined

        messages = [
            {"role": "system", "content": """Объедините информацию в связное описание места, фокусируясь на:
    - главных достопримечательностях
    - климате и сезонности посещения
    - транспортной инфраструктуре
    - уникальных особенностях места
    - практических советах для туристов
    Информация должна быть полезной для планирования поездки."""},
            {"role": "user", "content": combined}
        ]
        response = await self.client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    async def prepare_rag_documents(self, 
                                  cities_chunks: Dict[str, List[str]], 
                                  user_preferences: str) -> List[dict]:
        available_tokens = self.context_manager.get_available_tokens(user_preferences, is_rag=True)

        # Reserve tokens for system messages and final response
        working_tokens = available_tokens - self.max_final_response_tokens
        tokens_per_city = working_tokens // len(cities_chunks)

        final_documents = []
        doc_id = 0

        for city, chunks in cities_chunks.items():
            try:
                # First stage: compress individual chunks
                tokens_per_chunk = tokens_per_city // (len(chunks) or 1)
                compressed_chunks = await asyncio.gather(
                    *[self.compress_chunk(chunk, tokens_per_chunk) for chunk in chunks]
                )

                # Second stage: merge compressed chunks for each city
                city_summary = await self.merge_summaries(
                    compressed_chunks, 
                    city, 
                    tokens_per_city
                )

                # Final compression if still needed
                if self.context_manager.count_tokens(city_summary) > tokens_per_city:
                    city_summary = await self.compress_chunk(city_summary, tokens_per_city)

                final_documents.append({
                    "doc_id": doc_id,
                    "title": city,
                    "content": city_summary
                })
                doc_id += 1

            except Exception as e:
                print(f"Error processing city {city}: {e}")
                continue

        # Final safety check
        total_tokens = sum(self.context_manager.count_tokens(doc["content"]) 
                          for doc in final_documents)

        if total_tokens > working_tokens:
            # Emergency compression of all documents
            compressed_docs = []
            new_tokens_per_city = working_tokens // len(final_documents)

            for doc in final_documents:
                compressed_content = await self.compress_chunk(
                    doc["content"], 
                    new_tokens_per_city
                )
                compressed_docs.append({
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "content": compressed_content
                })
            final_documents = compressed_docs

        return final_documents

    async def get_preferences(self, user_input: str) -> str:
        max_tokens = self.max_summary_tokens
        response = await self.client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": Config.SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    async def get_rag_response(self, user_preferences: str, documents: List[dict]) -> Tuple[str, str]:
        # Calculate available tokens for responses
        available_tokens = self.context_manager.get_available_tokens(user_preferences, is_rag=True)
        docs_content = json.dumps(documents, ensure_ascii=False)
        docs_tokens = self.context_manager.count_tokens(docs_content)

        if docs_tokens > available_tokens * 0.7:  # Leave 30% for responses
            # Emergency compression of all documents
            compressed_docs = []
            max_tokens_per_doc = (available_tokens * 0.7) // len(documents)

            for doc in documents:
                compressed_content = await self.compress_chunk(
                    doc["content"], 
                    max_tokens_per_doc
                )
                compressed_docs.append({
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "content": compressed_content
                })
            documents = compressed_docs

        messages = [
            {'role': 'system', 'content': Config.GROUNDED_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_preferences}
        ]

        doc_message = {'role': 'user', 'content': f"Available information:\n{json.dumps(documents, ensure_ascii=False)}"}

        # Get relevant docs with limited tokens
        response = await self.client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=messages + [doc_message],
            temperature=0.0,
            max_tokens=self.max_summary_tokens
        )
        relevant_docs = response.choices[0].message.content

        # Get final answer with remaining tokens
        final_response = await self.client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=messages + [doc_message, {'role': 'assistant', 'content': relevant_docs}],
            temperature=0.3,
            max_tokens=self.max_final_response_tokens
        )
        return relevant_docs, final_response.choices[0].message.content


