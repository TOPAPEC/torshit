import wikipediaapi
import asyncio
from dataclasses import dataclass
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

from config import Config

@dataclass
class WikiContent:
    summary: str
    full_text: str
    chunks: List[str]

class TextProcessor:
    def __init__(self, max_chunk_size: int = 4000):
        self.max_chunk_size = max_chunk_size

    def create_chunks(self, title: str, text: str) -> List[str]:
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)
            if current_size + para_size > self.max_chunk_size:
                if current_chunk:
                    chunks.append(f"Title: {title}\n\n" + '\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        if current_chunk:
            chunks.append(f"Title: {title}\n\n" + '\n\n'.join(current_chunk))
        return chunks

class WikiService:
    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia('torshitapp/1.0', language='ru')
        self.text_processor = TextProcessor()

    async def get_wiki_content(self, city: str) -> WikiContent:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            page = await loop.run_in_executor(pool, self.wiki.page, city)
            if page.exists():
                chunks = self.text_processor.create_chunks(city, page.text)
                return WikiContent(page.summary, page.text, chunks)
        return None

    async def get_all_cities_content(self) -> Dict[str, WikiContent]:
        tasks = [self.get_wiki_content(city) for city in Config.RESORT_CITIES]
        results = await asyncio.gather(*tasks)
        return {city: content for city, content in zip(Config.RESORT_CITIES, results) if content}
