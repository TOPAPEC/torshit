import wikipediaapi
import asyncio
from dataclasses import dataclass
from typing import Dict
from concurrent.futures import ThreadPoolExecutor

@dataclass
class WikiContent:
    summary: str
    full_text: str

class WikiService:
    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia('torshitapp/1.0', language='ru')

    async def get_wiki_content(self, city: str) -> WikiContent:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            page = await loop.run_in_executor(pool, self.wiki.page, city)
            if page.exists():
                return WikiContent(page.summary, page.text)
        return None

    async def get_all_cities_content(self) -> Dict[str, WikiContent]:
        tasks = [self.get_wiki_content(city) for city in Config.RESORT_CITIES]
        results = await asyncio.gather(*tasks)
        return {city: content for city, content in zip(Config.RESORT_CITIES, results) if content}
