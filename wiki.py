import wikipediaapi
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from config import Config
from osm_service import OSMService, CityPOIs

@dataclass
class WikiContent:
    summary: str
    full_text: str
    chunks: List[str]
    pois: Optional[CityPOIs] = None

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
        self.osm_service = OSMService()

    async def get_wiki_content(self, city: str) -> WikiContent:
        loop = asyncio.get_event_loop()
        
        # Fetch Wikipedia content
        with ThreadPoolExecutor() as pool:
            page = await loop.run_in_executor(pool, self.wiki.page, city)
            if not page.exists():
                return None
            
            chunks = self.text_processor.create_chunks(city, page.text)
            
            # Fetch POIs in parallel
            pois = await self.osm_service.get_city_pois(city)
            
            # Add POI information to chunks
            poi_description = self.osm_service.format_poi_description(pois)
            if poi_description:
                chunks.append(f"Title: {city}\n\nТуристическая информация:\n{poi_description}")
            
            return WikiContent(page.summary, page.text, chunks, pois)

    async def get_cities_by_type(self, location_type: str) -> Dict[str, WikiContent]:
        """Get content for cities of a specific type (e.g., 'море', 'город')."""
        if location_type not in Config.RESORT_CITIES:
            print(f"Warning: Unknown location type '{location_type}', falling back to all cities")
            cities = [city for sublist in Config.RESORT_CITIES.values() for city in sublist]
        else:
            cities = Config.RESORT_CITIES[location_type]
        
        tasks = [self.get_wiki_content(city) for city in cities]
        results = await asyncio.gather(*tasks)
        return {city: content for city, content in zip(cities, results) if content}

    async def get_all_cities_content(self) -> Dict[str, WikiContent]:
        """Get content for all cities (fallback method)."""
        all_cities = [city for sublist in Config.RESORT_CITIES.values() for city in sublist]
        tasks = [self.get_wiki_content(city) for city in all_cities]
        results = await asyncio.gather(*tasks)
        return {city: content for city, content in zip(all_cities, results) if content}
