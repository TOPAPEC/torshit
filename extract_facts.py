import asyncio
import json
from typing import List, Dict
from wiki import WikiService
from llm import LLMService
from config import Config
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

async def extract_tourist_facts(text: str, llm_service: LLMService) -> List[str]:
    """Extract tourist facts from text using LLM."""
    messages = [
        {"role": "system", "content": """Извлеките из текста отдельные туристические факты. Каждый факт должен быть:
- Самодостаточным (понятным без контекста)
- Полезным для туристов
- Конкретным (содержать специфическую информацию)

Правила:
1. Начинайте каждый факт с тире (-)
2. Не включайте метатекст вроде "Из текста можно извлечь" или "Эти факты могут быть полезны"
3. Не используйте заголовки или категории
4. Каждый факт должен быть полным предложением
5. Избегайте дублирования информации

Факты могут касаться:
- Достопримечательностей
- Истории
- Культуры
- Практической информации
- Транспорта
- Климата
- Сезонности посещения"""},
        {"role": "user", "content": text}
    ]
    
    response = await llm_service.client.chat.completions.create(
        model=Config.LLM_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=512
    )
    
    # Split response into individual facts and clean them up
    facts = []
    for fact in response.choices[0].message.content.split('\n'):
        fact = fact.strip()
        if fact and fact.startswith('-'):
            # Remove the leading dash and clean up the text
            fact = fact[1:].strip()
            if fact and not any(skip in fact.lower() for skip in [
                'из текста можно извлечь',
                'эти факты могут быть',
                'форматируем',
                'остальная часть текста',
                '**'
            ]):
                facts.append(fact)
    return facts

from difflib import SequenceMatcher

def similar(a: str, b: str, threshold: float = 0.85) -> bool:
    """Check if two strings are similar using SequenceMatcher."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

async def categorize_single_fact(fact: str, llm_service: LLMService) -> tuple[str, str]:
    """Categorize a single fact using LLM."""
    messages = [
        {"role": "system", "content": """Укажите категорию для данного факта. Категории:
- История
- Достопримечательности
- Транспорт
- Культура
- Практическая информация
- Развлечения
- Природа

Ответьте одним словом - названием категории."""},
        {"role": "user", "content": fact}
    ]
    
    response = await llm_service.client.chat.completions.create(
        model=Config.LLM_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=64
    )
    
    category = response.choices[0].message.content.strip()
    return fact, category

async def categorize_facts(facts: List[str], llm_service: LLMService) -> Dict[str, List[str]]:
    """Categorize facts using LLM in parallel."""
    categories = {
        "История": [],
        "Достопримечательности": [],
        "Транспорт": [],
        "Культура": [],
        "Практическая информация": [],
        "Развлечения": [],
        "Природа": []
    }
    
    print(f"Categorizing {len(facts)} facts in parallel...")
    tasks = [categorize_single_fact(fact, llm_service) for fact in facts]
    results = await tqdm_asyncio.gather(*tasks, desc="Categorizing facts")
    
    for fact, category in results:
        if category in categories:
            categories[category].append(fact)
        else:
            # If category doesn't match exactly, find the closest match
            closest = min(categories.keys(), key=lambda k: SequenceMatcher(None, k.lower(), category.lower()).ratio())
            categories[closest].append(fact)
    
    return categories

async def process_city(city: str) -> Dict[str, Dict[str, List[str]]]:
    """Process a city and extract tourist facts."""
    wiki_service = WikiService()
    llm_service = LLMService()
    
    # Get Wikipedia content
    wiki_content = await wiki_service.get_wiki_content(city)
    if not wiki_content:
        print(f"Could not fetch Wikipedia content for {city}")
        return {city: {}}
    
    # Process chunks in parallel with progress bar
    tasks = [extract_tourist_facts(chunk, llm_service) for chunk in wiki_content.chunks]
    chunk_facts = await tqdm_asyncio.gather(*tasks, desc=f"Processing {city}", unit="chunk")
    
    # Flatten facts
    all_facts = [fact for facts_list in chunk_facts for fact in facts_list]
    
    # Remove duplicates and similar facts while preserving order
    seen = set()
    unique_facts = []
    for fact in all_facts:
        if not any(similar(fact, existing) for existing in seen):
            seen.add(fact)
            unique_facts.append(fact)
    
    # Categorize facts
    print("Categorizing facts...")
    categorized_facts = await categorize_facts(unique_facts, llm_service)
    
    return {city: categorized_facts}

async def process_cities(cities: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """Process multiple cities in parallel."""
    tasks = [process_city(city) for city in cities]
    city_results = await asyncio.gather(*tasks)
    
    # Merge results
    results = {}
    for city_result in city_results:
        results.update(city_result)
    return results

async def main():
    with tqdm(total=3, desc="Tourist facts extraction") as pbar:
        # For now, only process Pskov
        cities = ["Псков"]
        pbar.set_description("Processing cities")
        results = await process_cities(cities)
        pbar.update(1)

        pbar.set_description("Saving results")
        # Save results to JSON file
        with open('tourist_facts.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        pbar.update(1)
        
        pbar.set_description("Complete")
        pbar.update(1)

if __name__ == "__main__":
    asyncio.run(main())
