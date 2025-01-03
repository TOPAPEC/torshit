from wiki import WikiService
from embeddings import EmbeddingService
from llm import LLMService
from seasons import SEASONS
from activities import ActivityMatcher
from temperature import normalize_temperature_text, is_temperature_in_range
from config import Config
import re
import json
import asyncio

class TravelAdvisor:
    def __init__(self, model_context_length: int = 10000):
        print("Initializing TravelAdvisor")
        self.wiki_service = WikiService()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService(model_context_length)
        self.context_manager = self.llm_service.context_manager
        self.activity_matcher = ActivityMatcher(self.llm_service)
        
        # Load tourist facts
        with open('tourist_facts.json', 'r', encoding='utf-8') as f:
            self.tourist_facts = json.load(f)
        
        # Calculate embeddings for all facts using batching
        from tqdm import tqdm
        self.fact_embeddings = {}
        
        # Count total facts for progress bar
        total_facts = sum(
            len(facts) 
            for categories in self.tourist_facts.values() 
            for facts in categories.values()
        )
        print(f"Found {total_facts} facts to process")
        
        # Collect all facts with progress bar
        all_facts = []
        fact_to_city_category = {}
        with tqdm(total=total_facts, desc="Collecting facts") as pbar:
            for city, categories in self.tourist_facts.items():
                for category, facts in categories.items():
                    for fact in facts:
                        all_facts.append(fact)
                        fact_to_city_category[fact] = (city, category)
                        pbar.update(1)
        
        # Get embeddings in batches
        print("Computing embeddings...")
        fact_embeddings = self.embedding_service.get_embeddings_batch(all_facts)
        
        # Organize embeddings back into structure
        print("Organizing results...")
        for fact, embedding in fact_embeddings.items():
            city, category = fact_to_city_category[fact]
            if city not in self.fact_embeddings:
                self.fact_embeddings[city] = {}
            if category not in self.fact_embeddings[city]:
                self.fact_embeddings[city][category] = []
            self.fact_embeddings[city][category].append((fact, embedding))
        
    def _filter_cities_by_season(self, cities_content: dict, season: str, preferences: str = "") -> dict:
        """Filter cities based on seasonal criteria and preferences"""
        if not season:
            return cities_content
            
        filtered_cities = {}
        season_data = SEASONS[season]
        
        # Extract temperature preference if specified
        temp_pref = None
        if "температура:" in preferences.lower():
            temp_match = re.search(r'температура:.*?(\d+)', preferences.lower())
            if temp_match:
                temp_pref = int(temp_match.group(1))
        
        for city, content in cities_content.items():
            city_text = content.summary.lower()
            matches_season = False
            city_temps = []
            
            # Extract all temperatures
            temp_matches = re.finditer(r'температура.*?(-?\d+)', city_text)
            for match in temp_matches:
                temp = int(match.group(1))
                city_temps.append(temp)
            
            # Check seasonal temperature ranges
            if city_temps:
                avg_temp = sum(city_temps) / len(city_temps)
                if season_data['temp_range'][0] <= avg_temp <= season_data['temp_range'][1]:
                    matches_season = True
                    
                    # Apply temperature preference if specified
                    if temp_pref and any(temp > temp_pref for temp in city_temps):
                        matches_season = False
            
            # Check for required infrastructure based on preferences, but be more lenient for beach cities
            if city not in Config.RESORT_CITIES.get('море', []):  # Only apply strict checks for non-beach cities
                if 'спа' in preferences.lower() or 'массаж' in preferences.lower():
                    if not any(kw in city_text.lower() for kw in ['спа', 'массаж', 'велнес', 'wellness', 'санатори']):
                        continue
                        
                if 'аквапарк' in preferences.lower():
                    if not any(kw in city_text.lower() for kw in ['аквапарк', 'водные аттракцион', 'развлечения']):
                        continue
                    
            # Check for seasonal activities and infrastructure
            if any(keyword in city_text for keyword in season_data['keywords']):
                matches_season = True
                
            # Additional winter sports check with expanded keywords
            if season == 'winter' and 'горнолыж' in preferences.lower():
                if not any(kw in city_text.lower() for kw in [
                    'горнолыж', 'лыж', 'подъемник', 'трасс', 'склон', 
                    'катани', 'сноуборд', 'горнолыжный курорт'
                ]):
                    matches_season = False
                    
            # Additional summer beach check with expanded criteria   
            if season == 'summer' and 'пляж' in preferences.lower():
                # For beach cities, be more lenient with seasonal matching
                if city in Config.RESORT_CITIES.get('море', []):
                    matches_season = True
                else:
                    beach_keywords = ['пляж', 'море', 'песч', 'курорт', 'набережн', 'побереж', 'залив', 'бухт']
                    infra_keywords = ['развлечен', 'аквапарк', 'атракцион', 'детская площадка', 'отдых']
                    
                    has_beach = any(kw in city_text.lower() for kw in beach_keywords)
                    has_infrastructure = 'развлечения для детей' not in preferences.lower() or \
                                       any(kw in city_text.lower() for kw in infra_keywords)
                    
                    if has_beach:  # Only require beach presence for summer season
                        matches_season = True
            
            if matches_season:
                filtered_cities[city] = content
        
        # If no cities match criteria, try relaxing constraints before returning original
        if not filtered_cities:
            # Try matching just keywords without temperature
            for city, content in cities_content.items():
                if any(keyword in content.summary.lower() for keyword in season_data['keywords']):
                    filtered_cities[city] = content
                    
        return filtered_cities if filtered_cities else cities_content

    async def process_request(self, user_input: str):
        try:
            # Extract preferences and season
            preferences = await self.llm_service.get_preferences(user_input)
            print(f"Extracted preferences: {preferences}")
            
            # Extract activities and season
            activities = await self.activity_matcher.get_activities(user_input + "\n" + preferences)
            if activities:
                print(f"🎯 Detected activities: {', '.join(f'{act}({conf:.2f})' for act, conf in activities)}")
            
            season = await self.llm_service.get_season(user_input + "\n" + preferences)
            if season:
                print(f"🌍 Detected season: {season}")
                
            # Store primary activity for ranking
            primary_activity = activities[0][0] if activities else None

            # Calculate available tokens
            available_tokens = self.context_manager.get_available_tokens(preferences, is_rag=True)
            print(f"Available tokens: {available_tokens}")

            # Determine location type from preferences and activities
            location_type = None
            pref_text = preferences.lower()
            
            # Check for beach/sea indicators
            if any(word in pref_text for word in ['пляж', 'море', 'песок', 'пляжный отдых']):
                location_type = 'море'
            # Check for mountain/skiing indicators    
            elif any(word in pref_text for word in ['горы', 'лыж', 'горнолыж']):
                location_type = 'горы'
            # Check for spa/wellness indicators
            elif any(word in pref_text for word in ['спа', 'санатори', 'оздоровительн', 'лечебн', 'массаж']):
                location_type = 'spa'
            # Check for city/cultural indicators
            elif any(word in pref_text for word in ['музей', 'культур', 'город', 'архитектур']):
                location_type = 'город'
            
            # Also check activities
            if activities:
                activity = activities[0][0]
                if activity in ['beach_vacation', 'water_sports']:
                    location_type = 'море'
                elif activity in ['winter_sports', 'skiing']:
                    location_type = 'горы'
                elif activity in ['cultural_tourism', 'city_break']:
                    location_type = 'город'
            
            print(f"Determined location type: {location_type}")
            cities_content = await self.wiki_service.get_cities_by_type(location_type) if location_type else await self.wiki_service.get_all_cities_content()
            
            if not cities_content:
                print("No cities content found")
                return None, None, None, None, None

            # Normalize temperature data
            normalized_cities = {}
            for city, content in cities_content.items():
                content.summary = normalize_temperature_text(content.summary)
                normalized_cities[city] = content
            cities_content = normalized_cities
            
            # Enhanced activity filtering with infrastructure requirements
            if primary_activity:
                filtered_cities = {}
                for city, content in cities_content.items():
                    activity_score = self.activity_matcher.get_activity_score(content.summary, primary_activity)
                    city_text = content.summary.lower()
                    
                    # Use a lower threshold for beach_vacation to be more inclusive
                    min_score = 0.1 if primary_activity == 'beach_vacation' else 0.2
                    
                    if activity_score >= min_score:
                        # For beach_vacation, check if it's in the море category or has beach-related keywords
                        if primary_activity == 'beach_vacation':
                            beach_keywords = ['пляж', 'море', 'побереж', 'залив', 'бухт', 'курорт']
                            if city in Config.RESORT_CITIES.get('море', []) or any(kw in city_text for kw in beach_keywords):
                                filtered_cities[city] = content
                                
                        # For other activities, use activity-specific checks
                        elif primary_activity == 'spa_wellness':
                            spa_keywords = ['спа', 'массаж', 'велнес', 'wellness', 'санатори', 'оздоровит', 
                                          'лечебн', 'курорт', 'отдых', 'процедур', 'релакс', 'термальн',
                                          'источник', 'грязелечени', 'минеральн', 'нарзан', 'бювет']
                            if any(kw in city_text for kw in spa_keywords) or \
                               city in Config.RESORT_CITIES.get('spa', []) or \
                               city in ['Кисловодск', 'Пятигорск', 'Ессентуки', 'Железноводск']:
                                filtered_cities[city] = content
                                
                        elif primary_activity == 'winter_sports':
                            if any(kw in city_text for kw in ['горнолыж', 'лыж', 'подъемник', 'трасс', 'склон', 'зимн']):
                                filtered_cities[city] = content
                                
                        elif primary_activity == 'cultural_tourism':
                            if any(kw in city_text for kw in ['музе', 'памятник', 'достопримечательност', 'истори', 'культур']):
                                filtered_cities[city] = content
                                
                        else:  # Default case for other activities
                            filtered_cities[city] = content
                            
                if filtered_cities:  # Only update if we found matching cities
                    cities_content = filtered_cities
                    print(f"Found {len(cities_content)} cities matching activity and infrastructure criteria")
            
            # Apply seasonal filtering with preferences
            cities_content = self._filter_cities_by_season(cities_content, season, preferences)
            print(f"Found {len(cities_content)} cities matching all criteria")

            # Prepare embeddings
            summaries = [content.summary for content in cities_content.values()]
            summaries.append(preferences)
            print(f"Total summaries to embed: {len(summaries)}")

            # Get embeddings
            all_embeddings = self.embedding_service.get_embeddings_batch(summaries)

            # Separate embeddings
            preferences_embedding = all_embeddings[preferences]
            cities_embeddings = {
                city: all_embeddings[content.summary]
                for city, content in cities_content.items()
            }

            print("Finding top cities")
            top_cities = self.embedding_service.get_top_cities(
                preferences_embedding,
                cities_embeddings,
                {k: v.summary for k, v in cities_content.items()},
                top_n=3,
                season=season,
                activity=primary_activity,
                activity_matcher=self.activity_matcher
            )

            selected_cities = [city for city, _ in top_cities]
            cities_chunks = {
                city: cities_content[city].chunks 
                for city in selected_cities 
                if city in cities_content
            }

            # Find relevant facts for all cities at once
            relevant_facts = {}
            all_city_facts = []
            
            # First collect top facts for all cities
            for city in selected_cities:
                if city in self.fact_embeddings:
                    city_facts = []
                    for category, facts in self.fact_embeddings[city].items():
                        for fact, fact_embedding in facts:
                            similarity = self.embedding_service.cosine_similarity(
                                preferences_embedding.reshape(1, -1),
                                fact_embedding.reshape(1, -1)
                            )[0][0]
                            city_facts.append((fact, similarity))
                    
                    # Sort facts by relevance and take top 15
                    city_facts.sort(key=lambda x: x[1], reverse=True)
                    top_facts = city_facts[:15]
                    facts_text = "\n".join([fact for fact, _ in top_facts])
                    all_city_facts.append({
                        "city": city,
                        "facts": facts_text
                    })
            
            if all_city_facts:
                # Process each city's facts in parallel
                async def summarize_city_facts(city_data):
                    messages = [
                        {"role": "system", "content": f"""На основе предпочтений пользователя и фактов о городе, выберите и перефразируйте 5-7 самых релевантных фактов.
Предпочтения пользователя: {preferences}

Правила:
1. Выбирайте факты, напрямую связанные с интересами и требованиями пользователя
2. Объединяйте похожие факты
3. Удаляйте избыточную информацию
4. Формулируйте факты кратко и четко
5. Сохраняйте только практически полезную информацию для планирования поездки"""},
                        {"role": "user", "content": city_data["facts"]}
                    ]
                    
                    response = await self.llm_service.client.chat.completions.create(
                        model=Config.LLM_MODEL,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=512
                    )
                    
                    return city_data["city"], [
                        (fact.strip(), 1.0) 
                        for fact in response.choices[0].message.content.split('\n')
                        if fact.strip()
                    ]
                
                # Create tasks for all cities
                tasks = [summarize_city_facts(city_data) for city_data in all_city_facts]
                
                # Run all tasks in parallel
                results = await asyncio.gather(*tasks)
                
                # Organize results
                relevant_facts = {city: facts for city, facts in results}

            # Final LLM summarization of recommendations
            summarized_facts = {}
            for city, facts in relevant_facts.items():
                messages = [
                    {"role": "system", "content": f"""Создайте краткое описание города, включая только релевантную информацию для запроса:

ОСНОВНЫЕ ФАКТЫ:
• Общая информация об инфраструктуре города
• Особенности и преимущества для запрошенного типа отдыха
• Уникальные характеристики города

Если в запросе указан сезон:
Температура в [сезон]: XX°C

Для пляжного отдыха - только популярные пляжи:
• [название пляжа]
• [название пляжа]

Для семейного отдыха - только развлечения для детей:
• [название места]
• [название места]

Для культурного туризма - только основные достопримечательности:
• [название места]
• [название места]

Для зимнего отдыха - только горнолыжная инфраструктура:
• [название объекта]
• [название объекта]

Для оздоровительного отдыха - только спа и санатории:
• [название объекта]
• [название объекта]

Запрос пользователя: {preferences}"""},
                    {"role": "user", "content": "\n".join([fact for fact, _ in facts])}
                ]
                
                response = await self.llm_service.client.chat.completions.create(
                    model=Config.LLM_MODEL,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=256
                )
                
                summarized_facts[city] = [(fact.strip(), 1.0) for fact in response.choices[0].message.content.split('\n') if fact.strip()]

            return cities_chunks, top_cities, preferences, available_tokens, summarized_facts

        except Exception as e:
            print(f"Error occurred in process_request: {str(e)}")
            print(f"Error type: {type(e)}")
            raise
