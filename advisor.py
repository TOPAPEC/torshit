from wiki import WikiService
from embeddings import EmbeddingService
from llm import LLMService

class TravelAdvisor:
    def __init__(self, model_context_length: int = 10000):
        print("Initializing TravelAdvisor")
        self.wiki_service = WikiService()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService(model_context_length)
        self.context_manager = self.llm_service.context_manager

    async def process_request(self, user_input: str):
        try:
            # print("Starting request processing")
            print(f"User input: {user_input}")

            # print("Getting initial preferences")
            preferences = await self.llm_service.get_preferences(user_input)
            print(f"Extracted preferences: {preferences}")

            # print("Calculating available tokens")
            available_tokens = self.context_manager.get_available_tokens(preferences, is_rag=True)
            print(f"Available tokens: {available_tokens}")

            # Extract location type from preferences
            location_type = None
            pref_lines = preferences.split('\n')
            for i, line in enumerate(pref_lines):
                if 'МЕСТОПОЛОЖЕНИЕ:' in line or '**МЕСТОПОЛОЖЕНИЕ:**' in line:
                    for j in range(i + 1, len(pref_lines)):
                        if 'тип:' in pref_lines[j].lower():
                            type_line = pref_lines[j].lower()
                            if 'море' in type_line:
                                location_type = 'море'
                            elif 'горы' in type_line:
                                location_type = 'горы'
                            elif 'город' in type_line:
                                location_type = 'город'
                            break
                    break
            
            if not location_type:
                print("Location type not found in preferences, fetching all cities")
                cities_content = await self.wiki_service.get_all_cities_content()
            else:
                print(f"Fetching cities for type: {location_type}")
                cities_content = await self.wiki_service.get_cities_by_type(location_type)
            
            if not cities_content:
                print("No cities content found")
                return None, None, None, None, None

            # print("Cities content retrieved:")
            # for city, content in cities_content.items():
                # print(f"{city}: {content.summary}")
                # print("-" * 20)

            # print("Preparing for embeddings")
            summaries = [content.summary for content in cities_content.values()]
            summaries.append(preferences)
            print(f"Total summaries to embed: {len(summaries)}")

            print("Getting batch embeddings")
            all_embeddings = self.embedding_service.get_embeddings_batch(summaries)
            # print("Embeddings retrieved successfully")

            print("Separating embeddings")
            preferences_embedding = all_embeddings[preferences]
            cities_embeddings = {
                city: all_embeddings[content.summary]
                for city, content in cities_content.items()
            }
            print(f"Number of city embeddings: {len(cities_embeddings)}")

            print("Finding top cities")
            top_cities = self.embedding_service.get_top_cities(
                preferences_embedding,
                cities_embeddings,
                {k: v.summary for k, v in cities_content.items()},
                top_n=3
            )
            print(f"Top cities found: {[city for city, _ in top_cities]}")

            selected_cities = [city for city, _ in top_cities]
            cities_chunks = {
                city: cities_content[city].chunks 
                for city in selected_cities 
                if city in cities_content
            }
            # print(f"Chunks retrieved for cities: {list(cities_chunks.keys())}")

            return cities_chunks, top_cities, preferences, available_tokens

        except Exception as e:
            print(f"Error occurred in process_request: {str(e)}")
            print(f"Error type: {type(e)}")
            raise
