from wiki import WikiService
from embeddings import EmbeddingService
from llm import LLMService

class TravelAdvisor:
    def __init__(self, model_context_length: int = 10000):
        self.wiki_service = WikiService()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService(model_context_length)
        self.context_manager = self.llm_service.context_manager

    async def process_request(self, user_input: str):
        try:
            # Get initial preferences to include their tokens in calculations
            preferences = await self.llm_service.get_preferences(user_input)
            available_tokens = self.context_manager.get_available_tokens(preferences, is_rag=True)

            # Get cities content
            cities_content = await self.wiki_service.get_all_cities_content()
            if not cities_content:
                return None, None, None, None

            for city, content in cities_content.items():
                print(f"{city}: {content}")
                print("-" * 20)

            # Get embeddings and top cities (limit to 2-3 cities to manage context)
            embeddings = {
                city: self.embedding_service.get_embedding(content.summary)
                for city, content in cities_content.items()
            }


            # Get top cities with scores
            top_cities = self.embedding_service.get_top_cities(
                preferences, 
                embeddings, 
                {k: v.summary for k, v in cities_content.items()},
                top_n=2  # Limiting to top 2 cities for better context management
            )

            # Prepare chunks for selected cities
            selected_cities = [city for city, _ in top_cities]
            cities_chunks = {
                city: cities_content[city].chunks 
                for city in selected_cities 
                if city in cities_content
            }

            # Calculate approximate tokens per city
            total_cities = len(cities_chunks)
            if total_cities == 0:
                return None, None, None, None

            # Prepare RAG documents with context management
            documents = await self.llm_service.prepare_rag_documents(
                cities_chunks,
                preferences
            )

            # Final RAG processing
            relevant_docs, final_answer = await self.llm_service.get_rag_response(
                preferences,
                documents
            )

            # Prepare result summary
            result_summary = {
                "preferences": preferences,
                "top_cities": [
                    {
                        "city": city,
                        "similarity_score": score,
                        "summary": next((doc["content"] for doc in documents if doc["title"] == city), None)
                    }
                    for city, score in top_cities
                ],
                "relevant_docs": relevant_docs,
                "final_answer": final_answer
            }

            return result_summary

        except Exception as e:
            print(f"Error occurred: {e}")
            return None

