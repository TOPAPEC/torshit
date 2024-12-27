from wiki import WikiService
from embeddings import EmbeddingService
from llm import LLMService

class TravelAdvisor:
    def __init__(self):
        self.wiki_service = WikiService()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()

    async def process_request(self, user_input: str):
        try:
            cities_content = await self.wiki_service.get_all_cities_content()
            if not cities_content:
                return None, None, None, None

            embeddings = {
                city: self.embedding_service.get_embedding(content.summary)
                for city, content in cities_content.items()
            }

            preferences = self.llm_service.get_preferences(user_input)
            similar_cities = self.embedding_service.find_similar_cities(
                preferences, embeddings, {k: v.summary for k, v in cities_content.items()}
            )

            documents = [{"doc_id": idx, "title": city, "content": content.full_text}
                        for idx, (city, content) in enumerate(cities_content.items())]
            relevant_docs, final_answer = self.llm_service.get_rag_response(preferences, documents)

            return preferences, similar_cities, relevant_docs, final_answer
        except Exception as e:
            print(f"Error occurred: {e}")
            return None, None, None, None
