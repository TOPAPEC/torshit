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
            print("Getting top cities")
            top_cities = self.embedding_service.get_top_cities(
                preferences, embeddings, {k: v.summary for k, v in cities_content.items()}
            )

            selected_cities = [city for city, _ in top_cities]
            print("Creating RAG docs for most relevant cities")
            documents = self.llm_service.create_rag_documents(cities_content, selected_cities)
            print("Performing RAG")
            relevant_docs, final_answer = self.llm_service.get_rag_response(preferences, documents)

            return preferences, top_cities, relevant_docs, final_answer
        except Exception as e:
            print(f"Error occurred: {e}")
            return None, None, None, None

