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
            # Get cities content
            cities_content = await self.wiki_service.get_all_cities_content()
            if not cities_content:
                return None, None, None, None

            # Get embeddings and top cities
            embeddings = {
                city: self.embedding_service.get_embedding(content.summary)
                for city, content in cities_content.items()
            }

            preferences = await self.llm_service.get_preferences(user_input)
            top_cities = self.embedding_service.get_top_cities(
                preferences, embeddings, {k: v.summary for k, v in cities_content.items()}
            )

            # Process chunks for top cities in parallel
            selected_cities = [city for city, _ in top_cities]
            all_chunks = []
            chunk_preferences = {}

            for city in selected_cities:
                if city in cities_content:
                    chunks = cities_content[city].chunks
                    chunk_summaries = await self.llm_service.process_chunks(chunks)
                    chunk_preferences[city] = chunk_summaries

                    # Create enhanced chunks with preferences
                    for chunk, summary in zip(chunks, chunk_summaries):
                        all_chunks.append({
                            "city": city,
                            "content": chunk,
                            "extracted_info": summary
                        })

            # Create RAG documents with enhanced information
            documents = [
                {
                    "doc_id": idx,
                    "title": chunk["city"],
                    "content": f"{chunk['extracted_info']}"
                }
                for idx, chunk in enumerate(all_chunks)
            ]

            relevant_docs, final_answer = await self.llm_service.get_rag_response(preferences, documents)

            return preferences, top_cities, relevant_docs, final_answer
        except Exception as e:
            print(f"Error occurred: {e}")
            return None, None, None, None

