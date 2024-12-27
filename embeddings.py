import torch
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingService:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/ruBert-base")
        self.model = AutoModel.from_pretrained("sberbank-ai/ruBert-base")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, text: str):
        encoded_input = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return self.mean_pooling(model_output, encoded_input['attention_mask'])[0].numpy()

    def get_top_cities(self, user_preferences: str, embeddings: Dict, descriptions: Dict, top_n=2) -> List[Tuple[str, float]]:
        user_embedding = self.get_embedding(user_preferences)
        similarities = {
            city: cosine_similarity(user_embedding.reshape(1, -1), emb.reshape(1, -1))[0][0]
            for city, emb in embeddings.items()
        }
        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]

