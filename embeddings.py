import torch
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

import torch
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import mmh3
import pandas as pd
from pathlib import Path

class EmbeddingService:
    def __init__(self, cache_file: str = "emb_service_cache.parquet"):
        self.tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/ruBert-base")
        self.model = AutoModel.from_pretrained("sberbank-ai/ruBert-base")
        self.cache_file = Path(cache_file)
        self.cache_df = self._load_cache()

    def _load_cache(self) -> pd.DataFrame:
        """Load cache from parquet file or create new cache."""
        if self.cache_file.exists():
            return pd.read_parquet(self.cache_file)
        return pd.DataFrame(columns=['text_hash', 'text', 'embedding'])

    def _compute_hash(self, text: str) -> str:
        """Compute MurmurHash3 hash of input text."""
        return str(mmh3.hash128(text))

    def _save_to_cache(self, text: str, embedding: np.ndarray):
        """Save embedding to cache."""
        text_hash = self._compute_hash(text)

        # Check if hash already exists
        if text_hash not in self.cache_df['text_hash'].values:
            new_row = pd.DataFrame({
                'text_hash': [text_hash],
                'text': [text],
                'embedding': [embedding]
            })
            self.cache_df = pd.concat([self.cache_df, new_row], ignore_index=True)
            self.cache_df.to_parquet(self.cache_file)

    def _load_from_cache(self, text: str) -> np.ndarray:
        """Load embedding from cache if it exists."""
        text_hash = self._compute_hash(text)
        cached_item = self.cache_df[self.cache_df['text_hash'] == text_hash]
        if not cached_item.empty:
            return cached_item.iloc[0]['embedding']
        return None

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching."""
        # Try to load from cache first
        cached_embedding = self._load_from_cache(text)
        if cached_embedding is not None:
            return cached_embedding

        # Compute new embedding if not in cache
        encoded_input = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embedding = self.mean_pooling(model_output, encoded_input['attention_mask'])[0].numpy()

        # Save to cache
        self._save_to_cache(text, embedding)

        return embedding

    def get_top_cities(
        self, 
        preferences: str,
        cities_embeddings: Dict[str, np.ndarray],
        cities_descriptions: Dict[str, str],
        top_n: int = 2
    ) -> List[Tuple[str, float]]:
        """
        Get top cities based on similarity with user preferences.

        Args:
            preferences: User preferences text
            cities_embeddings: Dictionary of city embeddings {city_name: embedding}
            cities_descriptions: Dictionary of city descriptions {city_name: description}
            top_n: Number of top cities to return

        Returns:
            List of tuples (city_name, similarity_score)
        """
        print(f"Computing similarities for {len(cities_embeddings)} cities")

        similarities = {}
        preferences_embedding = cities_embeddings[preferences]

        for city, city_embedding in cities_embeddings.items():
            # Skip the preferences embedding from comparison
            if city == preferences:
                continue

            similarity = cosine_similarity(
                preferences_embedding.reshape(1, -1),
                city_embedding.reshape(1, -1)
            )[0][0]

            similarities[city] = similarity
            print(f"Similarity for {city}: {similarity:.4f}")

        top_cities = sorted(
            similarities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]

        print(f"Selected top {top_n} cities: {[city for city, score in top_cities]}")
        return top_cities

    def clear_cache(self):
        """Clear the embedding cache."""
        if self.cache_file.exists():
            self.cache_file.unlink()
        self.cache_df = pd.DataFrame(columns=['text_hash', 'text', 'embedding'])

    def get_embeddings_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Get embeddings for multiple texts efficiently."""
        text_to_hash = {text: self._compute_hash(text) for text in texts}
        hash_to_text = {h: t for t, h in text_to_hash.items()}

        all_hashes = set(text_to_hash.values())
        cached_mask = self.cache_df['text_hash'].isin(all_hashes)
        cached_entries = self.cache_df[cached_mask]

        result = {}
        for _, row in cached_entries.iterrows():
            text = hash_to_text[row['text_hash']]
            result[text] = row['embedding']

        texts_to_compute = [text for text in texts if text not in result]
        if texts_to_compute:
            encoded_input = self.tokenizer(
                texts_to_compute,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )

            with torch.no_grad():
                model_output = self.model(**encoded_input)

            embeddings = self.mean_pooling(
                model_output,
                encoded_input['attention_mask']
            ).numpy()

            # Save new embeddings to cache and result
            for text, embedding in zip(texts_to_compute, embeddings):
                self._save_to_cache(text, embedding)
                result[text] = embedding

        return result 
