import torch
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from seasons import SEASONS
import numpy as np
import mmh3
import pandas as pd
from pathlib import Path
from tqdm import tqdm

class EmbeddingService:
    def __init__(self, cache_file: str = "emb_service_cache.parquet", batch_size: int = 128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model with optimizations
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sberbank-ai/ruBert-base",
            use_fast=True,  # Use fast tokenizer
            model_max_length=512  # Set max length upfront
        )
        self.model = AutoModel.from_pretrained(
            "sberbank-ai/ruBert-base",
            return_dict=True  # Changed to True to get structured output
        ).to(self.device).eval()  # Set to eval mode
        
        self.batch_size = batch_size
        self.cache_file = Path(cache_file)
        self.cache_df = self._load_cache()
        self._pending_cache_updates = []

    def _load_cache(self) -> pd.DataFrame:
        """Load cache from parquet file or create new cache."""
        if self.cache_file.exists():
            return pd.read_parquet(self.cache_file)
        return pd.DataFrame(columns=['text_hash', 'text', 'embedding'])

    def _compute_hash(self, text: str) -> str:
        """Compute MurmurHash3 hash of input text."""
        return str(mmh3.hash128(text))

    def _save_to_cache(self, text: str, embedding: np.ndarray):
        """Queue embedding for batch cache update."""
        text_hash = self._compute_hash(text)
        if text_hash not in self.cache_df['text_hash'].values:
            self._pending_cache_updates.append({
                'text_hash': text_hash,
                'text': text,
                'embedding': embedding
            })

    def _flush_cache_updates(self):
        """Write all pending cache updates to disk."""
        if self._pending_cache_updates:
            new_rows = pd.DataFrame(self._pending_cache_updates)
            self.cache_df = pd.concat([self.cache_df, new_rows], ignore_index=True)
            self.cache_df.to_parquet(self.cache_file)
            self._pending_cache_updates = []

    def _load_from_cache(self, text: str) -> np.ndarray:
        """Load embedding from cache if it exists."""
        text_hash = self._compute_hash(text)
        cached_item = self.cache_df[self.cache_df['text_hash'] == text_hash]
        if not cached_item.empty:
            return cached_item.iloc[0]['embedding']
        return None

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching."""
        cached_embedding = self._load_from_cache(text)
        if cached_embedding is not None:
            return cached_embedding
        return self.get_embeddings_batch([text])[text]

    def get_top_cities(
        self, 
        preferences_embedding: np.ndarray,
        cities_embeddings: Dict[str, np.ndarray],
        cities_descriptions: Dict[str, str],
        top_n: int = 2,
        season: Optional[str] = None,
        activity: Optional[str] = None,
        activity_matcher = None
    ) -> List[Tuple[str, float]]:
        """Get top cities based on similarity with user preferences."""
        print(f"Computing similarities for {len(cities_embeddings)} cities")
        similarities = {}

        # Convert to tensors for batch processing
        pref_tensor = torch.from_numpy(preferences_embedding).to(self.device)
        city_tensors = {
            city: torch.from_numpy(emb).to(self.device)
            for city, emb in cities_embeddings.items()
        }

        # Batch compute similarities
        for city, city_tensor in city_tensors.items():
            similarity = torch.cosine_similarity(
                pref_tensor.reshape(1, -1),
                city_tensor.reshape(1, -1)
            ).item()

            city_text = cities_descriptions[city].lower()
            
            if season and season in SEASONS:
                season_data = SEASONS[season]
                seasonal_boost = 0.0
                
                keyword_matches = sum(1 for keyword in season_data['keywords'] 
                                   if keyword in city_text)
                seasonal_boost += 0.05 * keyword_matches
                
                import re
                temp_matches = re.finditer(r'температура.*?(-?\d+)', city_text)
                for match in temp_matches:
                    temp = int(match.group(1))
                    if season_data['temp_range'][0] <= temp <= season_data['temp_range'][1]:
                        seasonal_boost += 0.1
                        break
                
                similarity *= (1 + seasonal_boost)
            
            if activity and activity_matcher:
                activity_score = activity_matcher.get_activity_score(city_text, activity)
                if activity_score > 0:
                    activity_boost = 0.5 * activity_score
                    similarity *= (1 + activity_boost)
                else:
                    similarity *= 0.5

            similarities[city] = similarity

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
        self._pending_cache_updates = []

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between two vectors."""
        return cosine_similarity(a, b)

    def get_embeddings_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Get embeddings for multiple texts efficiently using batching."""
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
            print("Pre-tokenizing all texts...")
            # Pre-tokenize all texts with padding
            all_encoded = self.tokenizer(
                texts_to_compute,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Process in batches with progress bar
            total_batches = (len(texts_to_compute) + self.batch_size - 1) // self.batch_size
            with tqdm(total=total_batches, desc="Computing embeddings") as pbar:
                for i in range(0, len(texts_to_compute), self.batch_size):
                    batch_end = min(i + self.batch_size, len(texts_to_compute))
                    batch_texts = texts_to_compute[i:batch_end]
                    
                    # Extract batch tensors
                    batch_input = {
                        'input_ids': all_encoded['input_ids'][i:batch_end].to(self.device),
                        'attention_mask': all_encoded['attention_mask'][i:batch_end].to(self.device),
                    }

                    with torch.no_grad(), torch.cuda.amp.autocast():  # Enable automatic mixed precision
                        model_output = self.model(**batch_input)

                    embeddings = self.mean_pooling(
                        model_output,
                        batch_input['attention_mask']
                    ).cpu().numpy()

                    # Queue embeddings for cache update
                    for text, embedding in zip(batch_texts, embeddings):
                        self._save_to_cache(text, embedding)
                        result[text] = embedding
                    
                    pbar.update(1)
                    
                    # Clear GPU cache periodically
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Flush all cache updates at once
            print("Saving to cache...")
            self._flush_cache_updates()

        return result
