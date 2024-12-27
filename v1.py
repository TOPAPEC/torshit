import openai
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer 
from sklearn.metrics.pairwise import cosine_similarity
import wikipediaapi
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

openai.api_key = 'xxx'
endpoint = 'http://localhost:8000/v1'
model = 'Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24'
openai.api_base = endpoint

system_prompt = """Ты - помощник по анализу туристических предпочтений. Выдели из запроса пользователя следующие параметры:

1. Тип отдыха (море/горы/экскурсии и т.д.)

2. Желаемая погода (температура, осадки)

3. Даты поездки

4. Бюджет (если указан)

5. Дополнительные пожелания

Формат ответа:
Тип отдыха: 
Погода:
Даты:
Бюджет:
Дополнительно:"""

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_city_descriptions():
    wiki = wikipediaapi.Wikipedia('ru')
    resort_cities = [
        'Сочи', 'Анапа', 'Геленджик', 'Ялта', 'Алушта', 'Евпатория',
        'Кисловодск', 'Пятигорск', 'Домбай', 'Шерегеш', 'Байкальск',
        'Светлогорск', 'Зеленоградск', 'Калининград'
    ]
    descriptions = {}
    for city in resort_cities:
        page = wiki.page(city)
        if page.exists():
            descriptions[city] = page.summary
    return descriptions

def create_embeddings():
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/ruBert-base")
    model = AutoModel.from_pretrained("sberbank-ai/ruBert-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    descriptions = get_city_descriptions()
    embeddings = {}

    for city, desc in descriptions.items():
        encoded_input = tokenizer(desc, padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings[city] = sentence_embeddings[0].cpu().numpy()

    return embeddings, descriptions

def find_similar_cities(user_preferences, embeddings, descriptions, top_n=3):
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/ruBert-base")
    model = AutoModel.from_pretrained("sberbank-ai/ruBert-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    encoded_input = tokenizer(user_preferences, padding=True, truncation=True, max_length=512, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**encoded_input)

    user_embedding = mean_pooling(model_output, encoded_input['attention_mask'])[0].cpu().numpy()

    similarities = {}
    for city, emb in embeddings.items():
        similarity = cosine_similarity(
            user_embedding.reshape(1, -1),
            emb.reshape(1, -1)
        )[0][0]
        similarities[city] = similarity

    sorted_cities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return [(city, descriptions[city]) for city, _ in sorted_cities[:top_n]]

def process_user_request(user_input):
    response = openai.ChatCompletion.create(
        model=model,
        temperature=0.0,
        frequency_penalty=0.0,
        max_tokens=2048,
        top_p=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )

    preferences = response["choices"][0]["message"]["content"]
    embeddings, descriptions = create_embeddings()
    similar_cities = find_similar_cities(preferences, embeddings, descriptions)

    return preferences, similar_cities

if __name__ == "__main__":
    user_input = "Хочу поехать на море в августе, чтобы было тепло около 25-30 градусов и песчаный пляж. Бюджет до 100000 рублей."
    preferences, recommendations = process_user_request(user_input)
    print("Выделенные предпочтения:")
    print(preferences)
    print("\nРекомендованные города:")
    for city, description in recommendations:
        print(f"\n{city}:")
        print(description)

