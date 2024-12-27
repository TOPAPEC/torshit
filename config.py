class Config:
    OPENAI_KEY = 'xxx'
    ENDPOINT = 'http://localhost:8000/v1'
    LLM_MODEL = 'Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24'
    SYSTEM_PROMPT = """Ты - помощник по анализу туристических предпочтений..."""
    GROUNDED_SYSTEM_PROMPT = "Your task is to answer the user's questions..."
    RESORT_CITIES = ['Сочи', 'Анапа', 'Геленджик', 'Ялта', 'Алушта', 'Евпатория']

