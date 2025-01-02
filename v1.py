import asyncio

from advisor import TravelAdvisor

async def main():
    advisor = TravelAdvisor()
    user_input = """Хочу поехать на море в августе, чтобы было тепло около 25-30 градусов 
                    и песчаный пляж. Бюджет до 100000 рублей."""

    cities_chunks, top_cities, preferences, available_tokens = await advisor.process_request(user_input)

    if cities_chunks:
        print("\n=== АНАЛИЗ ЗАПРОСА ===")
        print(preferences)

        print("\n=== РЕКОМЕНДОВАННЫЕ ГОРОДА ===")
        for city, score in top_cities:
            print(f"\n{city} (Score: {score:.3f})")
            if city in cities_chunks:
                relevant_info = []
                for chunk in cities_chunks[city]:
                    if any(keyword in chunk.lower() for keyword in ['пляж', 'море', 'погода', 'температура', 'климат']):
                        relevant_info.append(chunk.split('\n\n', 1)[1] if '\n\n' in chunk else chunk)
                # if relevant_info:
                    # print("Релевантная информация:")
                    # print('\n'.join(relevant_info[:2]))  # Show first 2 relevant chunks
    else:
        print("Не удалось обработать запрос")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nПрограмма завершена пользователем")
        raise
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")
        raise
