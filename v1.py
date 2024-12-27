import asyncio

from advisor import TravelAdvisor

async def main():
    advisor = TravelAdvisor()
    user_input = """Хочу поехать на море в августе, чтобы было тепло около 25-30 градусов 
                    и песчаный пляж. Бюджет до 100000 рублей."""

    result = await advisor.process_request(user_input)

    if result:
        print("\n=== АНАЛИЗ ЗАПРОСА ===")
        print(result["preferences"])

        print("\n=== РЕКОМЕНДОВАННЫЕ ГОРОДА ===")
        for city_info in result["top_cities"]:
            print(f"\n{city_info['city']} (Score: {city_info['similarity_score']:.3f}):")
            print(city_info['summary'])

        print("\n=== РЕЛЕВАНТНЫЕ ДОКУМЕНТЫ ===")
        print(result["relevant_docs"])

        print("\n=== ИТОГОВАЯ РЕКОМЕНДАЦИЯ ===")
        print(result["final_answer"])
    else:
        print("Не удалось обработать запрос")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nПрограмма завершена пользователем")
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")

