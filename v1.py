import asyncio

from advisor import TravelAdvisor

async def main():
    # Example user request
    user_input = """Хочу поехать на море в августе, чтобы было тепло около 25-30 градусов 
                    и песчаный пляж. Бюджет до 100000 рублей."""

    # Initialize travel advisor
    advisor = TravelAdvisor()

    # Process request and get results
    preferences, recommendations, relevant_docs, final_answer = await advisor.process_request(user_input)

    # Print results if successful
    if preferences:
        print("\n=== АНАЛИЗ ЗАПРОСА ===")
        print(preferences)

        print("\n=== РЕКОМЕНДОВАННЫЕ ГОРОДА ===")
        for city, description in recommendations:
            print(f"\n{city}:")
            print(description[:200] + "...")

        print("\n=== РЕЛЕВАНТНЫЕ ДОКУМЕНТЫ ===")
        print(relevant_docs)

        print("\n=== ИТОГОВАЯ РЕКОМЕНДАЦИЯ ===")
        print(final_answer)
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

