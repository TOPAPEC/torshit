import asyncio
from advisor import TravelAdvisor

async def process_and_evaluate_query(advisor, query, query_description):
    print(f"\n{'='*20} ТЕСТОВЫЙ ЗАПРОС: {query_description} {'='*20}")
    print(f"Запрос: {query}")
    
    try:
        cities_chunks, top_cities, preferences, available_tokens, relevant_facts = await advisor.process_request(query)
        cities_content = await advisor.wiki_service.get_all_cities_content()
        
        if not cities_chunks:
            print("❌ Не удалось обработать запрос")
            return
        
        print("\n📋 АНАЛИЗ ЗАПРОСА:")
        print(preferences)
        
        print("\n🎯 РЕКОМЕНДОВАННЫЕ ГОРОДА:")
        for city, score in top_cities:
            print(f"\n🏛️ {city} (Релевантность: {score:.3f})")
            if city in relevant_facts and relevant_facts[city]:
                print("\n📚 Интересные факты о городе:")
                for fact, relevance in relevant_facts[city]:
                    print(f"  • {fact}")
                print()
                        
    except Exception as e:
        print(f"❌ Ошибка при обработке запроса: {str(e)}")
        raise

async def main():
    advisor = TravelAdvisor()
    
    test_queries = [
        """Куда-нибудь подальше и где похолоднее. Где много комаров
        """,
        # """Мы с мужем очень устали на работе и хотим в августе просто полежать на пляже, 
        #    послушать шум волн. Последний раз были на море 3 года назад... Надоела эта московская 
        #    суета. Хочется теплого моря и мягкого песочка под ногами. На двоих можем потратить 
        #    около 100 тысяч, не больше.""",
        
        # """В честь нашей годовщины свадьбы решили устроить себе роскошный отпуск. 
        #    Хочется чего-то особенного - спа-процедуры, массажи, чтобы сервис был на высшем 
        #    уровне. Бюджет примерно 300 тысяч, хотим чтобы все было идеально.""",
        
        # """Дети (5 и 7 лет) все уши прожужжали про море и развлечения. Муж сейчас 
        #    между работами, так что бюджет ограничен - максимум 80 тысяч. Было бы здорово, 
        #    если бы там был аквапарк или другие развлечения для детей, а то они быстро 
        #    устают просто на пляже лежать.""",
        
        # """Недавно начала увлекаться историей архитектуры, насмотрелась роликов 
        #    на ютубе про разные исторические места. Хочу поехать куда-нибудь, где можно 
        #    походить по музеям и старинным улочкам. Только вот жару плохо переношу, 
        #    больше 25 градусов уже тяжело. На отпуск отложила 150 тысяч.""",
        
        # """Друзья позвали покататься на лыжах в январе. Я последний раз лет 5 назад 
        #    на горных лыжах стоял, но очень хочется попробовать снова. Могу потратить 
        #    до 200 тысяч, включая экипировку и все развлечения."""
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📊 Обработка запроса {i} из {len(test_queries)}")
        try:
            await process_and_evaluate_query(advisor, query, f"Тестовый запрос {i}")
            print(f"✅ Запрос {i} обработан успешно")
        except Exception as e:
            print(f"❌ Ошибка при обработке запроса {i}: {str(e)}")
            continue

if __name__ == "__main__":
    try:
        print("\n🚀 Запуск тестирования системы рекомендаций...")
        asyncio.run(main())
        print("\n✅ Тестирование завершено")
    except KeyboardInterrupt:
        print("\n❌ Программа завершена пользователем")
        raise
    except Exception as e:
        print(f"\n❌ Произошла ошибка: {e}")
        raise
