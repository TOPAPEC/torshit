import asyncio
from advisor import TravelAdvisor

async def process_and_evaluate_query(advisor, query, query_description):
    print(f"\n{'='*20} ТЕСТОВЫЙ ЗАПРОС: {query_description} {'='*20}")
    print(f"Запрос: {query}")
    
    try:
        cities_chunks, top_cities, preferences, available_tokens = await advisor.process_request(query)
        cities_content = await advisor.wiki_service.get_all_cities_content()
        
        if not cities_chunks:
            print("❌ Не удалось обработать запрос")
            return
        
        print("\n📋 АНАЛИЗ ЗАПРОСА:")
        print(preferences)
        
        print("\n🎯 РЕКОМЕНДОВАННЫЕ ГОРОДА:")
        for city, score in top_cities:
            print(f"\n🏛️ {city} (Релевантность: {score:.3f})")
            if city in cities_chunks:
                all_text = " ".join(cities_chunks[city])
                        
                # Climate information
                if 'температура' in all_text.lower() or any(month in all_text.lower() for month in ['январ', 'феврал', 'март', 'апрел', 'май', 'июн', 'июл', 'август', 'сентябр', 'октябр', 'ноябр', 'декабр']):
                    climate_info = "🌡️ "
                    for sentence in all_text.split('.'):
                        if any(month in sentence.lower() for month in ['январ', 'феврал', 'март', 'апрел', 'май', 'июн', 'июл', 'август', 'сентябр', 'октябр', 'ноябр', 'декабр']) and ('температура' in sentence.lower() or 'градус' in sentence.lower()):
                            climate_info += sentence.strip() + ". "
                            break
                    if not any(month in climate_info.lower() for month in ['январ', 'феврал', 'март', 'апрел', 'май', 'июн', 'июл', 'август', 'сентябр', 'октябр', 'ноябр', 'декабр']):
                        for sentence in all_text.split('.'):
                            if 'температура' in sentence.lower():
                                climate_info += sentence.strip() + ". "
                                break
                    print(climate_info.strip())
                        
                # OpenStreetMap POI information
                if city in cities_content and cities_content[city].pois:
                    pois = cities_content[city].pois
                    
                    # Beaches
                    if pois.beaches:
                        print("🏖️ Пляжи:")
                        for beach in pois.beaches:
                            print(f"  • {beach.name}")
                    elif 'пляж' in all_text.lower() or 'море' in all_text.lower():
                        beach_info = "🏖️ "
                        if 'песчаный' in all_text.lower():
                            beach_info += "Есть песчаные пляжи. "
                        if 'купальный сезон' in all_text.lower():
                            for sentence in all_text.split('.'):
                                if 'купальный сезон' in sentence.lower():
                                    beach_info += sentence.strip() + ". "
                                    break
                        print(beach_info.strip())
                    
                    # Tourist attractions
                    if pois.tourist_attractions:
                        print("🏛️ Достопримечательности:")
                        for attraction in pois.tourist_attractions[:5]:
                            print(f"  • {attraction.name}")
                    
                    # Entertainment
                    if pois.entertainment:
                        print("🎡 Развлечения:")
                        for venue in pois.entertainment[:5]:
                            print(f"  • {venue.name}")
                    
                    # Sports facilities
                    if pois.sports_facilities:
                        print("🏃 Спортивные объекты:")
                        for facility in pois.sports_facilities[:5]:
                            print(f"  • {facility.name}")
                
                # Tourist infrastructure from Wikipedia
                if 'туристи' in all_text.lower():
                    for sentence in all_text.split('.'):
                        if 'туристи' in sentence.lower():
                            print("🏨 " + sentence.strip() + ".")
                            break
    except Exception as e:
        print(f"❌ Ошибка при обработке запроса: {str(e)}")
        raise

async def main():
    advisor = TravelAdvisor()
    
    test_queries = [
        ("""Хочу поехать на море в августе, чтобы было тепло около 25-30 градусов 
            и песчаный пляж. Бюджет до 100000 рублей.""", 
         "Пляжный отдых со средним бюджетом"),
        
        ("""Ищу премиальный курорт с развитой туристической инфраструктурой, 
            спа-центрами и высоким уровнем сервиса. Бюджет до 300000 рублей.""",
         "Премиальный отдых"),
        
        ("""Нужен недорогой семейный отдых с детьми, желательно с аквапарком 
            или развлечениями для детей. Бюджет до 80000 рублей.""",
         "Семейный отдых с ограниченным бюджетом"),
        
        ("""Интересует отдых с посещением исторических мест и музеев, 
            желательно не жарче 25 градусов. Бюджет до 150000 рублей.""",
         "Культурный туризм"),
        
        ("""Хочу активный зимний отдых в горах с горнолыжными трассами 
            в январе. Бюджет до 200000 рублей.""",
         "Зимний активный отдых")
    ]
    
    for i, (query, description) in enumerate(test_queries, 1):
        print(f"\n📊 Обработка запроса {i} из {len(test_queries)}")
        try:
            await process_and_evaluate_query(advisor, query, description)
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
