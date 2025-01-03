from typing import Dict, List, Set, Optional, Tuple
import re

ACTIVITIES = {
    'winter_sports': {
        'keywords': [
            'горнолыжн', 'лыж', 'сноуборд', 'зимний спорт', 'зимний отдых',
            'катание на лыжах', 'горный курорт', 'зимние развлечения',
            'катание с гор', 'зимние виды спорта', 'горнолыжный курорт',
            'красная поляна', 'роза хутор', 'домбай', 'приэльбрусье', 'архыз',
            'шерегеш'  # Added all major ski resorts
        ],
        'required_facilities': {
            'ski_slopes': ['горнолыжн', 'лыжн', 'трасс', 'склон', 'спуск'],
            'ski_lifts': ['подъемник', 'канатная дорога', 'фуникулер', 'бугель'],
            'equipment_rental': ['прокат', 'экипировк', 'снаряжени', 'инструктор'],
            'winter_infrastructure': ['красная поляна', 'роза хутор', 'домбай', 'приэльбрусье', 'архыз', 'шерегеш']
        },
        'required_conditions': {
            'terrain': ['гор', 'склон', 'хребет', 'вершин'],
            'weather': ['снег', 'зимн', 'холодн', 'сезон']
        },
        'incompatible_features': ['аквапарк', 'пляж'],
        'season': 'winter',
        'min_facilities_required': 2,
        'strict_matching': True,
        'required_keywords': ['горнолыжн', 'лыж', 'сноуборд']
    },
    'beach_vacation': {
        'keywords': [
            'пляж', 'море', 'купаться', 'загорать', 'морской курорт',
            'пляжный отдых', 'морской отдых', 'приморский курорт',
            'песчаный пляж', 'морское побережье', 'пляжный сезон',
            'набережн', 'побережь', 'залив', 'бухт', 'черное море',
            'каспийское море', 'балтийское море'
        ],
        'required_facilities': {
            'beaches': ['пляж', 'набережн', 'побережь', 'море', 'залив', 'бухт'],
            'water_activities': ['купальн', 'аквапарк', 'дайвинг', 'серфинг', 'водные развлечения', 'отдых'],
            'beach_services': ['лежак', 'зонт', 'кабинк', 'пляжный сервис', 'курорт']
        },
        'required_conditions': {
            'water': ['море', 'океан', 'залив', 'бухт', 'побережь'],
            'weather': ['солнечн', 'тепл', 'жарк', 'лет']
        },
        'incompatible_features': ['горнолыжн', 'снег'],
        'season': 'summer',
        'min_facilities_required': 1
    },
    'cultural_tourism': {
        'keywords': [
            'музей', 'историческ', 'культурн', 'экскурси', 'достопримечательност',
            'архитектур', 'памятник', 'храм', 'собор', 'культурное наследие',
            'исторический центр', 'старый город'
        ],
        'required_facilities': {
            'museums': ['музей', 'галере', 'выставк', 'экспозиц'],
            'historical_sites': ['памятник', 'храм', 'монастыр', 'крепост', 'дворец', 'собор'],
            'theaters': ['театр', 'филармони', 'концертн', 'опер']
        },
        'required_conditions': {
            'infrastructure': ['экскурси', 'туристическ', 'культурн'],
        },
        'incompatible_features': [],  # Cultural tourism compatible with all
        'season': None,
        'min_facilities_required': 1
    },
    'family_vacation': {
        'keywords': [
            'с детьми', 'семейный', 'детский', 'аквапарк', 'семейный отдых',
            'для всей семьи', 'детские развлечения', 'семейный курорт',
            'детская площадка', 'детские аттракционы'
        ],
        'required_facilities': {
            'entertainment': ['аквапарк', 'парк', 'зоопарк', 'цирк', 'аттракцион'],
            'children_activities': ['детск', 'игров', 'развлекательн', 'семейн'],
            'safety': ['пляж', 'променад', 'парк', 'безопасн']
        },
        'required_conditions': {
            'infrastructure': ['инфраструктур', 'благоустро', 'удобств'],
            'accessibility': ['транспорт', 'добраться', 'доступн']
        },
        'incompatible_features': [],  # Family vacation can be anywhere
        'season': None,
        'min_facilities_required': 2
    },
    'spa_wellness': {
        'keywords': [
            'спа', 'оздоровительный', 'санаторий', 'лечебный курорт',
            'оздоровление', 'wellness', 'термальные источники', 'грязелечение',
            'массаж', 'релакс', 'оздоровительные процедуры', 'кисловодск',
            'пятигорск', 'ессентуки', 'железноводск', 'минеральные воды'
        ],
        'required_facilities': {
            'spa_centers': ['спа', 'массаж', 'процедур', 'оздоровительн', 'wellness'],
            'medical': ['санатори', 'лечебн', 'терапи', 'реабилитац', 'профилактори'],
            'wellness': ['термальн', 'источник', 'грязелечени', 'минеральн', 'нарзан', 'бювет']
        },
        'required_conditions': {
            'environment': ['чист', 'экологичн', 'природ', 'климат', 'воздух'],
            'infrastructure': ['медицинск', 'оздоровительн', 'лечебн', 'санаторн']
        },
        'incompatible_features': [],  # Spa/wellness can be anywhere
        'season': None,
        'min_facilities_required': 2,  # Increased to require more spa facilities
        'strict_matching': True,  # Added to require specific spa-related matches
        'required_keywords': ['спа', 'санатори', 'лечебн', 'оздоровительн']  # Must have at least one of these
    }
}

class ActivityMatcher:
    def __init__(self, llm_service=None):
        self.llm_service = llm_service

    def _rule_based_extract(self, text: str) -> List[Tuple[str, float]]:
        """Extract activities using rule-based matching with confidence scores"""
        text = text.lower()
        matches = []
        
        for activity_name, activity_data in ACTIVITIES.items():
            confidence = 0.0
            keyword_matches = 0
            
            # Check keywords
            for keyword in activity_data['keywords']:
                if keyword in text:
                    keyword_matches += 1
                    confidence += 1.0 / len(activity_data['keywords'])
            
            # Check required facilities
            facilities_matched = 0
            total_facilities = sum(len(keywords) for keywords in activity_data['required_facilities'].values())
            
            for facility_type, keywords in activity_data['required_facilities'].items():
                for keyword in keywords:
                    if keyword in text:
                        facilities_matched += 1
                        confidence += 0.5 / total_facilities
            
            # Only include activities with significant confidence
            if confidence > 0.2:  # At least 20% confidence
                matches.append((activity_name, min(confidence, 1.0)))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)

    async def extract_activity_llm(self, text: str) -> Optional[str]:
        """Extract activity using LLM"""
        if not self.llm_service:
            return None
            
        prompt = f"""Определите основной тип активности для путешествия из текста.
        Варианты:
        - winter_sports (зимние виды спорта, горные лыжи)
        - beach_vacation (пляжный отдых, море)
        - cultural_tourism (культурный туризм, музеи, достопримечательности)
        - family_vacation (семейный отдых, развлечения для детей)
        - spa_wellness (спа, оздоровительный отдых)
        
        Если активность не подходит ни под одну категорию, верните null.
        Отвечайте одним словом - названием категории на английском или null."""
        
        try:
            response = await self.llm_service.extract_activity_llm(text, prompt)
            activity = response.strip().lower()
            return activity if activity in ACTIVITIES else None
        except Exception as e:
            print(f"Error in LLM activity extraction: {e}")
            return None

    async def get_activities(self, text: str) -> List[Tuple[str, float]]:
        """Get activities with confidence scores using both rule-based and LLM methods"""
        # First try rule-based extraction
        rule_based_matches = self._rule_based_extract(text)
        
        # If we have high confidence matches, return them
        if rule_based_matches and rule_based_matches[0][1] > 0.6:
            return rule_based_matches
        
        # Try LLM as fallback
        llm_activity = await self.extract_activity_llm(text)
        if llm_activity:
            # Combine LLM result with rule-based matches
            llm_confidence = 0.8  # High confidence in LLM result
            
            # Check if LLM activity is already in rule-based matches
            for i, (activity, confidence) in enumerate(rule_based_matches):
                if activity == llm_activity:
                    # Boost confidence of existing match
                    rule_based_matches[i] = (activity, min(confidence + 0.2, 1.0))
                    return rule_based_matches
            
            # Add LLM result if not in rule-based matches
            return [(llm_activity, llm_confidence)] + rule_based_matches
        
        return rule_based_matches

    def get_activity_score(self, city_text: str, activity: str) -> float:
        """Calculate how well a city matches an activity's requirements"""
        if activity not in ACTIVITIES:
            return 0.0
            
        activity_data = ACTIVITIES[activity]
        base_score = 0.0
        city_text = city_text.lower()
        
        # Check incompatible features first
        if 'incompatible_features' in activity_data:
            if any(feature in city_text for feature in activity_data['incompatible_features']):
                return 0.0
        
        # For activities with strict matching, require at least one keyword match
        if activity_data.get('strict_matching', False):
            if 'required_keywords' in activity_data:
                if not any(keyword in city_text for keyword in activity_data['required_keywords']):
                    return 0.0
            elif not any(keyword in city_text for keyword in activity_data['keywords']):
                return 0.0
        
        # Calculate facility score
        facilities_score = 0.0
        facilities_found = 0
        for facility_type, keywords in activity_data['required_facilities'].items():
            facility_matches = sum(1 for keyword in keywords if keyword in city_text)
            if facility_matches > 0:
                facilities_found += 1
                facilities_score += facility_matches / len(keywords)
        
        # Check minimum facilities requirement
        if facilities_found < activity_data['min_facilities_required']:
            return 0.0
        
        base_score += facilities_score / len(activity_data['required_facilities'])
        
        # Calculate conditions score
        if 'required_conditions' in activity_data:
            conditions_score = 0.0
            for condition_type, keywords in activity_data['required_conditions'].items():
                condition_matches = sum(1 for keyword in keywords if keyword in city_text)
                conditions_score += condition_matches / len(keywords)
            base_score += conditions_score / len(activity_data['required_conditions'])
        
        # Normalize final score
        return min(base_score / 2, 1.0)  # Average of facilities and conditions scores
