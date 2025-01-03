from typing import Dict, List, Tuple, Optional

SEASONS = {
    'winter': {
        'months': [12, 1, 2],
        'keywords': ['зима', 'зимний', 'лыжи', 'горнолыжный'],
        'temp_range': (-30, 5),
    },
    'spring': {
        'months': [3, 4, 5],
        'keywords': ['весна', 'весенний'],
        'temp_range': (5, 20),
    },
    'summer': {
        'months': [6, 7, 8],
        'keywords': ['лето', 'летний', 'пляж', 'море'],
        'temp_range': (20, 35),
    },
    'fall': {
        'months': [9, 10, 11],
        'keywords': ['осень', 'осенний'],
        'temp_range': (5, 20),
    }
}

MONTH_MAPPING = {
    'январ': 1, 'феврал': 2, 'март': 3, 'апрел': 4,
    'май': 5, 'июн': 6, 'июл': 7, 'август': 8,
    'сентябр': 9, 'октябр': 10, 'ноябр': 11, 'декабр': 12
}

def get_season_from_month(month: int) -> Optional[str]:
    """Get season name from month number"""
    for season, data in SEASONS.items():
        if month in data['months']:
            return season
    return None

def get_season_from_keywords(text: str) -> Optional[str]:
    """Get season from seasonal keywords in text"""
    text = text.lower()
    for season, data in SEASONS.items():
        if any(keyword in text for keyword in data['keywords']):
            return season
    return None

def get_season_from_text(text: str) -> Optional[str]:
    """Try to determine season from text using month names or keywords"""
    # First try to find month
    text = text.lower()
    for month_key, month_num in MONTH_MAPPING.items():
        if month_key in text:
            season = get_season_from_month(month_num)
            if season:
                return season
    
    # If no month found, try keywords
    return get_season_from_keywords(text)
