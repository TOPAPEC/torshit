import json
from typing import Dict, List, Set
import re

def analyze_city_data(city_data: str) -> Dict[str, List[str]]:
    """Analyze city data for key tourism features"""
    analysis = {
        'winter_sports': [],
        'beach': [],
        'cultural': [],
        'family': [],
        'spa': []
    }
    
    # Winter sports features
    winter_patterns = [
        r'горнолыжн\w*', r'лыж\w*', r'трасс\w*', r'подъемник\w*',
        r'красная поляна', r'роза хутор', r'сноуборд\w*'
    ]
    for pattern in winter_patterns:
        matches = re.finditer(pattern, city_data.lower())
        for match in matches:
            analysis['winter_sports'].append(f"{match.group(0)} (context: {city_data[max(0, match.start()-30):match.end()+30]})")
    
    # Beach features
    beach_patterns = [
        r'пляж\w*', r'море\w*', r'купальн\w*', r'аквапарк\w*'
    ]
    for pattern in beach_patterns:
        matches = re.finditer(pattern, city_data.lower())
        for match in matches:
            analysis['beach'].append(f"{match.group(0)} (context: {city_data[max(0, match.start()-30):match.end()+30]})")
    
    # Cultural features
    cultural_patterns = [
        r'музей\w*', r'театр\w*', r'памятник\w*', r'собор\w*',
        r'достопримечательност\w*'
    ]
    for pattern in cultural_patterns:
        matches = re.finditer(pattern, city_data.lower())
        for match in matches:
            analysis['cultural'].append(f"{match.group(0)} (context: {city_data[max(0, match.start()-30):match.end()+30]})")
    
    return analysis

def load_poi_cache() -> Dict[str, str]:
    """Load POI cache data"""
    with open('poi_cache.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_all_cities():
    """Analyze all cities in the POI cache"""
    poi_data = load_poi_cache()
    
    print("=== City Analysis Report ===\n")
    for city, data in poi_data.items():
        print(f"\n=== {city} ===")
        analysis = analyze_city_data(data)
        
        # Report missing key features
        if not analysis['winter_sports'] and 'сочи' in city.lower():
            print("❌ Missing winter sports info (should have Krasnaya Polyana, Rosa Khutor)")
        if not analysis['beach'] and any(x in city.lower() for x in ['сочи', 'анапа']):
            print("❌ Missing beach info")
            
        # Show found features
        for category, features in analysis.items():
            if features:
                print(f"\n{category.upper()} features found:")
                for feature in features:
                    print(f"  • {feature}")

if __name__ == "__main__":
    analyze_all_cities()
