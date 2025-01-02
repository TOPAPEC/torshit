import asyncio
import aiohttp
import json
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class POIData:
    name: str
    type: str
    category: str
    description: Optional[str] = None

@dataclass
class CityPOIs:
    tourist_attractions: List[POIData]
    beaches: List[POIData]
    
    entertainment: List[POIData]
    sports_facilities: List[POIData]

class OSMService:
    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        self.cache = {}
        self.overpass_url = "https://overpass-api.de/api/interpreter"
        self.categories = {
            "tourist_attractions": [
                "tourism=museum",
                "tourism=attraction",
                "historic=*",
                "tourism=viewpoint"
            ],
            "beaches": [
                "natural=beach",
                "leisure=beach_resort"
            ],
            "entertainment": [
                "leisure=water_park",
                "leisure=park",
                "leisure=playground",
                "amenity=theatre",
                "amenity=cinema"
            ],
            "sports_facilities": [
                "leisure=sports_centre",
                "sport=skiing",
                "sport=swimming"
            ]
        }
        if use_cache:
            self._load_cache()

    def _load_cache(self):
        """Load POI data from cache file if it exists"""
        try:
            with open('poi_cache.json', 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                
            # Convert cached data back to POIData objects
            for city, data in cache_data.items():
                self.cache[city] = CityPOIs(
                    tourist_attractions=[
                        POIData(**poi_data)
                        for poi_data in data['tourist_attractions']
                    ],
                    beaches=[
                        POIData(**poi_data)
                        for poi_data in data['beaches']
                    ],
                    entertainment=[
                        POIData(**poi_data)
                        for poi_data in data['entertainment']
                    ],
                    sports_facilities=[
                        POIData(**poi_data)
                        for poi_data in data['sports_facilities']
                    ]
                )
            print(f"Loaded POI cache with data for {len(self.cache)} cities")
        except FileNotFoundError:
            print("Cache file not found. Will fetch data from API.")
        except Exception as e:
            print(f"Error loading cache: {str(e)}")

    def _build_query(self, city: str, category_filters: List[str]) -> str:
        """Build Overpass QL query for POIs in a city"""
        area_query = f"""
        area["name"="{city}"]["place"~"city|town"]["admin_level"~"4|6"]["boundary"="administrative"]->.searchArea;
        (
        """
        
        # Add each category filter
        nodes = []
        for category in category_filters:
            key, value = category.split('=')
            if value == '*':
                nodes.append(f'node["{key}"](area.searchArea);')
            else:
                nodes.append(f'node["{key}"="{value}"](area.searchArea);')
        
        query = area_query + '\n'.join(nodes) + """
        );
        out body;
        """
        return query

    async def _fetch_pois(self, session: aiohttp.ClientSession, city: str, category: str) -> List[POIData]:
        """Fetch POIs for a specific category"""
        query = self._build_query(city, self.categories[category])
        
        try:
            async with session.post(self.overpass_url, data={"data": query}) as response:
                if response.status == 200:
                    data = await response.json()
                    pois = []
                    
                    for element in data.get("elements", []):
                        tags = element.get("tags", {})
                        poi = POIData(
                            name=tags.get("name", "Unknown"),
                            type=next((k for k, v in tags.items() if k in ["tourism", "leisure", "sport", "natural", "amenity"]), "unknown"),
                            category=category,
                            description=tags.get("description", None)
                        )
                        pois.append(poi)
                    
                    return pois
                else:
                    print(f"Error fetching {category} POIs for {city}: {response.status}")
                    return []
        except Exception as e:
            print(f"Exception fetching {category} POIs for {city}: {str(e)}")
            return []

    async def get_city_pois(self, city: str) -> CityPOIs:
        """Get all POIs for a city"""
        if self.use_cache and city in self.cache:
            return self.cache[city]
        
        # Fallback to API if cache is not available or not being used
        async with aiohttp.ClientSession() as session:
            tasks = []
            for category in self.categories.keys():
                tasks.append(self._fetch_pois(session, city, category))
            
            results = await asyncio.gather(*tasks)
            
            return CityPOIs(
                tourist_attractions=results[0],
                beaches=results[1],
                entertainment=results[2],
                sports_facilities=results[3]
            )

    def format_poi_description(self, pois: CityPOIs) -> str:
        """Format POIs into a readable description"""
        sections = []
        
        if pois.tourist_attractions:
            sections.append("🏛️ Достопримечательности:")
            for poi in pois.tourist_attractions[:5]:  # Limit to top 5
                sections.append(f"  • {poi.name}")
        
        if pois.beaches:
            sections.append("🏖️ Пляжи и набережные:")
            for poi in pois.beaches:
                sections.append(f"  • {poi.name}")
        
        if pois.entertainment:
            sections.append("🎡 Развлечения:")
            for poi in pois.entertainment[:5]:
                sections.append(f"  • {poi.name}")
        
        if pois.sports_facilities:
            sections.append("🏃 Спортивные объекты:")
            for poi in pois.sports_facilities[:5]:
                sections.append(f"  • {poi.name}")
        
        return "\n".join(sections)
