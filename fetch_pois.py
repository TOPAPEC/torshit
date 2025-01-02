import asyncio
import aiohttp
import json
from urllib.parse import quote
from typing import Optional
from osm_service import OSMService, POIData, CityPOIs
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fetch_and_cache_pois():
    print("Starting POI data collection...")
    osm_service = OSMService(use_cache=False)  # Don't use cache while building it
    
    # Get all cities from config
    all_cities = []
    for city_list in Config.RESORT_CITIES.values():
        all_cities.extend(city_list)
    all_cities = list(set(all_cities))  # Remove duplicates
    
    print(f"Fetching POI data for {len(all_cities)} cities...")
    
    # Create a dictionary to store POI data
    poi_cache = {}
    
    async def get_area_id(session: aiohttp.ClientSession, city: str) -> Optional[int]:
        """Get area ID for a city using Nominatim"""
        encoded_city = quote(city)
        nominatim_url = f"https://nominatim.openstreetmap.org/search?q={encoded_city}&format=json&limit=1"
        try:
            async with session.get(
                nominatim_url,
                headers={"User-Agent": "TorshitApp/1.0"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data:
                        # Get OSM ID and convert to area ID
                        osm_id = int(data[0]['osm_id'])
                        # If it's a relation, convert to area ID format
                        if data[0]['osm_type'] == 'relation':
                            return 3600000000 + osm_id
                        elif data[0]['osm_type'] == 'way':
                            return 2400000000 + osm_id
                        else:
                            return None
                return None
        except Exception as e:
            logger.error(f"Error getting area ID for {city}: {str(e)}")
            return None

    async def process_response(response_data: dict) -> list:
        """Process JSON response from Overpass API"""
        pois = []
        for element in response_data.get('elements', []):
            tags = element.get('tags', {})
            if 'name' in tags:
                poi_type = next((k for k, v in tags.items() if k in ["tourism", "leisure", "sport", "natural", "amenity"]), "unknown")
                pois.append({
                    "name": tags.get('name', 'Unknown'),
                    "type": poi_type,
                    "description": tags.get('description')
                })
        return pois
    
    # Create a single session for all requests
    async with aiohttp.ClientSession(headers={"User-Agent": "TorshitApp/1.0"}) as session:
        for city in all_cities:
            print(f"\nFetching data for {city}...")
            try:
                # Build query for all categories at once to reduce API calls
                # First get the area ID
                area_id = await get_area_id(session, city)
                if not area_id:
                    logger.error(f"Could not find area ID for {city}")
                    raise Exception("Area ID not found")

                # Use area ID in query
                query = f"""
                [out:json][timeout:25];
                area({area_id})->.searchArea;
                (
                    node["tourism"~"museum|attraction|viewpoint"](area.searchArea);
                    node["historic"](area.searchArea);
                    node["natural"="beach"](area.searchArea);
                    node["leisure"~"beach_resort|water_park|park|playground|sports_centre"](area.searchArea);
                    node["amenity"~"theatre|cinema"](area.searchArea);
                    node["sport"~"skiing|swimming"](area.searchArea);
                );
                out body;
                """
                
                # Add delay before API call to avoid rate limiting
                await asyncio.sleep(1)
                
                async with session.post(
                    osm_service.overpass_url,
                    data={"data": query},
                    timeout=30
                ) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        pois = await process_response(response_data)
                        logger.info(f"Found {len(pois)} POIs for {city}")
                        
                        # Categorize POIs
                        tourist_attractions = []
                        beaches = []
                        entertainment = []
                        sports_facilities = []
                        
                        for poi in pois:
                            poi_data = POIData(
                                name=poi['name'],
                                type=poi['type'],
                                category='unknown',
                                description=poi.get('description')
                            )
                            
                            if poi['type'] in ['tourism', 'historic']:
                                tourist_attractions.append(poi_data)
                            elif poi['type'] == 'natural' or 'beach' in poi['type']:
                                beaches.append(poi_data)
                            elif poi['type'] in ['leisure', 'amenity'] and poi['type'] not in ['sports_centre']:
                                entertainment.append(poi_data)
                            elif poi['type'] in ['sport', 'sports_centre']:
                                sports_facilities.append(poi_data)
                        
                        poi_cache[city] = {
                            "tourist_attractions": [
                                {"name": poi.name, "type": poi.type, "category": poi.category, "description": poi.description}
                                for poi in tourist_attractions
                            ],
                            "beaches": [
                                {"name": poi.name, "type": poi.type, "category": poi.category, "description": poi.description}
                                for poi in beaches
                            ],
                            "entertainment": [
                                {"name": poi.name, "type": poi.type, "category": poi.category, "description": poi.description}
                                for poi in entertainment
                            ],
                            "sports_facilities": [
                                {"name": poi.name, "type": poi.type, "category": poi.category, "description": poi.description}
                                for poi in sports_facilities
                            ]
                        }
                        logger.info(f"Successfully fetched and categorized data for {city}")
                        if not any([tourist_attractions, beaches, entertainment, sports_facilities]):
                            logger.warning(f"No POIs found for {city} - this might indicate an issue with the area query")
                        
                        # Save after each successful city fetch
                        with open('poi_cache.json', 'w', encoding='utf-8') as f:
                            json.dump(poi_cache, f, ensure_ascii=False, indent=2)
                    else:
                        logger.error(f"Error fetching data for {city}: HTTP {response.status}")
                        response_text = await response.text()
                        logger.error(f"Response: {response_text[:200]}...")  # Log first 200 chars of error response
                        poi_cache[city] = {
                            "tourist_attractions": [],
                            "beaches": [],
                            "entertainment": [],
                            "sports_facilities": []
                        }
            
            except Exception as e:
                logger.error(f"Exception while fetching data for {city}: {str(e)}", exc_info=True)
                poi_cache[city] = {
                    "tourist_attractions": [],
                    "beaches": [],
                    "entertainment": [],
                    "sports_facilities": []
                }
                continue
    
    print("\nPOI data collection completed!")
    print(f"Data saved to poi_cache.json")

if __name__ == "__main__":
    asyncio.run(fetch_and_cache_pois())
