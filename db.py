import psycopg2
from psycopg2.extras import Json
import wikipediaapi
import overpy
import requests
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from geoalchemy2 import Geometry
import osrm
from datetime import datetime
import time
from typing import Dict, List, Tuple
import logging

class TourismDatabase:
    def __init__(self):
        self.engine = create_engine('postgresql://user:password@localhost:5432/tourism_db')
        self.wiki = wikipediaapi.Wikipedia('ru')
        self.osm_api = overpy.Overpass()
        self.geolocator = Nominatim(user_agent="tourism_app")
        self.osrm_client = osrm.Client(host='http://router.project-osrm.org')

    def init_database(self):
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE EXTENSION IF NOT EXISTS postgis;

                CREATE TABLE IF NOT EXISTS attractions (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255),
                    description TEXT,
                    wiki_url VARCHAR(255),
                    location GEOMETRY(Point, 4326),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source VARCHAR(50),
                    metadata JSONB
                );

                CREATE TABLE IF NOT EXISTS hotels (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255),
                    description TEXT,
                    location GEOMETRY(Point, 4326),
                    stars INTEGER,
                    amenities JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source VARCHAR(50),
                    metadata JSONB
                );

                CREATE TABLE IF NOT EXISTS distance_cache (
                    id SERIAL PRIMARY KEY,
                    from_point GEOMETRY(Point, 4326),
                    to_point GEOMETRY(Point, 4326),
                    distance_meters INTEGER,
                    duration_seconds INTEGER,
                    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))

    def fetch_wiki_attractions(self, city: str):
        page = self.wiki.page(city)
        if not page.exists():
            return []

        attractions = []
        for link in page.links.values():
            if any(keyword in link.title

                   .lower() for keyword in ['музей', 'памятник', 'собор', 'парк']):
                attraction_page = self.wiki.page(link.title)
                if attraction_page.exists():
                    try:
                        location = self.geolocator.geocode(f"{link.title}, {city}")
                        if location:
                            attractions.append({
                                'name': link.title,
                                'description': attraction_page.summary,
                                'wiki_url': attraction_page.fullurl,
                                'lat': location.latitude,
                                'lon': location.longitude,
                                'source': 'wikipedia'
                            })
                    except Exception as e:
                        logging.error(f"Error geocoding {link.title}: {e}")
                        continue
        return attractions

    def fetch_osm_hotels(self, city: str):
        location = self.geolocator.geocode(city)
        if not location:
            return []

        query = f"""
        [out:json];
        area[name="{city}"]->.searchArea;
        (
          way["tourism"="hotel"](area.searchArea);
          node["tourism"="hotel"](area.searchArea);
        );
        out body;
        >;
        out skel qt;
        """

        result = self.osm_api.query(query)
        hotels = []

        for way in result.ways:
            try:
                name = way.tags.get('name', 'Unknown')
                stars = way.tags.get('stars', None)
                amenities = {k: v for k, v in way.tags.items() if 'amenity' in k}

                center_lat = sum(node.lat for node in way.nodes) / len(way.nodes)
                center_lon = sum(node.lon for node in way.nodes) / len(way.nodes)

                hotels.append({
                    'name': name,
                    'description': way.tags.get('description', ''),
                    'lat': center_lat,
                    'lon': center_lon,
                    'stars': stars,
                    'amenities': amenities,
                    'source': 'osm'
                })
            except Exception as e:
                logging.error(f"Error processing hotel {name}: {e}")
                continue

        return hotels

    def calculate_route(self, from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> Dict:
        try:
            response = self.osrm_client.route(
                coordinates=[[from_lon, from_lat], [to_lon, to_lat]],
                overview=False,
                alternatives=False,
                steps=False
            )

            return {
                'distance': response['routes'][0]['distance'],
                'duration': response['routes'][0]['duration']
            }
        except Exception as e:
            logging.error(f"Error calculating route: {e}")
            return None

    def save_attractions(self, attractions: List[Dict]):
        with self.engine.connect() as conn:
            for attraction in attractions:
                conn.execute(text("""
                    INSERT INTO attractions (name, description, wiki_url, location, source, metadata)
                    VALUES (:name, :description, :wiki_url, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), :source, :metadata)
                """), {
                    'name': attraction['name'],
                    'description': attraction['description'],

                             
                    'wiki_url': attraction['wiki_url'],
                    'lat': attraction['lat'],
                    'lon': attraction['lon'],
                    'source': attraction['source'],
                    'metadata': Json({})
                })

    def save_hotels(self, hotels: List[Dict]):
        with self.engine.connect() as conn:
            for hotel in hotels:
                conn.execute(text("""
                    INSERT INTO hotels (name, description, location, stars, amenities, source, metadata)
                    VALUES (:name, :description, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), :stars, :amenities, :source, :metadata)
                """), {
                    'name': hotel['name'],
                    'description': hotel['description'],
                    'lat': hotel['lat'],
                    'lon': hotel['lon'],
                    'stars': hotel['stars'],
                    'amenities': Json(hotel['amenities']),
                    'source': hotel['source'],
                    'metadata': Json({})
                })

    def get_cached_distance(self, from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> Dict:
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT distance_meters, duration_seconds
                FROM distance_cache
                WHERE ST_DWithin(from_point, ST_SetSRID(ST_MakePoint(:from_lon, :from_lat), 4326), 0.001)
                AND ST_DWithin(to_point, ST_SetSRID(ST_MakePoint(:to_lon, :to_lat), 4326), 0.001)
                AND calculated_at > NOW() - INTERVAL '7 days'
            """), {
                'from_lat': from_lat,
                'from_lon': from_lon,
                'to_lat': to_lat,
                'to_lon': to_lon
            }).fetchone()

            if result:
                return {
                    'distance': result[0],
                    'duration': result[1]
                }
            return None

    def cache_distance(self, from_lat: float, from_lon: float, to_lat: float, to_lon: float, distance: int, duration: int):
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO distance_cache (from_point, to_point, distance_meters, duration_seconds)
                VALUES (
                    ST_SetSRID(ST_MakePoint(:from_lon, :from_lat), 4326),
                    ST_SetSRID(ST_MakePoint(:to_lon, :to_lat), 4326),
                    :distance,
                    :duration
                )
            """), {
                'from_lat': from_lat,
                'from_lon': from_lon,
                'to_lat': to_lat,
                'to_lon': to_lon,
                'distance': distance,
                'duration': duration
            })

if __name__ == "__main__":
    db = TourismDatabase()
    db.init_database()

    cities = ['Москва', 'Санкт-Петербург', 'Казань', 'Сочи']

    for city in cities:
        attractions = db.fetch_wiki_attractions(city)
        db.save_attractions(attractions)

        hotels = db.fetch_osm_hotels(city)
        db.save_hotels(hotels)

        print(f"Processed {len(attractions)} attractions and {len(hotels)} hotels in {city}")

