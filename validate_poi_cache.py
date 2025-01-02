import json
import sys

def validate_poi_cache(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Check if data is a dictionary
        if not isinstance(data, dict):
            print("Error: POI cache is not a valid JSON object")
            return False
            
        # Check for multiple cities
        cities = list(data.keys())
        if len(cities) < 2:
            print(f"Warning: Only found {len(cities)} cities")
            
        # Validate city structure
        required_categories = {
            'tourist_attractions',
            'beaches',
            'entertainment',
            'sports_facilities'
        }
        
        for city, categories in data.items():
            if not isinstance(categories, dict):
                print(f"Error: City {city} has invalid structure")
                return False
            if not required_categories.issubset(categories.keys()):
                print(f"Error: City {city} is missing required categories")
                return False
            for category, pois in categories.items():
                if not isinstance(pois, list):
                    print(f"Error: {city} {category} is not a list")
                    return False
                
        print("POI cache validation successful")
        print(f"Found {len(cities)} cities")
        return True
        
    except json.JSONDecodeError:
        print("Error: POI cache is corrupted or not valid JSON")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_poi_cache.py <path_to_poi_cache.json>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    if not validate_poi_cache(file_path):
        sys.exit(1)
