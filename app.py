from flask import Flask, render_template, request, jsonify
import asyncio
from advisor import TravelAdvisor

app = Flask(__name__)
advisor = TravelAdvisor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('message', '')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Run the async advisor in a sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        cities_chunks, top_cities, preferences, available_tokens = loop.run_until_complete(
            advisor.process_request(user_input)
        )
        loop.close()

        if not cities_chunks:
            return jsonify({'error': 'Could not process request'}), 500

        # Format the response
        response = {
            'preferences': preferences,
            'recommendations': []
        }

        for city, score in top_cities:
            city_info = {
                'name': city,
                'score': f"{score:.3f}",
                'details': []
            }
            
            if city in cities_chunks:
                # Extract relevant information based on user preferences
                all_text = " ".join(cities_chunks[city])
                
                # Create a focused summary based on the city's features
                summary = []
                
                # Check for beach and sea information
                if 'пляж' in all_text.lower() or 'море' in all_text.lower():
                    beach_info = "🏖️ "
                    if 'песчаный' in all_text.lower():
                        beach_info += "Есть песчаные пляжи. "
                    if 'купальный сезон' in all_text.lower():
                        for sentence in all_text.split('.'):
                            if 'купальный сезон' in sentence.lower():
                                beach_info += sentence.strip() + ". "
                                break
                    summary.append(beach_info.strip())
                
                # Check for climate information
                if 'август' in all_text.lower() or 'температура' in all_text.lower():
                    climate_info = "🌡️ "
                    for sentence in all_text.split('.'):
                        if 'август' in sentence.lower() and ('температура' in sentence.lower() or 'градус' in sentence.lower()):
                            climate_info += sentence.strip() + ". "
                            break
                    if not 'август' in climate_info.lower():
                        for sentence in all_text.split('.'):
                            if 'лет' in sentence.lower() and 'температура' in sentence.lower():
                                climate_info += sentence.strip() + ". "
                                break
                    summary.append(climate_info.strip())
                
                # Add tourist infrastructure info if available
                if 'туристи' in all_text.lower():
                    for sentence in all_text.split('.'):
                        if 'туристи' in sentence.lower():
                            summary.append("🏨 " + sentence.strip() + ".")
                            break
                
                if summary:
                    city_info['details'] = summary
                else:
                    city_info['details'] = [f"🎯 {city} подходит под ваши критерии поиска."]
            
            response['recommendations'].append(city_info)

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
