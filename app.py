from flask import Flask, render_template, request, jsonify
import asyncio
from advisor import TravelAdvisor
from asgiref.sync import async_to_sync

app = Flask(__name__)
# Initialize advisor
advisor = TravelAdvisor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
async def ask():
    try:
        query = request.json.get('message')
        if not query:
            return jsonify({'error': 'No message provided'}), 400

        # Process the query using TravelAdvisor
        cities_chunks, top_cities, preferences, available_tokens, relevant_facts = await advisor.process_request(query)
        
        if not cities_chunks:
            return jsonify({
                'preferences': 'Не удалось обработать запрос',
                'recommendations': []
            })

        # Format preferences for display
        preferences_html = f"<h3>📋 Анализ запроса:</h3><p>{preferences}</p>"

        # Format recommendations
        recommendations = []
        for city, score in top_cities:
            details = []
            
            # Add score
            details.append(f"Релевантность: {score:.3f}")
            
            # Add relevant facts
            if city in relevant_facts and relevant_facts[city]:
                facts_text = "📚 Интересные факты:"
                for fact, _ in relevant_facts[city]:
                    facts_text += f"<br>• {fact}"
                details.append(facts_text)
            
            # Add climate information if available
            if city in cities_chunks:
                all_text = " ".join(cities_chunks[city])
                if 'температура' in all_text.lower() or any(month in all_text.lower() for month in ['январ', 'феврал', 'март', 'апрел', 'май', 'июн', 'июл', 'август', 'сентябр', 'октябр', 'ноябр', 'декабр']):
                    climate_info = "🌡️ "
                    for sentence in all_text.split('.'):
                        if any(month in sentence.lower() for month in ['январ', 'феврал', 'март', 'апрел', 'май', 'июн', 'июл', 'август', 'сентябр', 'октябр', 'ноябр', 'декабр']) and ('температура' in sentence.lower() or 'градус' in sentence.lower()):
                            climate_info += sentence.strip() + ". "
                            break
                    if climate_info != "🌡️ ":
                        details.append(climate_info.strip().replace(",", "."))

            recommendations.append({
                'name': city,
                'details': details
            })

        return jsonify({
            'preferences': preferences_html,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({
            'preferences': f'Произошла ошибка: {str(e)}',
            'recommendations': []
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
