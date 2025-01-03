import re
from typing import Optional, Tuple

def extract_and_normalize_temperature(text: str) -> Optional[Tuple[float, float]]:
    """
    Extract and normalize temperature values from text.
    Returns (min_temp, max_temp) if found, None otherwise.
    """
    text = text.lower()
    
    def normalize_temp_value(temp_str: str) -> Optional[float]:
        """Normalize temperature value handling various formats"""
        temp_str = temp_str.replace(',', '.').strip()
        try:
            temp = float(temp_str)
            # Handle common data entry errors
            if temp > 100:  # Likely missing decimal point
                temp = temp / 10
            if temp > 50:  # Still too high after division
                temp = temp / 10
            if -60 <= temp <= 50:
                return int(temp)  # Truncate decimal part
            return None
        except ValueError:
            return None
    
    # Common patterns for temperature ranges
    patterns = [
        # Range pattern: "от -5 до +2°C"
        r'от\s*(-?\d+[.,]?\d*)\s*до\s*(-?\d+[.,]?\d*)\s*(?:°|градус|c°|°c)',
        # Single value with month: "в январе -5°C"
        r'(?:январ|феврал|март|апрел|май|июн|июл|август|сентябр|октябр|ноябр|декабр)[а-я]*\s*[-—]?\s*(?:плюс\s*)?(-?\d+[.,]?\d*)\s*(?:°|градус|c°|°c)',
        # Simple range: "-5...+2°C"
        r'(-?\d+[.,]?\d*)\.\.\.(-?\d+[.,]?\d*)\s*(?:°|градус|c°|°c)',
        # Average temperature: "средняя температура +15°C"
        r'средн[а-я]*\s*температур[а-я]*\s*[-—]?\s*(?:плюс\s*)?(-?\d+[.,]?\d*)\s*(?:°|градус|c°|°c)',
        # Temperature with plus/minus: "температура +34,7°C"
        r'температур[а-я]*\s*[-—]?\s*(?:плюс\s*)?(-?\d+[.,]?\d*)\s*(?:°|градус|c°|°c)',
        # Month temperature: "июля — +25,5°C"
        r'(?:январ|феврал|март|апрел|май|июн|июл|август|сентябр|октябр|ноябр|декабр)[а-я]*\s*[-—]\s*(?:плюс\s*)?(-?\d+[.,]?\d*)\s*(?:°|градус|c°|°c)',
        # Temperature with explicit plus: "плюс 25 градусов"
        r'плюс\s*(-?\d+[.,]?\d*)\s*(?:°|градус|c°|°c)',
        # Temperature with minus: "минус 5 градусов"
        r'минус\s*(\d+[.,]?\d*)\s*(?:°|градус|c°|°c)'
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        temps = []
        for match in matches:
            groups = match.groups()
            for temp_str in groups:
                if temp_str:
                    # Normalize temperature value
                    if temp_str:
                        if temp_str.startswith('минус'):
                            temp_str = '-' + temp_str[5:]
                        temp = normalize_temp_value(temp_str)
                        if temp is not None:
                            temps.append(temp)
        
        if temps:
            return min(temps), max(temps)
    
    return None

def normalize_temperature_text(text: str) -> str:
    """
    Normalize temperature values in text by removing obviously wrong values
    and standardizing format.
    """
    def normalize_temp(match):
        temp_str = match.group(1)
        if temp_str.startswith('минус'):
            temp_str = '-' + temp_str[5:]
        temp = normalize_temp_value(temp_str)
        if temp is not None:
            return f"{int(temp)}°C"  # Truncate decimal part
        return "N/A°C"
    
    # Replace temperature values
    text = re.sub(r'(-?\d+)\s*(?:°|градус|c°|°c)', normalize_temp, text)
    
    return text

def is_temperature_in_range(text: str, min_required: float, max_required: float) -> bool:
    """
    Check if text contains temperature values within the required range.
    """
    temps = extract_and_normalize_temperature(text)
    if not temps:
        return False
        
    min_temp, max_temp = temps
    return min_required <= max_temp <= max_required
