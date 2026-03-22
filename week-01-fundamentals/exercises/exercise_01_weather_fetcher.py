"""
Exercise 1: Weather Fetcher
=============================
Difficulty: Beginner | Time: 1.5 hours

Task:
Create a function that takes a city name, calls the wttr.in API,
and returns a dictionary with temperature and conditions.
Handle errors gracefully.

Instructions:
1. Complete the fetch_weather() function below
2. Handle edge cases: empty city, API timeout, invalid city
3. Test with at least 3 different cities
4. Bonus: Add wind speed and humidity to the output

Run: python exercise_01_weather_fetcher.py
"""

import json

import requests


def fetch_weather(city: str) -> dict:
    """Fetch weather data for a given city.

    Args:
        city: Name of the city

    Returns:
        Dictionary with keys: city, temperature_c, conditions
        (Bonus: wind_kmh, humidity_percent when available)

    Raises:
        ValueError: If city is empty or API call fails
    """
    if city is None or not str(city).strip():
        raise ValueError("City name cannot be empty")

    city_clean = str(city).strip()
    url = f"https://wttr.in/{requests.utils.quote(city_clean)}"

    try:
        response = requests.get(
            url,
            params={"format": "j1"},
            timeout=15,
            headers={"User-Agent": "curl/7.68.0"},
        )
    except requests.Timeout:
        raise ValueError("Weather request timed out; try again later.") from None
    except requests.RequestException as e:
        raise ValueError(f"Could not reach weather service: {e}") from e

    if response.status_code != 200:
        raise ValueError(
            f"Could not get weather for '{city_clean}' "
            "(invalid location or service error)."
        )

    try:
        data = response.json()
    except json.JSONDecodeError:
        raise ValueError(f"Invalid response for '{city_clean}'.") from None

    current = data.get("current_condition") or []
    if not current:
        raise ValueError(f"No weather data available for '{city_clean}'.")

    cc = current[0]
    temp_raw = cc.get("temp_C", "")
    try:
        temperature_c = int(temp_raw) if str(temp_raw).strip().lstrip("-").isdigit() else float(temp_raw)
    except (TypeError, ValueError):
        temperature_c = temp_raw

    desc_items = cc.get("weatherDesc") or []
    conditions = desc_items[0]["value"] if desc_items else "Unknown"

    out: dict = {
        "city": city_clean,
        "temperature_c": temperature_c,
        "conditions": conditions,
    }

    # Bonus: wind speed and humidity
    if "windspeedKmph" in cc and cc["windspeedKmph"] != "":
        try:
            out["wind_kmh"] = int(cc["windspeedKmph"])
        except (TypeError, ValueError):
            out["wind_kmh"] = cc["windspeedKmph"]
    if "humidity" in cc and cc["humidity"] != "":
        try:
            out["humidity_percent"] = int(cc["humidity"])
        except (TypeError, ValueError):
            out["humidity_percent"] = cc["humidity"]

    return out


# === Test your implementation ===
if __name__ == "__main__":
    # Test 1: Valid city
    print("Test 1: London")
    result = fetch_weather("London")
    print(result)

    # Test 2: Another valid city
    print("\nTest 2: Tokyo")
    result = fetch_weather("Tokyo")
    print(result)

    # Test 3: Third city
    print("\nTest 3: New York")
    result = fetch_weather("New York")
    print(result)

    # Test 4: Error handling - empty city
    print("\nTest 4: Empty city (should raise ValueError)")
    try:
        fetch_weather("")
    except ValueError as e:
        print(f"Caught error: {e}")

    # Test 5: Invalid / unknown location (wttr often returns HTTP 500)
    print("\nTest 5: Invalid city (should raise ValueError)")
    try:
        r = fetch_weather("InvalidCityXyz123NotReal")
        print(f"No error — API returned data (may be a guessed location): {r}")
    except ValueError as e:
        print(f"Caught error: {e}")
