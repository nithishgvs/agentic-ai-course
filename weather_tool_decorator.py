from langchain_core.tools import tool


@tool
def get_weather(city: str, units: str = "celcius") -> str:
    """Get the current weather for a city.

        Args:
            city: The city name to look up weather for.
            units: Temperature units, either 'celsius' or 'fahrenheit'.
        """
    # Simulated weather data
    weather_data = {
        "Tokyo": {"temp_c": 18, "condition": "Rainy"},
        "London": {"temp_c": 12, "condition": "Cloudy"},
        "New York": {"temp_c": 25, "condition": "Sunny"},
        "Paris": {"temp_c": 20, "condition": "Partly Cloudy"},
    }

    data = weather_data.get(city)
    if data is None:
        return f"Weather data is not available for '{city}'."

    temp = data["temp_c"]

    if units == "fahrenheit":
        temp = temp * 9 / 5 + 32
        unit_label = "F"
    else:
        unit_label = "C"

    return f"{city}: {temp}{unit_label}, {data['condition']}"


print(get_weather.name)
print(get_weather.description)

# Invoke it
print(get_weather.invoke({"city": "Tokyo"}))
