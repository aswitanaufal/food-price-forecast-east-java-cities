import json
from datetime import datetime, timedelta
from pathlib import Path

import requests


# ex: Kota Surabaya coordinates
LATITUDE = -7.2575
LONGITUDE = 112.7521

RAW_DIR = Path("project/data/raw/open_meteo")
RAW_DIR.mkdir(parents=True, exist_ok=True)


url = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "hourly": (
        "temperature_2m,"
        "relative_humidity_2m,"
        "precipitation,"
        "wind_speed_10m,"
        "weather_code"
    ),
    "timezone": "Asia/Jakarta",
    "forecast_days": 2,
}


print("Fetching Open-Meteo weather data...")

response = requests.get(
    url,
    params=params,
    timeout=30,
)

print(f"HTTP status: {response.status_code}")

response.raise_for_status()

data = response.json()

print("Open-Meteo data received successfully.")
print("Location:", data.get("latitude"), data.get("longitude"))
print("Timezone:", data.get("timezone"))

hourly = data.get("hourly", {})

print("\nAvailable hourly variables:")
print(list(hourly.keys()))

print("\nNumber of timestamps:")
print(len(hourly.get("time", [])))


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_file = RAW_DIR / f"weather_raw_{timestamp}.json"

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(
        data,
        file,
        ensure_ascii=False,
        indent=2,
    )

print(f"\nRaw weather data saved to: {output_file}")