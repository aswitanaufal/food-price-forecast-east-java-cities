import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


PIHPS_URL = (
    "https://www.bi.go.id/hargapangan/"
    "WebSite/TabelHarga/GetGridDataKomoditas"
)

RAW_DIR = Path("project/data/raw/pihps")
PROCESSED_DIR = Path("project/data/processed")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Pada endpoint PIHPS yang digunakan project ini, ID 16 = Jawa Timur.
PROVINCE_ID = "16"

COMMODITIES = {
    "bawang_merah": "cat_5",
    "cabai_merah": "cat_7",
    "cabai_rawit": "cat_8",
}

TARGET_CITIES = {
    "Kota Surabaya",
    "Kota Malang",
    "Kota Kediri",
    "Kota Blitar",
    "Kota Probolinggo",
    "Kota Pasuruan",
    "Kota Mojokerto",
    "Kota Madiun",
    "Kota Batu",
}

CITY_FIELDS = (
    "name",
    "kota",
    "wilayah",
    "region_name",
    "nama_kota",
    "nama_wilayah",
    "regency_name",
    "kabupaten_kota",
)


def normalize_text(value):
    return " ".join(str(value).strip().lower().split())


CITY_ALIASES = {}

for city in TARGET_CITIES:
    CITY_ALIASES[normalize_text(city)] = city
    CITY_ALIASES[normalize_text(city.replace("Kota ", ""))] = city


def get_city_name(row):
    for field in CITY_FIELDS:
        value = row.get(field)

        if value:
            return str(value).strip()

    return ""


def get_target_city(row):
    city_name = get_city_name(row)
    return CITY_ALIASES.get(normalize_text(city_name))


end_date = datetime.now()
start_date = end_date - timedelta(days=30)

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": (
        "https://www.bi.go.id/hargapangan/"
        "TabelHarga/PasarTradisionalDaerah"
    ),
}
# delete previous outputs [for testing only]
def clear_previous_outputs():
    """Menghapus hasil fetch dan hasil transform sebelumnya."""
    for directory in (RAW_DIR, PROCESSED_DIR):
        for file_path in directory.glob("*"):
            if file_path.is_file():
                file_path.unlink()

print("Fetching PIHPS data...")
clear_previous_outputs()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print(
    f"Date range: {start_date.strftime('%Y-%m-%d')} - "
    f"{end_date.strftime('%Y-%m-%d')}"
)

all_data = []

for commodity_name, commodity_id in COMMODITIES.items():
    params = {
        "price_type_id": 1,
        "comcat_id": commodity_id,
        "province_id": PROVINCE_ID,
        "regency_id": "",
        "showKota": "true",
        "showPasar": "false",
        "tipe_laporan": 1,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    }

    print(f"\nFetching {commodity_name} ({commodity_id})...")

    response = requests.get(
        PIHPS_URL,
        params=params,
        headers=headers,
        timeout=30,
    )

    print(f"HTTP status: {response.status_code}")
    response.raise_for_status()

    result = response.json()

    raw_file = RAW_DIR / f"{commodity_name}_{timestamp}.json"

    with open(raw_file, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    data = []

    for row in result.get("data", []):
        if str(row.get("level")) != "2":
            continue

        target_city = get_target_city(row)

        if target_city is None:
            continue
 
        row["city"] = target_city
        row["commodity"] = commodity_name
        row["comcat_id"] = commodity_id

        data.append(row)

    all_data.extend(data)

    returned_cities = {row["city"] for row in data}
    missing_cities = TARGET_CITIES - returned_cities

    print(f"Target city rows: {len(data)}")
    print(f"Cities returned: {sorted(returned_cities)}")

    if missing_cities:
        print(f"Cities without data: {sorted(missing_cities)}")

    print(f"Raw data saved to: {raw_file}")

if all_data:
    df = pd.DataFrame(all_data)
    processed_file = PROCESSED_DIR / f"pihps_poc_{timestamp}.csv"

    df.to_csv(
        processed_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nTotal city rows: {len(df)}")
    print(f"Unique cities: {df['city'].nunique()}")
    print(f"Processed data saved to: {processed_file}")
else:
    print("No target city data returned from PIHPS.")