import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import requests


RAW_DIR = Path("project/data/raw")

PIHPS_RAW_DIR = RAW_DIR / "pihps"
OPEN_METEO_RAW_DIR = RAW_DIR / "open_meteo"

PIHPS_RAW_DIR.mkdir(parents=True, exist_ok=True)
OPEN_METEO_RAW_DIR.mkdir(parents=True, exist_ok=True)


PIHPS_URL = (
    "https://www.bi.go.id/hargapangan/"
    "WebSite/TabelHarga/GetGridDataKomoditas"
)

# id jawa timur
PIHPS_PROVINCE_ID = "16"

# id pasar tradisional
PIHPS_PRICE_TYPE_ID = 1

# laporan harian
PIHPS_REPORT_TYPE = 1

# komoditas
PIHPS_COMMODITIES = {
    "bawang_merah": "cat_5",
    "cabai_merah": "cat_7",
    "cabai_rawit": "cat_8",
}


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# latitude and longitude kota
CITIES = {
    "Kota Surabaya": {
        "latitude": -7.2575,
        "longitude": 112.7521,
    },
    "Kota Malang": {
        "latitude": -7.9839,
        "longitude": 112.6214,
    },
    "Kota Kediri": {
        "latitude": -7.8167,
        "longitude": 112.0167,
    },
    "Kota Blitar": {
        "latitude": -8.0955,
        "longitude": 112.1609,
    },
    "Kota Probolinggo": {
        "latitude": -7.7543,
        "longitude": 113.2159,
    },
    "Kota Madiun": {
        "latitude": -7.6298,
        "longitude": 111.5239,
    },
}


# informasi timestamp (Y-M-D | H-M-S)
def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# format json
def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=4
        )

# request and retry
def request_with_retry(url, params, headers=None, source_name="API", max_retries=3, timeout=60,):

    for attempt in range(1, max_retries + 1):

        try:
            print(f"[INFO] Request {source_name} " f"(attempt: {attempt}/{max_retries})")

            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
            )

            response.raise_for_status()

            return response

        except requests.exceptions.Timeout:

            print(f"[TIMEOUT] {source_name} " f"on {attempt} attempt")

            if attempt < max_retries:

                wait_time = 5 * attempt

                print(
                    f"[INFO] Waiting {wait_time} sec "
                    f"for retry..."
                )

                time.sleep(wait_time)

        except requests.exceptions.RequestException as error:

            print(f"[FAILED] Request {source_name}: "f"{error}")
            break

    print(f"[FAILED] {source_name} failed after " f"{max_retries} attemps")
    return None


# ingest pihps data
def fetch_pihps_data():

    print("\n=== INGESTING PIHPS DATA ===")

    today = datetime.now().date()
    yesterday = today - timedelta(days=1)

    start_date = yesterday.strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    timestamp = get_timestamp()

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://www.bi.go.id/hargapangan",
    }

    success = []
    failed = []

    for commodity_name, commodity_id in PIHPS_COMMODITIES.items():

        params = {
            "price_type_id": PIHPS_PRICE_TYPE_ID,
            "comcat_id": commodity_id,
            "province_id": PIHPS_PROVINCE_ID,
            "regency_id": "",
            "showKota": "true",
            "showPasar": "false",
            "tipe_laporan": PIHPS_REPORT_TYPE,
            "start_date": start_date,
            "end_date": end_date,
        }

        response = request_with_retry(
            url=PIHPS_URL,
            params=params,
            headers=headers,
            source_name=f"PIHPS {commodity_name}",
        )

        if response is None:
            failed.append(commodity_name)
            continue

        try:
            data = response.json()

        except ValueError as error:

            print(f"[FAILED] Response PIHPS " f"{commodity_name} invalid JSON file: {error}")
            failed.append(commodity_name)
            continue

        output_filename = (
            f"pihps_{commodity_name}_{timestamp}.json"
        )

        output_path = PIHPS_RAW_DIR / output_filename

        save_json(data, output_path)

        print(f"[SUCCESS] PIHPS {commodity_name} " f"saved to {output_path}")
        success.append(commodity_name)

        time.sleep(1)

    print("\n=== PIHPS SUMMARY ===")
    print(f"Success: {len(success)}")
    print(f"Failed : {len(failed)}")

    if failed:
        print("Failed commodities:")
        for commodity in failed:
            print(f"- {commodity}")


# ingest open-meteo data
def fetch_open_meteo_data():

    print("\n=== INGESTING OPEN-METEO DATA ===")

    timestamp = get_timestamp()

    success = []
    failed = []

    for city_name, coordinate in CITIES.items():

        latitude = coordinate["latitude"]
        longitude = coordinate["longitude"]

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "precipitation,"
                "wind_speed_10m,"
                "weather_code"
            ),
            "forecast_days": 2,
            "timezone": "Asia/Jakarta",
        }

        response = request_with_retry(
            url=OPEN_METEO_URL,
            params=params,
            source_name=f"Open-Meteo {city_name}",
        )

        if response is None:

            failed.append(city_name)

            continue

        try:
            data = response.json()

        except ValueError as error:

            print(f"[FAILED] Response Open-Meteo " f"{city_name} invalid JSON file: {error}")
            failed.append(city_name)
            continue

        data["project_metadata"] = {
            "city": city_name,
            "latitude": latitude,
            "longitude": longitude,
            "source": "Open-Meteo Forecast API",
            "ingestion_timestamp": datetime.now().isoformat(),
        }

        safe_city_name = (
            city_name.lower()
            .replace(" ", "_")
            .replace(".", "")
        )

        output_filename = (
            f"open_meteo_{safe_city_name}_{timestamp}.json"
        )

        output_path = (
            OPEN_METEO_RAW_DIR / output_filename
        )

        save_json(data, output_path)

        print(f"[SUCCESS] Open-Meteo {city_name} " f"saved to {output_path}")
        success.append(city_name)

        time.sleep(1)

    print("\n=== OPEN-METEO SUMMARY ===")
    print(f"Success: {len(success)}")
    print(f"Failed : {len(failed)}")

    if failed:
        print("Failed cities:")
        for city in failed:
            print(f"- {city}")


def main():

    print("==============================================")
    print("AUTOMATIC DATA INGESTION")
    print("PIHPS and Open-Meteo")
    print("==============================================")

    fetch_pihps_data()
    fetch_open_meteo_data()

    print("\n=== INGESTION COMPLETE ===")


if __name__ == "__main__":
    main()