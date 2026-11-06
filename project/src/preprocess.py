import json
import re
from datetime import datetime
from pathlib import Path
import pandas as pd


RAW_DIR = Path("project/data/raw")

PIHPS_RAW_DIR = RAW_DIR / "pihps"
OPEN_METEO_RAW_DIR = RAW_DIR / "open_meteo"

PROCESSED_DIR = Path("project/data/processed")

PIHPS_RAW_DIR.mkdir(parents=True, exist_ok=True)
OPEN_METEO_RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


TARGET_CITIES = [
    "Kota Surabaya",
    "Kota Malang",
    "Kota Kediri",
    "Kota Blitar",
    "Kota Probolinggo",
    "Kota Madiun",
    ]

# imputasi & normalisasi harga
def clean_price(value):
   
    if value is None:
        return None

    value = str(value).strip()

    if value == "" or value == "-":
        return None

    cleaned_value = re.sub(r"[^0-9]", "", value)

    if cleaned_value == "":
        return None

    return float(cleaned_value)


def extract_date_columns(columns):
    
    date_columns = []

    for column in columns:
        try:
            datetime.strptime(str(column), "%d/%m/%Y")
            date_columns.append(column)
        except ValueError:
            continue

    return date_columns

# preprocessing pihps data
# conver wide -> long data (dibutuhkan info kolom date + city)
# merge dataset
def preprocess_pihps():

    print("\n=== Preprocessing PIHPS ===")

    all_records = []

    raw_files = sorted(PIHPS_RAW_DIR.glob("*.json"))

    if not raw_files:
        print("[WARNING] PIHPS raw file not found")
        return pd.DataFrame()

    for raw_file in raw_files:

        print(f"Reading: {raw_file.name}")

        with open(raw_file, "r", encoding="utf-8") as file:
            raw_data = json.load(file)

        rows = raw_data.get("data", [])

        if not rows:
            print(f"[WARNING] data not found in {raw_file.name}")
            continue

        df = pd.DataFrame(rows)

        date_columns = extract_date_columns(df.columns)

        if not date_columns:
            print(
                f"[WARNING] Date columns not found"
                f"in {raw_file.name}"
            )
            continue

        # convert wide data -> long data
        long_df = df.melt(
            id_vars=["name", "level"],
            value_vars=date_columns,
            var_name="date",
            value_name="price",
        )

        long_df = long_df.rename(
            columns={
                "name": "city",
            }
        )

        long_df = long_df[
            long_df["level"].astype(str) == "2" #kota/kab
        ]

        long_df = long_df[
            long_df["city"].isin(TARGET_CITIES)
        ]

        long_df["price"] = long_df["price"].apply(clean_price)

        long_df["date"] = pd.to_datetime(
            long_df["date"],
            format="%d/%m/%Y",
            errors="coerce",
        )

        long_df["source"] = "PIHPS Bank Indonesia"

        long_df["ingestion_timestamp"] = datetime.now().isoformat()

        if "bawang_merah" in raw_file.name:
            long_df["commodity"] = "Bawang Merah"

        elif "cabai_merah" in raw_file.name:
            long_df["commodity"] = "Cabai Merah"

        elif "cabai_rawit" in raw_file.name:
            long_df["commodity"] = "Cabai Rawit"

        else:
            long_df["commodity"] = "Unknown"

        all_records.append(long_df)

    if not all_records:
        print("[INFO] No PIHPS data was successfully processed")
        return pd.DataFrame()
    
    # merge data
    result = pd.concat(all_records, ignore_index=True)

    result = result[
        [
            "date",
            "city",
            "commodity",
            "price",
            "source",
            "ingestion_timestamp",
        ]
    ]

    result = result.drop_duplicates()

    output_path = PROCESSED_DIR / "pihps_processed.csv"

    result.to_csv(output_path, index=False)

    print(f"[SUCCESS] PIHPS data saved to {output_path}")

    return result

# preprocessing open meteo data
# convert hourly -> daily
def preprocess_open_meteo():

    print("\n=== Preprocessing Open-Meteo ===")

    all_daily_data = []

    raw_files = sorted(OPEN_METEO_RAW_DIR.glob("*.json"))

    if not raw_files:
        print("[INFO] Open-Meteo raw file not found")
        return pd.DataFrame()

    for raw_file in raw_files:

        print(f"Reading: {raw_file.name}")

        with open(raw_file, "r", encoding="utf-8") as file:
            raw_data = json.load(file)

        hourly_data = raw_data.get("hourly", {})
        metadata = raw_data.get("project_metadata", {})

        if not hourly_data:
            print(
                f"[WARNING] Hourly data not found"
                f"in {raw_file.name}"
            )
            continue

        time_values = hourly_data.get("time", [])

        temperature_values = hourly_data.get(
            "temperature_2m", []
        )

        humidity_values = hourly_data.get(
            "relative_humidity_2m", []
        )

        precipitation_values = hourly_data.get(
            "precipitation", []
        )

        wind_values = hourly_data.get(
            "wind_speed_10m", []
        )

        weather_code_values = hourly_data.get(
            "weather_code", []
        )

        df = pd.DataFrame(
            {
                "datetime": time_values,
                "temperature": temperature_values,
                "humidity": humidity_values,
                "precipitation": precipitation_values,
                "wind_speed": wind_values,
                "weather_code": weather_code_values,
            }
        )

        if df.empty:
            continue

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce",
        )

        df["date"] = df["datetime"].dt.date

        # konversi kolom numerik
        numeric_columns = [
            "temperature",
            "humidity",
            "precipitation",
            "wind_speed",
            "weather_code",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        daily_df = (
            df.groupby("date")
            .agg(
                temperature_mean=("temperature", "mean"),
                temperature_max=("temperature", "max"),
                precipitation_sum=("precipitation", "sum"),
                humidity_mean=("humidity", "mean"),
                wind_speed_mean=("wind_speed", "mean"),
                weather_code=("weather_code", "max"),
            )
            .reset_index()
        )

        daily_df["city"] = metadata.get(
            "city",
            "Unknown",
        )

        daily_df["source"] = "Open-Meteo Forecast API"

        daily_df["ingestion_timestamp"] = metadata.get(
            "ingestion_timestamp",
            datetime.now().isoformat(),
        )

        all_daily_data.append(daily_df)

    if not all_daily_data:
        print("[INFO] No Open-Meteo data was successfully processed.")
        return pd.DataFrame()

    result = pd.concat(all_daily_data, ignore_index=True,)

    result = result[
        [
            "date",
            "city",
            "temperature_mean",
            "temperature_max",
            "precipitation_sum",
            "humidity_mean",
            "wind_speed_mean",
            "weather_code",
            "source",
            "ingestion_timestamp",
        ]
    ]

    result = result.drop_duplicates()

    output_path = PROCESSED_DIR / "open_meteo_processed.csv"

    result.to_csv(output_path, index=False)

    print(
        f"[SUCCESS] Open-Meteo data saved to {output_path}"
    )

    return result


def main():
    print("==============================================")
    print("PREPROCESSING DATA")
    print("PIHPS and Open-Meteo")
    print("==============================================")

    preprocess_pihps()
    preprocess_open_meteo()

    print("\n=== PREPROCESSING COMPLETE ===")


if __name__ == "__main__":
    main()