
from pathlib import Path

import numpy as np
import pandas as pd


# Lokasi file sesuai struktur repository
PROCESSED_DIR = Path("project/data/processed")

PIHPS_FILE = PROCESSED_DIR / "pihps_processed.csv"
WEATHER_FILE = PROCESSED_DIR / "open_meteo_processed.csv"

OUTPUT_FILE = PROCESSED_DIR / "food_price_features.csv"


WEATHER_FEATURES = [
    "temperature_mean",
    "temperature_max",
    "precipitation_sum",
    "humidity_mean",
    "wind_speed_mean",
    "weather_code",
]


def load_data():
# baca processed data
    if not PIHPS_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {PIHPS_FILE}"
        )

    if not WEATHER_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {WEATHER_FILE}"
        )

    prices = pd.read_csv(PIHPS_FILE)
    weather = pd.read_csv(WEATHER_FILE)

    prices["date"] = pd.to_datetime(
        prices["date"], errors="coerce"
    )

    weather["date"] = pd.to_datetime(
        weather["date"], errors="coerce"
    )

    prices["city"] = prices["city"].astype(str).str.strip()
    weather["city"] = weather["city"].astype(str).str.strip()

    prices["price"] = pd.to_numeric(
        prices["price"], errors="coerce"
    )

    # hapus data yang tidak memiliki tanggal, kota, atau harga.
    prices = prices.dropna(
        subset=["date", "city", "commodity", "price"]
    )

    weather = weather.dropna(
        subset=["date", "city"]
    )

    # satu harga untuk setiap tanggal, kota, dan komoditas.
    prices = (
        prices.sort_values("date")
        .drop_duplicates(
            subset=["date", "city", "commodity"],
            keep="last",
        )
    )

    weather = (
        weather.sort_values("date")
        .drop_duplicates(
            subset=["date", "city"],
            keep="last",
        )
    )

    return prices, weather


def create_features(prices, weather):
    # feature engineering

    available_weather = [
        col for col in WEATHER_FEATURES
        if col in weather.columns
    ]

    if not available_weather:
        raise ValueError(
            "weather feature not found in "
            "open_meteo_processed.csv."
        )

    # konversi fitur cuaca menjadi numerik
    for col in available_weather:
        weather[col] = pd.to_numeric(
            weather[col], errors="coerce"
        )

    # merge data harga dan cuaca berdasarkan tanggal, kota
    data = pd.merge(
        prices,
        weather[["date", "city"] + available_weather],
        on=["date", "city"],
        how="left",
    )

    # sort data
    data = data.sort_values(
        ["city", "commodity", "date"]
    ).reset_index(drop=True)

    grouped = data.groupby(
        ["city", "commodity"],
        sort=False,
    )["price"]

    # harga pada observasi 1 hari sebelumnya
    data["price_lag_1"] = grouped.shift(1)

    # harga 7 hari sebelumnya
    data["price_lag_7"] = grouped.shift(7)

    # rata-rata 7 hari sebelumnya
    data["rolling_mean_7"] = (
        data.groupby(["city", "commodity"])["price"]
        .transform(
            lambda series: series.shift(1).rolling(
                window=7,
                min_periods=7,
            ).mean()
        )
    )

    # selisih harga saat ini dengan observasi sebelumnya
    data["price_change"] = (
        data.groupby(["city", "commodity"])["price"]
        .diff()
    )

    # fitur kalender
    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month

    # forecast harga
    data["price_next_day"] = (
        data.groupby(["city", "commodity"])["price"]
        .shift(-1)
    )

    # hapus baris yang belum memiliki fitur lag atau target
    feature_columns = [
        "price_lag_1",
        "price_lag_7",
        "rolling_mean_7",
        "price_change",
    ] + available_weather

    data = data.dropna(subset=feature_columns + ["price_next_day"]).copy()


    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_FILE, index=False)

    print(f"[SUCCESS] Feature dataset: {OUTPUT_FILE}")
    print(f"[INFO] Total row: {len(data)}")
    print(f"[INFO] Total cities: {data['city'].nunique()}")
    print(f"[INFO] Total commodities: " f"{data['commodity'].nunique()}")
    print(f"[INFO] Available weather features: {available_weather}")

    return data


def main():
    prices, weather = load_data()
    create_features(prices, weather)


if __name__ == "__main__":
    main()
