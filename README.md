# food-forecast-east-java-cities
# Price Forecasts for Popular Food Staples in East Java Cities

## Project Overview

This project is an MLOps-based system for forecasting the prices of popular food commodities in cities across East Java, Indonesia. The project uses food price data from **PIHPS (Bank Indonesia)** and weather data from **Open-Meteo** as the primary data sources.

The main objective is to build a reproducible machine learning pipeline that can continuously ingest new data, process and evaluate the data, train forecasting models, and support continuous training as new data becomes available or the underlying data distribution changes.

The initial machine learning task is **regression-based forecasting**, where the system aims to predict the next-day price of selected food commodities using historical price information and weather-related features.

## Project Objectives

The main objectives of this project are:

- Collect periodically updated food price data from PIHPS.
- Collect weather data from Open-Meteo.
- Process and combine food price and weather data.
- Develop a model for next-day food price forecasting.
- Evaluate model performance using MAE, RMSE, and MAPE.
- Build a reproducible MLOps pipeline.
- Implement a continuous training strategy so the model can be updated when new data becomes available.

## Project Structure

The repository uses the following directory structure:

food-price-forecasts-east-java-cities/
│
├── .devcontainer/
│   └── devcontainer.json
│
├── .github/
│
├── project/
│   ├── configs/
│   ├── data/
|   |   └── raw
|   |   └── processed
│   ├── docs/
│   ├── models/
│   ├── notebooks/
│   ├── src/
│   └── tests/
│
├── .gitignore
├── LICENSE
└── README.md

