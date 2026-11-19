
FROM python:3.12-slim

# Reduce unnecessary Python file
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY project/deployment/requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Copy the API code, exported model, and feature data used by /predict/latest.
COPY project/deployment/app.py .
COPY project/deployment/model.joblib .
COPY project/data/processed/food_price_features.csv .

# Uvicorn listens on port 8000
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
