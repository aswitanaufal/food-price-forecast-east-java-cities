
FROM python:3.12-slim

# Reduce unnecessary Python file
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Copy the API code and the exported trained model to the image
COPY app.py .
COPY model.joblib .

# Flask listens on port 8000
EXPOSE 8000

CMD ["python", "app.py"]
