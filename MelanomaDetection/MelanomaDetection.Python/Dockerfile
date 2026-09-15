# Place this file at: MelanomaDetection/MelanomaDetection.Python/Dockerfile

FROM python:3.12-slim

WORKDIR /app

# opencv-python needs these system libs even though it's "headless" at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5002

CMD ["python", "main.py"]
