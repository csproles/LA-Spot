# Build context is the REPO ROOT (see docker-compose.yml), not this directory --
# v5_detector.py imports sibling packages (Code/, revised_abcd/, pipeline_v5/)
# that live next to MelanomaDetection/, not inside it, so the image has to
# mirror the repo layout instead of flattening this service into /app.

FROM python:3.12-slim

WORKDIR /app

# opencv-python needs these system libs even though it's "headless" at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY MelanomaDetection/MelanomaDetection.Python/requirements.txt MelanomaDetection/MelanomaDetection.Python/requirements.txt
RUN pip install --no-cache-dir -r MelanomaDetection/MelanomaDetection.Python/requirements.txt

COPY Code Code
COPY revised_abcd revised_abcd
COPY pipeline_v5 pipeline_v5
COPY MelanomaDetection/MelanomaDetection.Python MelanomaDetection/MelanomaDetection.Python

WORKDIR /app/MelanomaDetection/MelanomaDetection.Python

# The API runs as an unprivileged user. The entrypoint starts as root only long
# enough to hand /app/data (the database volume) to that user -- volumes made by
# an earlier root-run version are otherwise unwritable -- then drops privileges.
RUN useradd --system --uid 10001 --no-create-home app \
    && mkdir -p /app/data \
    && chown -R app:app /app/data \
    && chmod +x docker-entrypoint.sh

EXPOSE 5002

ENTRYPOINT ["./docker-entrypoint.sh"]

# One worker on purpose: analysis results and rate-limit counters live in this
# process's memory, so a second worker would each see only half of them.
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "60", "--bind", "0.0.0.0:5002", "main:app"]
