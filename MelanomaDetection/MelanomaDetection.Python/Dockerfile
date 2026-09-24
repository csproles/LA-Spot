# syntax=docker/dockerfile:1.7
# Build context is the REPO ROOT (see docker-compose.yml), not this directory --
# v5_detector.py imports sibling packages (Code/, revised_abcd/, pipeline_v5/)
# that live next to MelanomaDetection/, not inside it, so the image has to
# mirror the repo layout instead of flattening this service into /app.

# Risk model weights (~1.8GB: 5 CNN checkpoints + PCA/CatBoost pairs + a YOLO
# segmentation model -- see NOTES.md). Locally these are bind-mounted from the
# repo root's gitignored models/ (docker-compose.yml); a plain Docker build has
# no such mount and can't COPY a file that was never committed to git, so
# instead this stage downloads a zip of that same folder from a private GitHub
# Release asset using a token. It's its own stage, built separately from the
# final image below and only ever COPY --from=models'd for its extracted
# files, so GITHUB_TOKEN never ends up in a layer of the image that actually
# ships (ARG values otherwise persist in layer history -- Railway doesn't
# support BuildKit's --mount=type=secret, which would avoid this outright).
FROM python:3.12-slim AS models
WORKDIR /models
ARG GITHUB_TOKEN
ARG GITHUB_REPO=csproles/LA-Spot
ARG MODELS_RELEASE_TAG=models-v1
ARG MODELS_ASSET_NAME=models.zip
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip \
    && rm -rf /var/lib/apt/lists/* \
    && test -n "$GITHUB_TOKEN" || { echo "GITHUB_TOKEN build arg is required (models-v1 is a private-repo release asset)"; exit 1; } \
    && ASSET_API_URL=$(curl -fsSL \
         -H "Authorization: Bearer $GITHUB_TOKEN" \
         -H "Accept: application/vnd.github+json" \
         "https://api.github.com/repos/$GITHUB_REPO/releases/tags/$MODELS_RELEASE_TAG" \
       | python3 -c "import sys,json; d=json.load(sys.stdin); print(next(a['url'] for a in d['assets'] if a['name']=='$MODELS_ASSET_NAME'))") \
    && curl -fsSL \
         -H "Authorization: Bearer $GITHUB_TOKEN" \
         -H "Accept: application/octet-stream" \
         "$ASSET_API_URL" -o models.zip \
    && unzip -q models.zip -d . \
    && rm models.zip \
    && find . -mindepth 2 -type f -exec mv -t . {} + \
    && find . -mindepth 1 -type d -empty -delete \
    && test -f yolo_melanoma_seg.pt || { echo "yolo_melanoma_seg.pt not found anywhere in the extracted zip"; exit 1; }

FROM python:3.12-slim

WORKDIR /app

# opencv-python needs these system libs even though it's "headless" at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=models /models models

COPY MelanomaDetection/MelanomaDetection.Python/requirements.txt MelanomaDetection/MelanomaDetection.Python/requirements.txt
# --extra-index-url pulls CPU-only PyTorch wheels: the default PyPI build drags
# in several GB of NVIDIA CUDA runtime packages (cublas, cudnn, cufft, ...)
# that this container -- no GPU, ever -- never uses. No cache mount here:
# Railway's builder requires cache mount ids in its own scoped format
# (id=s/<service-id>-<path>), which isn't worth hand-wiring for a build-speed
# optimization; a plain install is slower to rebuild but has no such gotcha.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r MelanomaDetection/MelanomaDetection.Python/requirements.txt

COPY Code Code
COPY revised_abcd revised_abcd
COPY pipeline_v5 pipeline_v5
COPY melanoma_pipeline melanoma_pipeline
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
