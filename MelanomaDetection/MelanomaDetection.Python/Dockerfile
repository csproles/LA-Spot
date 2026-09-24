# syntax=docker/dockerfile:1.7
# Build context is the REPO ROOT (see docker-compose.yml), not this directory --
# v5_detector.py imports sibling packages (Code/, revised_abcd/, pipeline_v5/)
# that live next to MelanomaDetection/, not inside it, so the image has to
# mirror the repo layout instead of flattening this service into /app.

# Risk model weights (~1.8GB: 5 CNN checkpoints + PCA/CatBoost pairs + a YOLO
# segmentation model -- see NOTES.md). Locally these are bind-mounted from the
# repo root's gitignored models/ (docker-compose.yml); a plain Docker build has
# no such mount and can't COPY a file that was never committed to git, so
# instead this stage downloads a zip of that same folder from Google Drive
# (shared "anyone with the link" -- DRIVE_FILE_ID is not a secret, so unlike
# the private-GitHub-release approach this used to use, there's nothing here
# for Railway's build-log renderer to leak: it prints resolved ARG values into
# step descriptions, which is what kept exposing GITHUB_TOKEN). gdown handles
# Drive's large-file download-confirmation flow, which plain curl doesn't.
# Still its own stage, only ever COPY --from=models'd for its extracted files,
# so none of this stage's tooling ends up in the final image.
FROM python:3.12-slim AS models
WORKDIR /models
ARG DRIVE_FILE_ID=1IoveXXDaiaK05ChaRJHVw5A4aChyGPEj
RUN apt-get update && apt-get install -y --no-install-recommends unzip \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir gdown \
    && gdown "$DRIVE_FILE_ID" -O app_models.zip \
    && unzip -q app_models.zip -d . \
    && rm app_models.zip \
    && find . -mindepth 2 -type f -exec mv -t . {} + \
    && find . -mindepth 1 -type d -empty -delete \
    && MISSING="" \
    && for f in yolo_melanoma_seg.pt \
                fold_0_best.pt fold_1_best.pt fold_2_best.pt fold_3_best.pt fold_4_best.pt \
                pca_app_fold_0.pkl pca_app_fold_1.pkl pca_app_fold_2.pkl pca_app_fold_3.pkl pca_app_fold_4.pkl \
                catboost_app_fold_0.cbm catboost_app_fold_1.cbm catboost_app_fold_2.cbm catboost_app_fold_3.cbm catboost_app_fold_4.cbm \
                catboost_app_fold_0_feature_names.npy catboost_app_fold_1_feature_names.npy catboost_app_fold_2_feature_names.npy catboost_app_fold_3_feature_names.npy catboost_app_fold_4_feature_names.npy; \
       do [ -f "$f" ] || MISSING="$MISSING $f"; done \
    && if [ -n "$MISSING" ]; then echo "Missing from the models zip:$MISSING"; exit 1; fi

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
# Bound to [::] (all interfaces, IPv6) rather than 0.0.0.0: Railway's private
# network is IPv6-only, so an IPv4-only bind resolves fine over
# *.railway.internal DNS but has nothing actually listening on the address
# traffic arrives on. [::] still accepts IPv4 connections too (dual-stack on
# Linux by default), so this doesn't break local `docker compose` usage.
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "60", "--bind", "[::]:5002", "main:app"]
