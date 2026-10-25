FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_VISIBLE_DEVICES=-1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_NUM_INTRAOP_THREADS=4 \
    TF_NUM_INTEROP_THREADS=2 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    MODEL_DIR=/models/champion \
    DATABASE_URL=sqlite:////data/visionguard.sqlite3

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 visionguard \
    && useradd --uid 10001 --gid visionguard --no-create-home visionguard \
    && mkdir -p /data /models/champion \
    && chown visionguard:visionguard /data

# Keep this dependency layer reusable when application source changes.
COPY requirements-web.txt ./
RUN python -m pip install --no-cache-dir -r requirements-web.txt

COPY requirements-db.txt ./
RUN python -m pip install --no-cache-dir -r requirements-db.txt

COPY task4/ ./task4/
COPY visionguard/ ./visionguard/

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('APP_PORT','8000')+'/health',timeout=3).read()" || exit 1
CMD ["python", "-m", "task4.run", "--server", "gunicorn"]
