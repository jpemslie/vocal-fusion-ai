# VocalFusion — Python API + Celery worker
FROM python:3.11-slim

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
       fastapi \
       "uvicorn[standard]" \
       celery \
       redis

# Copy application code
COPY api_fast.py   .
COPY celery_app.py .
COPY tasks.py      .
COPY fuser.py      .
COPY listen.py     .
COPY reference_profile.json .

# Data directories (mounted as volumes in compose)
RUN mkdir -p vf_data/stems vf_data/mixes vf_data/uploads vf_data/jobs

EXPOSE 8000

# Default: run the API server.
# Override CMD in docker-compose to run the Celery worker.
CMD ["uvicorn", "api_fast:app", "--host", "0.0.0.0", "--port", "8000"]
