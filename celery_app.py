"""
VocalFusion Celery Application
===============================
Broker + backend: Redis (default: redis://localhost:6379/0)
Set REDIS_URL env var to override.

Start worker:
    celery -A celery_app worker --loglevel=info --concurrency=1
"""
import os
from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "vocalfusion",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
    worker_prefetch_multiplier=1,   # one task at a time (GPU-bound)
    worker_max_tasks_per_child=10,  # restart after 10 tasks (GPU mem cleanup)
    task_acks_late=True,            # ack after completion, not pickup
    task_reject_on_worker_lost=True,
)
