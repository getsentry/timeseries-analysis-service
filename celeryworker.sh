#!/bin/bash

if [ "$CELERY_WORKER_ENABLE" = "true" ]; then
    exec celery -A src.celery_app.tasks worker --loglevel=info -c 4 # 4 workers for now
else
    echo "Celery worker is disabled via environment variable CELERY_WORKER_ENABLE"
    exit 0
fi
