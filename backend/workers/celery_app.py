"""
Celery application configuration for background task processing
"""

from celery import Celery
from celery.schedules import crontab
import os

# Get Redis URL from environment
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# Create Celery app
celery_app = Celery(
    "modelforge",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
    "backend.workers.training_worker",
    "backend.workers.preprocessing_worker",
    "backend.workers.deployment_worker",
    "backend.workers.inference_worker",  # ← ADD THIS
]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task execution
    task_track_started=True,
    task_time_limit=86400,  # 24 hours max
    task_soft_time_limit=82800,  # 23 hours soft limit
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    
    # Results
    result_expires=3600,
    result_backend_transport_options={
        'visibility_timeout': 86400,
    },
    
    # Retry policy
    task_default_retry_delay=60,
    task_max_retries=3,
    
    # Worker settings
    worker_max_tasks_per_child=50,
    worker_disable_rate_limits=False,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        'cleanup-old-jobs': {
            'task': 'backend.workers.training_worker.cleanup_old_jobs',
            'schedule': crontab(hour=2, minute=0),
        },
        'update-deployment-costs': {
            'task': 'backend.workers.deployment_worker.update_deployment_costs',
            'schedule': crontab(minute='*/30'),
        },
    },
)

# Task routes
celery_app.conf.task_routes = {
    'backend.workers.training_worker.*': {'queue': 'training'},
    'backend.workers.preprocessing_worker.*': {'queue': 'preprocessing'},
    'backend.workers.deployment_worker.*': {'queue': 'deployment'},
    'backend.workers.inference_worker.*': {'queue': 'inference'},  
}

# Logging
celery_app.conf.worker_hijack_root_logger = False
celery_app.conf.worker_send_task_events = True
celery_app.conf.task_send_sent_event = True


# Celery signals
from celery.signals import task_prerun, task_postrun, task_failure

@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, **kwargs):
    print(f"📌 Task starting: {task.name} [{task_id}]")

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, state=None, **kwargs):
    print(f"✅ Task finished: {sender.name} [{task_id}] - State: {state}")

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, **kwargs):
    print(f"❌ Task failed: {sender.name} [{task_id}] - Error: {exception}")


if __name__ == '__main__':
    celery_app.start()