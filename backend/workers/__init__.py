"""
Workers package initialization
"""

from backend.workers.celery_app import celery_app

__all__ = ['celery_app']