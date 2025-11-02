"""
Celery worker for training jobs
"""

from celery import Task
from backend.workers.celery_app import celery_app
from backend.database.base import SessionLocal
from backend.models.training_job import TrainingJob, JobStatus
from backend.models.model import Model
from backend.core.pipeline import ModelForgePipeline
import traceback
from datetime import datetime, timedelta
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TrainingTask(Task):
    """Base task with database session management"""
    
    def __call__(self, *args, **kwargs):
        db = SessionLocal()
        try:
            return self.run(*args, **kwargs, db=db)
        finally:
            db.close()


@celery_app.task(bind=True, base=TrainingTask, name="backend.workers.training_worker.train_model")
def train_model(self, job_id: str, db=None):
    """
    Background task to train a model
    
    Args:
        job_id: Training job ID
        db: Database session (injected by TrainingTask)
    """
    
    print(f"🚀 Starting training job: {job_id}")
    
    # Get job from database
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise ValueError(f"Training job {job_id} not found")
    
    # Update job status
    job.status = JobStatus.RUNNING
    job.started_at = datetime.utcnow()
    job.celery_task_id = self.request.id
    db.commit()
    
    try:
        # Get dataset
        dataset = job.dataset
        if not dataset or not dataset.is_ready:
            raise ValueError("Dataset not ready for training")
        
        # Initialize pipeline
        pipeline = ModelForgePipeline(
            project_name=job.project_id,
            workspace_dir="./workspaces"
        )
        
        # Create model
        print(f"📦 Creating model: {job.backbone}")
        model = pipeline.create_model(
            num_classes=dataset.num_classes,
            backbone=job.backbone,
            finetuning_mode=job.finetuning_mode
        )
        
        # Training configuration
        config = job.config
        
        # Train model
        print(f"🏋️ Training model...")
        result = pipeline.train_model(
            model=model,
            dataset_path=dataset.storage_path,
            experiment_name=f"job_{job_id}",
            epochs=config.get('epochs', 30),
            batch_size=config.get('batch_size', 32),
            learning_rate=config.get('learning_rate', 1e-4),
            early_stopping_patience=config.get('early_stopping_patience', 10)
        )
        
        # Update job as completed
        job.complete(result['summary'])
        job.checkpoint_path = result['model_path']
        
        # Calculate GPU usage
        if job.started_at:
            elapsed_hours = (datetime.utcnow() - job.started_at).total_seconds() / 3600
            job.gpu_hours = elapsed_hours
            job.estimated_cost = elapsed_hours * 1.0
        
        db.commit()
        
        # Create model entry in registry
        model_entry = Model(
            name=f"Model_{job_id[:8]}",
            project_id=job.project_id,
            training_job_id=job.id,
            backbone=job.backbone,
            finetuning_mode=job.finetuning_mode,
            num_classes=dataset.num_classes,
            class_names=dataset.class_names,
            model_path=result['model_path'],
            model_size_bytes=0,
            best_val_acc=result['summary'].get('best_val_acc'),
            best_val_loss=result['summary'].get('best_val_loss'),
            total_epochs=result['summary'].get('total_epochs'),
            training_time_minutes=job.duration_seconds / 60 if job.started_at else 0,
            dataset_size=dataset.total_images
        )
        
        db.add(model_entry)
        db.commit()
        
        print(f"✅ Training completed successfully!")
        print(f"   Model saved at: {result['model_path']}")
        
        return {
            'status': 'completed',
            'job_id': job_id,
            'model_id': model_entry.id,
            'metrics': result['summary']
        }
    
    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        
        print(f"❌ Training failed: {error_msg}")
        
        job.fail(error_msg, error_traceback)
        db.commit()
        
        raise


@celery_app.task(name="backend.workers.training_worker.cancel_training_job")
def cancel_training_job(job_id: str):
    """Cancel a running training job"""
    db = SessionLocal()
    try:
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            return {'status': 'error', 'message': 'Job not found'}
        
        if not job.is_active:
            return {'status': 'error', 'message': 'Job is not active'}
        
        # Revoke Celery task
        if job.celery_task_id:
            celery_app.control.revoke(job.celery_task_id, terminate=True)
        
        # Update job status
        job.status = JobStatus.CANCELLED
        job.cancelled_at = datetime.utcnow()
        db.commit()
        
        print(f"🛑 Training job cancelled: {job_id}")
        
        return {'status': 'cancelled', 'job_id': job_id}
    
    finally:
        db.close()


@celery_app.task(name="backend.workers.training_worker.cleanup_old_jobs")
def cleanup_old_jobs():
    """Periodic task to cleanup old completed/failed jobs"""
    db = SessionLocal()
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        
        old_jobs = db.query(TrainingJob).filter(
            TrainingJob.completed_at < cutoff_date,
            TrainingJob.status.in_([JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED])
        ).all()
        
        count = len(old_jobs)
        
        for job in old_jobs:
            db.delete(job)
        
        db.commit()
        
        print(f"🧹 Cleaned up {count} old training jobs")
        
        return {'status': 'success', 'deleted_count': count}
    
    finally:
        db.close()