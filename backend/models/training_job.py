"""
Training job model
"""

from sqlalchemy import Column, String, Text, Integer, Float, DateTime, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
import uuid

from backend.database.base import Base


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingJob(Base):
    __tablename__ = "training_jobs"
    
    # Primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Basic info
    name = Column(String, nullable=True)
    
    # References
    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    dataset_id = Column(String, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)
    
    # Status
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, index=True)
    
    # Configuration
    backbone = Column(String, nullable=False)
    finetuning_mode = Column(String, nullable=False)
    config = Column(JSON, nullable=False)
    
    # Training parameters
    epochs = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    
    # Progress tracking
    current_epoch = Column(Integer, default=0)
    progress_percentage = Column(Float, default=0.0)
    
    # Metrics
    metrics = Column(JSON, nullable=True)
    best_val_acc = Column(Float, nullable=True)
    best_val_loss = Column(Float, nullable=True)
    final_train_loss = Column(Float, nullable=True)
    final_train_acc = Column(Float, nullable=True)
    
    # Resource usage
    gpu_hours = Column(Float, default=0.0)
    gpu_memory_peak_mb = Column(Float, nullable=True)
    estimated_cost = Column(Float, default=0.0)
    
    # Artifacts
    checkpoint_path = Column(String, nullable=True)
    logs_path = Column(String, nullable=True)
    tensorboard_path = Column(String, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    error_traceback = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Celery task info
    celery_task_id = Column(String, nullable=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    
    # Relationships
    project = relationship("Project", back_populates="training_jobs")
    dataset = relationship("Dataset", back_populates="training_jobs")
    model = relationship("Model", back_populates="training_job", uselist=False)
    
    def __repr__(self):
        return f"<TrainingJob(id={self.id}, status={self.status})>"
    
    @property
    def is_active(self) -> bool:
        return self.status in [JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING]
    
    @property
    def duration_seconds(self) -> float:
        if not self.started_at:
            return 0.0
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()
    
    def start(self):
        self.status = JobStatus.RUNNING
        self.started_at = datetime.utcnow()
    
    def complete(self, metrics: dict):
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.metrics = metrics
        if metrics:
            self.best_val_acc = metrics.get('best_val_acc')
            self.best_val_loss = metrics.get('best_val_loss')
    
    def fail(self, error: str, traceback: str = None):
        self.status = JobStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error
        self.error_traceback = traceback
    
    def update_progress(self, epoch: int, metrics: dict):
        self.current_epoch = epoch
        self.progress_percentage = (epoch / self.epochs) * 100 if self.epochs > 0 else 0
        if metrics:
            if not self.metrics:
                self.metrics = []
            self.metrics.append(metrics)