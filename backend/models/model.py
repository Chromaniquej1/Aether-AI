"""
Model registry model
"""

from sqlalchemy import Column, String, Text, Integer, Float, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from backend.database.base import Base


class Model(Base):
    __tablename__ = "models"
    
    # Primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Basic info
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    version = Column(String, default="1.0", nullable=False)
    
    # References
    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    training_job_id = Column(String, ForeignKey("training_jobs.id", ondelete="SET NULL"), nullable=True)
    
    # Model configuration
    backbone = Column(String, nullable=False)
    finetuning_mode = Column(String, nullable=False)
    num_classes = Column(Integer, nullable=False)
    class_names = Column(JSON, nullable=False)
    
    # Model artifacts
    model_path = Column(String, nullable=False)
    checkpoint_path = Column(String, nullable=True)
    onnx_path = Column(String, nullable=True)
    config_path = Column(String, nullable=True)
    
    # File info
    model_size_bytes = Column(Integer, nullable=True)
    
    # Performance metrics
    best_val_acc = Column(Float, nullable=True)
    best_val_loss = Column(Float, nullable=True)
    best_val_f1 = Column(Float, nullable=True)
    train_acc = Column(Float, nullable=True)
    train_loss = Column(Float, nullable=True)
    
    # Training summary
    total_epochs = Column(Integer, nullable=True)
    training_time_minutes = Column(Float, nullable=True)
    dataset_size = Column(Integer, nullable=True)
    
    # Metadata
    metadata = Column(JSON, nullable=True)
    tags = Column(JSON, nullable=True)
    
    # Status
    is_public = Column(Boolean, default=False)
    is_deployed = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="models")
    training_job = relationship("TrainingJob", back_populates="model")
    deployments = relationship("Deployment", back_populates="model", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Model(id={self.id}, name={self.name})>"
    
    @property
    def size_mb(self) -> float:
        return self.model_size_bytes / (1024 * 1024) if self.model_size_bytes else 0