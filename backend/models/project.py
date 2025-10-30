"""
Project model
"""

from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
import uuid

from backend.database.base import Base


class TaskType(str, enum.Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"


class Project(Base):
    __tablename__ = "projects"
    
    # Primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Basic info
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    task_type = Column(SQLEnum(TaskType), nullable=False)
    
    # Owner
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Settings
    workspace_path = Column(String, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="projects")
    datasets = relationship("Dataset", back_populates="project", cascade="all, delete-orphan")
    training_jobs = relationship("TrainingJob", back_populates="project", cascade="all, delete-orphan")
    models = relationship("Model", back_populates="project", cascade="all, delete-orphan")
    deployments = relationship("Deployment", back_populates="project", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Project(id={self.id}, name={self.name})>"
    
    @property
    def dataset_count(self) -> int:
        return len(self.datasets) if self.datasets else 0
    
    @property
    def model_count(self) -> int:
        return len(self.models) if self.models else 0