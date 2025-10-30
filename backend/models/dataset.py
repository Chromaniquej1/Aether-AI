"""
Dataset model
"""

from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
import uuid

from backend.database.base import Base


class DatasetStatus(str, enum.Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    INVALID = "invalid"
    FAILED = "failed"


class Dataset(Base):
    __tablename__ = "datasets"
    
    # Primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Basic info
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    task_type = Column(String, nullable=False)
    
    # Project reference
    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    
    # Status
    status = Column(SQLEnum(DatasetStatus), default=DatasetStatus.UPLOADING)
    
    # File info
    file_path = Column(String, nullable=True)
    storage_path = Column(String, nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    
    # Dataset metadata
    num_classes = Column(Integer, nullable=True)
    total_images = Column(Integer, nullable=True)
    class_names = Column(JSON, nullable=True)
    class_distribution = Column(JSON, nullable=True)
    image_stats = Column(JSON, nullable=True)
    
    # Validation
    is_valid = Column(Boolean, default=False)
    validation_errors = Column(JSON, nullable=True)
    
    # Preprocessing config
    preprocessing_config = Column(JSON, nullable=True)
    
    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    processed_at = Column(DateTime, nullable=True)
    
    # Relationships
    project = relationship("Project", back_populates="datasets")
    training_jobs = relationship("TrainingJob", back_populates="dataset")
    
    def __repr__(self):
        return f"<Dataset(id={self.id}, name={self.name})>"
    
    @property
    def is_ready(self) -> bool:
        return self.status == DatasetStatus.READY and self.is_valid
    
    def mark_as_ready(self):
        self.status = DatasetStatus.READY
        self.processed_at = datetime.utcnow()
    
    def mark_as_failed(self, errors: list):
        self.status = DatasetStatus.FAILED
        self.validation_errors = errors
        self.processed_at = datetime.utcnow()