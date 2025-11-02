"""
Deployment model
"""

from sqlalchemy import Column, String, Text, Integer, Float, DateTime, ForeignKey, JSON, Boolean, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
import uuid

from backend.database.base import Base


class DeploymentType(str, enum.Enum):
    API = "api"
    DEMO = "demo"
    BATCH = "batch"
    EDGE = "edge"


class DeploymentStatus(str, enum.Enum):
    CREATING = "creating"
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    DELETED = "deleted"


class Deployment(Base):
    __tablename__ = "deployments"
    
    # Primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Basic info
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    
    # References
    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    model_id = Column(String, ForeignKey("models.id", ondelete="CASCADE"), nullable=False)
    
    # Deployment configuration
    deployment_type = Column(SQLEnum(DeploymentType), nullable=False)
    status = Column(SQLEnum(DeploymentStatus), default=DeploymentStatus.CREATING)
    
    # Endpoint information
    endpoint_url = Column(String, nullable=True)
    api_key = Column(String, nullable=True)
    
    # Infrastructure
    instance_type = Column(String, nullable=True)
    replicas = Column(Integer, default=1)
    auto_scaling_enabled = Column(Boolean, default=False)
    
    # Configuration
    config = Column(JSON, nullable=True)
    
    # Usage tracking
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    avg_latency_ms = Column(Float, nullable=True)
    
    # Cost tracking
    estimated_cost_per_hour = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)
    
    # Health monitoring
    last_health_check = Column(DateTime, nullable=True)
    health_status = Column(String, default="unknown")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    deployed_at = Column(DateTime, nullable=True)
    stopped_at = Column(DateTime, nullable=True)
    
    # Relationships
    project = relationship("Project", back_populates="deployments")
    model = relationship("Model", back_populates="deployments")
    
    def __repr__(self):
        return f"<Deployment(id={self.id}, name={self.name})>"
    
    @property
    def is_active(self) -> bool:
        return self.status == DeploymentStatus.ACTIVE
    
    def activate(self):
        self.status = DeploymentStatus.ACTIVE
        self.deployed_at = datetime.utcnow()
    
    def deactivate(self):
        self.status = DeploymentStatus.INACTIVE
        self.stopped_at = datetime.utcnow()