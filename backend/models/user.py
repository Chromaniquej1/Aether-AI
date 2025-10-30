"""
User model for authentication and authorization
"""

from sqlalchemy import Column, String, Boolean, DateTime, Integer, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
import uuid

from backend.database.base import Base


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class SubscriptionTier(str, enum.Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class User(Base):
    __tablename__ = "users"
    
    # Primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Basic info
    email = Column(String, unique=True, nullable=False, index=True)
    username = Column(String, unique=True, nullable=True, index=True)
    full_name = Column(String, nullable=True)
    
    # Authentication
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # Authorization
    role = Column(SQLEnum(UserRole), default=UserRole.USER)
    
    # Subscription
    subscription_tier = Column(SQLEnum(SubscriptionTier), default=SubscriptionTier.FREE)
    subscription_expires_at = Column(DateTime, nullable=True)
    
    # OAuth
    oauth_provider = Column(String, nullable=True)
    oauth_id = Column(String, nullable=True)
    
    # API Key
    api_key = Column(String, unique=True, nullable=True, index=True)
    
    # Usage tracking
    gpu_hours_used = Column(Integer, default=0)
    storage_bytes_used = Column(Integer, default=0)
    api_calls_count = Column(Integer, default=0)
    
    # Limits
    max_projects = Column(Integer, default=3)
    max_gpu_hours_per_month = Column(Integer, default=10)
    max_storage_gb = Column(Integer, default=5)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)
    
    # Relationships
    projects = relationship("Project", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"
    
    @property
    def is_admin(self) -> bool:
        return self.role == UserRole.ADMIN
    
    @property
    def is_pro(self) -> bool:
        return self.subscription_tier in [SubscriptionTier.PRO, SubscriptionTier.ENTERPRISE]