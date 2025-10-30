"""
ModelForge-CV: FastAPI Backend with Database & Celery Integration
"""

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
from pathlib import Path
import shutil
import os

# Database imports
from backend.database.base import get_db, init_db
from backend.models.user import User
from backend.models.project import Project, TaskType
from backend.models.dataset import Dataset, DatasetStatus
from backend.models.training_job import TrainingJob, JobStatus
from backend.models.model import Model
from backend.models.deployment import Deployment

# Worker imports
from backend.workers.celery_app import celery_app
from backend.workers import training_worker, preprocessing_worker, deployment_worker

# Pydantic schemas
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================================================
# Pydantic Schemas
# ============================================================================

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    task_type: str = "classification"


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    task_type: str
    created_at: datetime
    dataset_count: int
    model_count: int
    
    class Config:
        from_attributes = True


class DatasetResponse(BaseModel):
    id: str
    name: str
    status: str
    num_classes: Optional[int]
    total_images: Optional[int]
    is_valid: bool
    uploaded_at: datetime
    
    class Config:
        from_attributes = True


class TrainingJobCreate(BaseModel):
    project_id: str
    dataset_id: str
    backbone: str = "resnet50"
    finetuning_mode: str = "lora"
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-4
    early_stopping_patience: int = 10


class TrainingJobResponse(BaseModel):
    id: str
    project_id: str
    dataset_id: str
    status: str
    backbone: str
    finetuning_mode: str
    progress_percentage: float
    current_epoch: int
    epochs: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class ModelResponse(BaseModel):
    id: str
    name: str
    backbone: str
    num_classes: int
    best_val_acc: Optional[float]
    is_deployed: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="ModelForge-CV API",
    description="End-to-End AutoML Platform for Computer Vision",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Workspace directory
WORKSPACE_DIR = Path("./workspaces")
WORKSPACE_DIR.mkdir(exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()
    print("✅ Database initialized")
    print("✅ FastAPI server started")


@app.on_event("shutdown")
async def shutdown_event():
    print("👋 FastAPI server shutting down")


# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)
    
    async def send_update(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except:
                    pass

manager = ConnectionManager()


# ============================================================================
# Project Management Endpoints
# ============================================================================

@app.post("/api/projects", response_model=ProjectResponse, status_code=201)
async def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    """Create a new project"""
    
    # Get or create demo user
    default_user = db.query(User).filter(User.email == "demo@modelforge.ai").first()
    if not default_user:
        default_user = User(
            email="jayantbiradar619@gmail.com",
            username="demo",
            hashed_password="dummy_hash",
            is_active=True
        )
        db.add(default_user)
        db.commit()
        db.refresh(default_user)
    
    # Create project
    db_project = Project(
        name=project.name,
        description=project.description,
        task_type=TaskType(project.task_type),
        user_id=default_user.id,
        workspace_path=str(WORKSPACE_DIR / f"project_{project.name}")
    )
    
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    
    # Create project directories
    project_dir = WORKSPACE_DIR / db_project.id
    project_dir.mkdir(exist_ok=True)
    (project_dir / "datasets").mkdir(exist_ok=True)
    (project_dir / "models").mkdir(exist_ok=True)
    (project_dir / "experiments").mkdir(exist_ok=True)
    
    return ProjectResponse(
        id=db_project.id,
        name=db_project.name,
        description=db_project.description,
        task_type=db_project.task_type.value,
        created_at=db_project.created_at,
        dataset_count=0,
        model_count=0
    )


@app.get("/api/projects", response_model=List[ProjectResponse])
async def list_projects(db: Session = Depends(get_db)):
    """List all projects"""
    projects = db.query(Project).all()
    
    return [
        ProjectResponse(
            id=p.id,
            name=p.name,
            description=p.description,
            task_type=p.task_type.value,
            created_at=p.created_at,
            dataset_count=p.dataset_count,
            model_count=p.model_count
        )
        for p in projects
    ]


@app.get("/api/projects/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str, db: Session = Depends(get_db)):
    """Get project details"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        task_type=project.task_type.value,
        created_at=project.created_at,
        dataset_count=project.dataset_count,
        model_count=project.model_count
    )


# ============================================================================
# Dataset Management Endpoints
# ============================================================================

@app.post("/api/projects/{project_id}/datasets", response_model=DatasetResponse, status_code=201)
async def upload_dataset(
    project_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and process a dataset"""
    
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Create dataset entry
    dataset = Dataset(
        name=file.filename,
        task_type=project.task_type.value,
        project_id=project_id,
        status=DatasetStatus.UPLOADING
    )
    
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    
    # Save uploaded file
    project_dir = WORKSPACE_DIR / project_id / "datasets"
    project_dir.mkdir(parents=True, exist_ok=True)
    file_path = project_dir / f"{dataset.id}.zip"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    dataset.file_path = str(file_path)
    db.commit()
    
    # Process dataset in background
    preprocessing_worker.process_dataset.apply_async(
        args=[dataset.id],
        queue='preprocessing'
    )
    
    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        status=dataset.status.value,
        num_classes=dataset.num_classes,
        total_images=dataset.total_images,
        is_valid=dataset.is_valid,
        uploaded_at=dataset.uploaded_at
    )


@app.get("/api/projects/{project_id}/datasets", response_model=List[DatasetResponse])
async def list_datasets(project_id: str, db: Session = Depends(get_db)):
    """List all datasets in a project"""
    datasets = db.query(Dataset).filter(Dataset.project_id == project_id).all()
    
    return [
        DatasetResponse(
            id=d.id,
            name=d.name,
            status=d.status.value,
            num_classes=d.num_classes,
            total_images=d.total_images,
            is_valid=d.is_valid,
            uploaded_at=d.uploaded_at
        )
        for d in datasets
    ]


# ============================================================================
# Training Job Endpoints
# ============================================================================

@app.post("/api/training/jobs", response_model=TrainingJobResponse, status_code=201)
async def create_training_job(job_request: TrainingJobCreate, db: Session = Depends(get_db)):
    """Create and start a training job"""
    
    project = db.query(Project).filter(Project.id == job_request.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    dataset = db.query(Dataset).filter(Dataset.id == job_request.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if not dataset.is_ready:
        raise HTTPException(status_code=400, detail="Dataset is not ready")
    
    # Create training job
    job = TrainingJob(
        project_id=job_request.project_id,
        dataset_id=job_request.dataset_id,
        backbone=job_request.backbone,
        finetuning_mode=job_request.finetuning_mode,
        epochs=job_request.epochs,
        batch_size=job_request.batch_size,
        learning_rate=job_request.learning_rate,
        config=job_request.dict(),
        status=JobStatus.PENDING
    )
    
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Start training in background
    task = training_worker.train_model.apply_async(
        args=[job.id],
        queue='training'
    )
    
    job.celery_task_id = task.id
    db.commit()
    
    return TrainingJobResponse(
        id=job.id,
        project_id=job.project_id,
        dataset_id=job.dataset_id,
        status=job.status.value,
        backbone=job.backbone,
        finetuning_mode=job.finetuning_mode,
        progress_percentage=job.progress_percentage,
        current_epoch=job.current_epoch,
        epochs=job.epochs,
        created_at=job.created_at
    )


async def get_training_job(job_id: str, db: Session = Depends(get_db)):
    """Get training job status"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return TrainingJobResponse(
        id=job.id,
        project_id=job.project_id,
        dataset_id=job.dataset_id,
        status=job.status.value,
        backbone=job.backbone,
        finetuning_mode=job.finetuning_mode,
        progress_percentage=job.progress_percentage,
        current_epoch=job.current_epoch,
        epochs=job.epochs,
        created_at=job.created_at
    )


@app.post("/api/training/jobs/{job_id}/cancel")
async def cancel_training_job(job_id: str, db: Session = Depends(get_db)):
    """Cancel a running training job"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if not job.is_active:
        raise HTTPException(status_code=400, detail="Job is not active")
    
    # Cancel via Celery
    training_worker.cancel_training_job.apply_async(args=[job_id])
    
    return {"status": "cancelling", "job_id": job_id}


@app.get("/api/projects/{project_id}/jobs", response_model=List[TrainingJobResponse])
async def list_training_jobs(project_id: str, db: Session = Depends(get_db)):
    """List all training jobs for a project"""
    jobs = db.query(TrainingJob).filter(TrainingJob.project_id == project_id).all()
    
    return [
        TrainingJobResponse(
            id=j.id,
            project_id=j.project_id,
            dataset_id=j.dataset_id,
            status=j.status.value,
            backbone=j.backbone,
            finetuning_mode=j.finetuning_mode,
            progress_percentage=j.progress_percentage,
            current_epoch=j.current_epoch,
            epochs=j.epochs,
            created_at=j.created_at
        )
        for j in jobs
    ]


# ============================================================================
# Model Management Endpoints
# ============================================================================

@app.get("/api/projects/{project_id}/models", response_model=List[ModelResponse])
async def list_models(project_id: str, db: Session = Depends(get_db)):
    """List all models in a project"""
    models = db.query(Model).filter(Model.project_id == project_id).all()
    
    return [
        ModelResponse(
            id=m.id,
            name=m.name,
            backbone=m.backbone,
            num_classes=m.num_classes,
            best_val_acc=m.best_val_acc,
            is_deployed=m.is_deployed,
            created_at=m.created_at
        )
        for m in models
    ]


@app.get("/api/models/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str, db: Session = Depends(get_db)):
    """Get model details"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return ModelResponse(
        id=model.id,
        name=model.name,
        backbone=model.backbone,
        num_classes=model.num_classes,
        best_val_acc=model.best_val_acc,
        is_deployed=model.is_deployed,
        created_at=model.created_at
    )


# ============================================================================
# WebSocket for Real-time Updates
# ============================================================================

@app.websocket("/ws/training/{job_id}")
async def training_websocket(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time training updates"""
    await manager.connect(websocket, job_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"status": "connected"})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    try:
        db.execute("SELECT 1")
        db_status = "healthy"
    except:
        db_status = "unhealthy"
    
    try:
        celery_status = "healthy" if celery_app.control.inspect().active() else "unhealthy"
    except:
        celery_status = "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" and celery_status == "healthy" else "degraded",
        "database": db_status,
        "celery": celery_status,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "ModelForge-CV API",
        "version": "2.0.0",
        "docs_url": "/docs",
        "features": [
            "PostgreSQL database",
            "Celery async workers",
            "WebSocket real-time updates",
            "Production-ready"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)