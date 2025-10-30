"""
Celery worker for model deployment
"""

from backend.workers.celery_app import celery_app
from backend.database.base import SessionLocal
from backend.models.deployment import Deployment, DeploymentStatus
from backend.models.model import Model
import traceback
from datetime import datetime


@celery_app.task(bind=True, name="backend.workers.deployment_worker.deploy_model")
def deploy_model(self, deployment_id: str):
    """
    Background task to deploy a model
    
    Args:
        deployment_id: Deployment ID
    """
    
    print(f"🚀 Deploying model: {deployment_id}")
    
    db = SessionLocal()
    try:
        deployment = db.query(Deployment).filter(Deployment.id == deployment_id).first()
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        model = db.query(Model).filter(Model.id == deployment.model_id).first()
        if not model:
            raise ValueError("Model not found")
        
        deployment.status = DeploymentStatus.CREATING
        db.commit()
        
        self.update_state(
            state='PROGRESS',
            meta={'status': 'preparing', 'progress': 20}
        )
        
        # Simulate deployment (implement actual logic here)
        if deployment.deployment_type.value == "api":
            print(f"📡 Deploying as REST API...")
            deployment.endpoint_url = f"https://api.modelforge.ai/v1/predict/{deployment.id}"
            
            import secrets
            deployment.api_key = secrets.token_urlsafe(32)
            deployment.instance_type = "cpu"
            deployment.estimated_cost_per_hour = 0.10
            
        elif deployment.deployment_type.value == "demo":
            print(f"🎨 Deploying as Gradio demo...")
            deployment.endpoint_url = f"https://demo.modelforge.ai/{deployment.id}"
            deployment.instance_type = "cpu"
            deployment.estimated_cost_per_hour = 0.05
        
        self.update_state(
            state='PROGRESS',
            meta={'status': 'activating', 'progress': 80}
        )
        
        deployment.activate()
        model.is_deployed = True
        
        db.commit()
        
        print(f"✅ Deployment completed!")
        
        return {
            'status': 'completed',
            'deployment_id': deployment_id,
            'endpoint_url': deployment.endpoint_url
        }
    
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Deployment failed: {error_msg}")
        
        deployment.status = DeploymentStatus.FAILED
        db.commit()
        
        raise
    
    finally:
        db.close()


@celery_app.task(name="backend.workers.deployment_worker.update_deployment_costs")
def update_deployment_costs():
    """Periodic task to update deployment costs"""
    db = SessionLocal()
    try:
        active_deployments = db.query(Deployment).filter(
            Deployment.status == DeploymentStatus.ACTIVE
        ).all()
        
        for deployment in active_deployments:
            if deployment.deployed_at:
                uptime_hours = (datetime.utcnow() - deployment.deployed_at).total_seconds() / 3600
                deployment.total_cost = uptime_hours * deployment.estimated_cost_per_hour
        
        db.commit()
        
        print(f"💰 Updated costs for {len(active_deployments)} deployments")
        
        return {'status': 'success', 'updated_count': len(active_deployments)}
    
    finally:
        db.close()