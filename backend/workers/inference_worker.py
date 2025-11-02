"""
ModelForge-CV: Inference Worker (Optional)
Celery tasks for asynchronous inference processing
"""

from celery import Task
from backend.workers.celery_app import celery_app
from backend.database.base import SessionLocal
from backend.models.model import Model
from backend.core.inference_engine import InferenceService
from pathlib import Path
import logging
import json
from typing import List, Dict
import os

logger = logging.getLogger(__name__)


class InferenceTask(Task):
    """Base class for inference tasks with database session"""
    
    def __call__(self, *args, **kwargs):
        with SessionLocal() as db:
            return self.run(*args, **kwargs, db=db)


@celery_app.task(
    bind=True,
    base=InferenceTask,
    name='backend.workers.inference_worker.batch_inference',
    max_retries=3
)
def batch_inference(
    self,
    model_id: str,
    image_paths: List[str],
    output_path: str = None,
    db=None
):
    """
    Run inference on a batch of images asynchronously
    
    Args:
        model_id: ID of the model to use
        image_paths: List of paths to images
        output_path: Optional path to save results JSON
        db: Database session (injected by InferenceTask)
    
    Returns:
        Dictionary with results
    """
    try:
        logger.info(f"📌 Starting batch inference task")
        logger.info(f"   - Model ID: {model_id}")
        logger.info(f"   - Images: {len(image_paths)}")
        
        # Get model from database
        model = db.query(Model).filter(Model.id == model_id).first()
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        if not model.model_path or not os.path.exists(model.model_path):
            raise ValueError(f"Model file not found at {model.model_path}")
        
        # Load inference service
        logger.info("🔧 Loading inference service...")
        service = InferenceService(model.model_path)
        
        # Process images
        logger.info("🖼️  Processing images...")
        results = []
        successful = 0
        failed = 0
        
        for idx, image_path in enumerate(image_paths):
            try:
                # Update progress
                progress = (idx + 1) / len(image_paths) * 100
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'current': idx + 1,
                        'total': len(image_paths),
                        'progress': progress,
                        'status': f'Processing image {idx + 1}/{len(image_paths)}'
                    }
                )
                
                # Run prediction
                result = service.predict(image_path)
                
                results.append({
                    'image_path': image_path,
                    'success': True,
                    'predicted_class': result['predicted_class'],
                    'confidence': result['confidence'],
                    'predictions': result['predictions']
                })
                successful += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to process {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'success': False,
                    'error': str(e)
                })
                failed += 1
        
        # Save results if output path specified
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"💾 Results saved to {output_path}")
        
        summary = {
            'status': 'completed',
            'model_id': model_id,
            'total_images': len(image_paths),
            'successful': successful,
            'failed': failed,
            'results': results,
            'output_path': output_path
        }
        
        logger.info(f"✅ Batch inference completed!")
        logger.info(f"   - Successful: {successful}")
        logger.info(f"   - Failed: {failed}")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Batch inference failed: {str(e)}")
        raise


@celery_app.task(
    bind=True,
    base=InferenceTask,
    name='backend.workers.inference_worker.inference_job',
    max_retries=3
)
def inference_job(
    self,
    model_id: str,
    dataset_path: str,
    output_dir: str,
    db=None
):
    """
    Run inference on an entire dataset
    
    Args:
        model_id: ID of the model to use
        dataset_path: Path to dataset directory
        output_dir: Directory to save results
        db: Database session (injected by InferenceTask)
    
    Returns:
        Dictionary with job results
    """
    try:
        logger.info(f"📌 Starting inference job")
        logger.info(f"   - Model ID: {model_id}")
        logger.info(f"   - Dataset: {dataset_path}")
        
        # Get model
        model = db.query(Model).filter(Model.id == model_id).first()
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Find all images in dataset
        dataset_path = Path(dataset_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        image_paths = []
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_paths.append(os.path.join(root, file))
        
        logger.info(f"📊 Found {len(image_paths)} images")
        
        if not image_paths:
            raise ValueError(f"No images found in {dataset_path}")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load inference service
        logger.info("🔧 Loading inference service...")
        service = InferenceService(model.model_path)
        
        # Process all images
        results = []
        class_counts = {}
        
        for idx, image_path in enumerate(image_paths):
            try:
                # Update progress
                progress = (idx + 1) / len(image_paths) * 100
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'current': idx + 1,
                        'total': len(image_paths),
                        'progress': progress,
                        'status': f'Processing {Path(image_path).name}'
                    }
                )
                
                # Run prediction
                result = service.predict(image_path, top_k=3)
                predicted_class = result['predicted_class']
                
                # Track class counts
                class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
                
                # Store result
                results.append({
                    'image': str(Path(image_path).relative_to(dataset_path)),
                    'predicted_class': predicted_class,
                    'confidence': result['confidence'],
                    'top_predictions': result['predictions']
                })
                
            except Exception as e:
                logger.error(f"❌ Failed to process {image_path}: {str(e)}")
                results.append({
                    'image': str(Path(image_path).relative_to(dataset_path)),
                    'error': str(e)
                })
        
        # Save results
        results_file = output_dir / f"inference_results_{model_id}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'model_id': model_id,
                'model_name': model.name,
                'backbone': model.backbone,
                'dataset_path': str(dataset_path),
                'total_images': len(image_paths),
                'class_distribution': class_counts,
                'results': results
            }, f, indent=2)
        
        # Generate summary report
        summary_file = output_dir / f"summary_{model_id}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Inference Job Summary\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Model: {model.name} ({model.backbone})\n")
            f.write(f"Dataset: {dataset_path}\n")
            f.write(f"Total Images: {len(image_paths)}\n\n")
            f.write(f"Class Distribution:\n")
            f.write(f"-" * 50 + "\n")
            for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(image_paths)) * 100
                f.write(f"{cls:20s}: {count:5d} ({percentage:5.1f}%)\n")
        
        logger.info(f"✅ Inference job completed!")
        logger.info(f"   - Results: {results_file}")
        logger.info(f"   - Summary: {summary_file}")
        
        return {
            'status': 'completed',
            'model_id': model_id,
            'total_images': len(image_paths),
            'class_distribution': class_counts,
            'results_file': str(results_file),
            'summary_file': str(summary_file)
        }
        
    except Exception as e:
        logger.error(f"❌ Inference job failed: {str(e)}")
        raise


@celery_app.task(name='backend.workers.inference_worker.warmup_model')
def warmup_model(model_id: str, model_path: str):
    """
    Warmup model by loading it into cache
    
    Args:
        model_id: Model identifier
        model_path: Path to model checkpoint
    
    Returns:
        Status message
    """
    try:
        logger.info(f"🔥 Warming up model {model_id}...")
        
        service = InferenceService(model_path)
        info = service.get_model_info()
        
        logger.info(f"✅ Model {model_id} warmed up successfully")
        logger.info(f"   - Backbone: {info['backbone']}")
        logger.info(f"   - Classes: {info['num_classes']}")
        logger.info(f"   - Device: {info['device']}")
        
        return {
            'status': 'warmed_up',
            'model_id': model_id,
            'info': info
        }
        
    except Exception as e:
        logger.error(f"❌ Model warmup failed: {str(e)}")
        raise