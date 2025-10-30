"""
Celery worker for dataset preprocessing
"""

from backend.workers.celery_app import celery_app
from backend.database.base import SessionLocal
from backend.models.dataset import Dataset, DatasetStatus
from backend.core.dataset_processor import ImageDatasetProcessor
import traceback
import os


@celery_app.task(bind=True, name="backend.workers.preprocessing_worker.process_dataset")
def process_dataset(self, dataset_id: str):
    """
    Background task to process uploaded dataset
    
    Args:
        dataset_id: Dataset ID
    """
    
    print(f"📦 Processing dataset: {dataset_id}")
    
    db = SessionLocal()
    try:
        # Get dataset from database
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Update status
        dataset.status = DatasetStatus.PROCESSING
        db.commit()
        
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'status': 'extracting', 'progress': 10}
        )
        
        # Initialize processor
        processor = ImageDatasetProcessor(
            workspace_path=f"./workspaces/{dataset.project_id}/datasets"
        )
        
        # Extract dataset
        print(f"📂 Extracting dataset...")
        extracted_path = processor.extract_dataset(
            zip_path=dataset.file_path,
            extract_to=None
        )
        
        self.update_state(
            state='PROGRESS',
            meta={'status': 'validating', 'progress': 40}
        )
        
        # Analyze dataset
        print(f"🔍 Analyzing dataset...")
        metadata = processor.analyze_dataset(str(extracted_path))
        
        self.update_state(
            state='PROGRESS',
            meta={'status': 'finalizing', 'progress': 80}
        )
        
        # Get file size
        file_size = os.path.getsize(dataset.file_path) if os.path.exists(dataset.file_path) else 0
        
        # Update dataset in database
        dataset.storage_path = str(extracted_path)
        dataset.file_size_bytes = file_size
        dataset.num_classes = metadata.num_classes
        dataset.total_images = metadata.total_images
        dataset.class_names = metadata.class_names
        dataset.class_distribution = metadata.class_distribution
        dataset.image_stats = metadata.image_stats
        dataset.is_valid = metadata.is_valid
        dataset.validation_errors = metadata.validation_errors
        
        # Get preprocessing config
        preprocessing_config = processor.get_preprocessing_config(metadata)
        dataset.preprocessing_config = preprocessing_config
        
        if metadata.is_valid:
            dataset.mark_as_ready()
            print(f"✅ Dataset processing completed!")
        else:
            dataset.mark_as_failed(metadata.validation_errors)
            print(f"❌ Dataset validation failed")
        
        db.commit()
        
        return {
            'status': 'completed' if metadata.is_valid else 'invalid',
            'dataset_id': dataset_id,
            'metadata': metadata.to_dict()
        }
    
    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        
        print(f"❌ Dataset processing failed: {error_msg}")
        
        dataset.mark_as_failed([error_msg])
        db.commit()
        
        raise
    
    finally:
        db.close()


@celery_app.task(name="backend.workers.preprocessing_worker.delete_dataset_files")
def delete_dataset_files(dataset_id: str):
    """Delete dataset files from storage"""
    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            return {'status': 'error', 'message': 'Dataset not found'}
        
        # Delete ZIP file
        if dataset.file_path and os.path.exists(dataset.file_path):
            os.remove(dataset.file_path)
        
        # Delete extracted directory
        if dataset.storage_path and os.path.exists(dataset.storage_path):
            import shutil
            shutil.rmtree(dataset.storage_path)
        
        return {'status': 'success', 'dataset_id': dataset_id}
    
    finally:
        db.close()