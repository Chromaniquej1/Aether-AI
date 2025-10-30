"""
ModelForge-CV: Dataset Processor
Handles image dataset ingestion, validation, and preprocessing
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import Counter
import zipfile
import shutil

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T


@dataclass
class DatasetMetadata:
    """Stores dataset statistics and metadata"""
    name: str
    task_type: str  # 'classification', 'detection', 'segmentation'
    num_classes: int
    class_names: List[str]
    class_distribution: Dict[str, int]
    total_images: int
    image_stats: Dict[str, any]
    is_valid: bool
    validation_errors: List[str]
    
    def to_dict(self):
        return asdict(self)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class DatasetValidator:
    """Validates dataset structure and content"""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    MIN_IMAGES_PER_CLASS = 5
    MAX_CLASS_IMBALANCE_RATIO = 100  # 100:1 ratio warning
    
    @staticmethod
    def validate_image(img_path: Path) -> Tuple[bool, Optional[str]]:
        """Check if image is valid and readable"""
        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                img.load()
            return True, None
        except Exception as e:
            return False, f"Corrupted image: {str(e)}"
    
    @staticmethod
    def detect_structure(dataset_path: Path) -> Tuple[str, Dict[str, List[Path]]]:
        """
        Auto-detect dataset structure:
        - Classification: train/class_name/*.jpg
        - Detection: images/ + annotations/ (COCO/YOLO format)
        - Segmentation: images/ + masks/
        """
        dataset_path = Path(dataset_path)
        
        # Check for classification structure
        subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        
        # Classification: multiple class folders with images
        if subdirs:
            class_images = {}
            for subdir in subdirs:
                images = [f for f in subdir.iterdir() 
                         if f.suffix.lower() in DatasetValidator.SUPPORTED_FORMATS]
                if images:
                    class_images[subdir.name] = images
            
            if len(class_images) >= 2:
                return 'classification', class_images
        
        # Detection: check for annotations folder
        if (dataset_path / 'annotations').exists():
            return 'detection', {}
        
        # Segmentation: check for masks folder
        if (dataset_path / 'masks').exists():
            return 'segmentation', {}
        
        return 'unknown', {}


class ImageDatasetProcessor:
    """Main processor for image datasets"""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.validator = DatasetValidator()
    
    def extract_dataset(self, zip_path: str, extract_to: Optional[str] = None) -> Path:
        """Extract uploaded ZIP dataset"""
        zip_path = Path(zip_path)
        extract_to = Path(extract_to) if extract_to else self.workspace_path / zip_path.stem
        
        print(f"📦 Extracting dataset to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        # Handle nested folders (common in ZIP uploads)
        subdirs = list(extract_to.iterdir())
        if len(subdirs) == 1 and subdirs[0].is_dir():
            actual_dataset = subdirs[0]
            temp_path = extract_to.parent / f"{extract_to.name}_temp"
            shutil.move(actual_dataset, temp_path)
            shutil.rmtree(extract_to)
            shutil.move(temp_path, extract_to)
        
        return extract_to
    
    def analyze_dataset(self, dataset_path: str) -> DatasetMetadata:
        """
        Comprehensive dataset analysis:
        - Detect structure and task type
        - Validate images
        - Compute statistics
        - Check for imbalances
        """
        dataset_path = Path(dataset_path)
        errors = []
        
        print("🔍 Analyzing dataset structure...")
        task_type, class_images = self.validator.detect_structure(dataset_path)
        
        if task_type == 'unknown':
            errors.append("Could not detect valid dataset structure")
            return DatasetMetadata(
                name=dataset_path.name,
                task_type='unknown',
                num_classes=0,
                class_names=[],
                class_distribution={},
                total_images=0,
                image_stats={},
                is_valid=False,
                validation_errors=errors
            )
        
        if task_type == 'classification':
            return self._analyze_classification_dataset(
                dataset_path, class_images, errors
            )
        
        # TODO: Add detection and segmentation analysis
        return None
    
    def _analyze_classification_dataset(
        self, 
        dataset_path: Path, 
        class_images: Dict[str, List[Path]],
        errors: List[str]
    ) -> DatasetMetadata:
        """Analyze classification dataset"""
        
        print(f"📊 Found {len(class_images)} classes")
        
        # Validate images and collect stats
        valid_class_images = {}
        image_sizes = []
        aspect_ratios = []
        
        for class_name, image_paths in class_images.items():
            valid_images = []
            
            for img_path in image_paths:
                is_valid, error = self.validator.validate_image(img_path)
                
                if is_valid:
                    valid_images.append(img_path)
                    try:
                        with Image.open(img_path) as img:
                            w, h = img.size
                            image_sizes.append((w, h))
                            aspect_ratios.append(w / h)
                    except:
                        pass
                else:
                    errors.append(f"{img_path.name}: {error}")
            
            if len(valid_images) < self.validator.MIN_IMAGES_PER_CLASS:
                errors.append(
                    f"Class '{class_name}' has only {len(valid_images)} images "
                    f"(minimum {self.validator.MIN_IMAGES_PER_CLASS} required)"
                )
            
            valid_class_images[class_name] = valid_images
        
        # Compute statistics
        class_distribution = {k: len(v) for k, v in valid_class_images.items()}
        total_images = sum(class_distribution.values())
        
        # Check class imbalance
        if class_distribution:
            max_count = max(class_distribution.values())
            min_count = min(class_distribution.values())
            if max_count / min_count > self.validator.MAX_CLASS_IMBALANCE_RATIO:
                errors.append(
                    f"Severe class imbalance detected: {max_count}:{min_count} ratio"
                )
        
        # Image statistics
        image_stats = {}
        if image_sizes:
            widths, heights = zip(*image_sizes)
            image_stats = {
                'mean_width': int(np.mean(widths)),
                'mean_height': int(np.mean(heights)),
                'median_width': int(np.median(widths)),
                'median_height': int(np.median(heights)),
                'mean_aspect_ratio': float(np.mean(aspect_ratios)),
                'size_variance': {
                    'width_std': float(np.std(widths)),
                    'height_std': float(np.std(heights))
                }
            }
        
        is_valid = len(errors) == 0 and len(valid_class_images) >= 2
        
        metadata = DatasetMetadata(
            name=dataset_path.name,
            task_type='classification',
            num_classes=len(valid_class_images),
            class_names=sorted(valid_class_images.keys()),
            class_distribution=class_distribution,
            total_images=total_images,
            image_stats=image_stats,
            is_valid=is_valid,
            validation_errors=errors
        )
        
        # Save metadata
        metadata.save(dataset_path / 'dataset_metadata.json')
        
        return metadata
    
    def get_preprocessing_config(self, metadata: DatasetMetadata) -> Dict:
        """Generate optimal preprocessing config based on dataset stats"""
        
        # Default configs
        config = {
            'target_size': 224,  # Default for most vision models
            'normalization': {
                'mean': [0.485, 0.456, 0.406],  # ImageNet stats
                'std': [0.229, 0.224, 0.225]
            },
            'augmentation': {
                'train': [
                    'random_horizontal_flip',
                    'random_rotation',
                    'color_jitter',
                    'random_crop'
                ],
                'val': []
            }
        }
        
        # Adjust based on dataset characteristics
        if metadata.image_stats:
            mean_width = metadata.image_stats.get('mean_width', 224)
            mean_height = metadata.image_stats.get('mean_height', 224)
            
            # Choose target size based on dataset resolution
            if mean_width < 150 or mean_height < 150:
                config['target_size'] = 128
            elif mean_width > 500 or mean_height > 500:
                config['target_size'] = 384
        
        # Adjust augmentation based on dataset size
        if metadata.total_images < 100:
            config['augmentation']['train'].extend([
                'random_affine',
                'gaussian_blur'
            ])
        
        return config

    
# Example usage
if __name__ == "__main__":
    processor = ImageDatasetProcessor(workspace_path="./workspaces/project_001")
    
    # Simulate dataset upload and extraction
    dataset_path = "./workspaces/project_001/dataset"
    
    # Analyze dataset
    metadata = processor.analyze_dataset(dataset_path)
    
    print("\n" + "="*60)
    print("📋 DATASET ANALYSIS RESULTS")
    print("="*60)
    print(f"Task Type: {metadata.task_type}")
    print(f"Classes: {metadata.num_classes}")
    print(f"Total Images: {metadata.total_images}")
    print(f"\nClass Distribution:")
    for cls, count in metadata.class_distribution.items():
        print(f"  {cls}: {count} images")
    
    if metadata.is_valid:
        print("\n✅ Dataset is valid and ready for training!")
        
        # Get preprocessing config
        preprocess_config = processor.get_preprocessing_config(metadata)
        print(f"\n🔧 Recommended preprocessing:")
        print(f"  Target size: {preprocess_config['target_size']}x{preprocess_config['target_size']}")
        print(f"  Augmentations: {', '.join(preprocess_config['augmentation']['train'])}")
    else:
        print("\n❌ Dataset validation failed:")
        for error in metadata.validation_errors:
            print(f"  - {error}")