"""
ModelForge-CV: Complete End-to-End Training Pipeline
Integrates dataset processing, model creation, and training
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import platform 
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from backend.core.model_manager import ModelManager, ModelConfig, ModelBackbone, FineTuningMode
from backend.core.training_engine import TrainingEngine, TrainingConfig
from backend.core.dataset_processor import ImageDatasetProcessor


# Use 0 workers on macOS to avoid multiprocessing issues
num_workers = 0 if platform.system() == 'Darwin' else 4


class ImageClassificationDataset(Dataset):
    """Custom dataset for image classification"""
    
    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        transform: Optional[transforms.Compose] = None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_data_transforms(
    target_size: int = 224,
    augment: bool = True
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Create training and validation transforms
    
    Args:
        target_size: Target image size
        augment: Whether to apply augmentation for training
    
    Returns:
        train_transform, val_transform
    """
    
    # ImageNet normalization stats (standard for pretrained models)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Training transforms with augmentation
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            normalize
        ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform


def prepare_dataloaders(
    dataset_path: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    target_size: int = 224,
    num_workers: int = 0,
    augment: bool = True
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Prepare training and validation dataloaders
    
    Args:
        dataset_path: Path to dataset directory
        batch_size: Batch size for training
        val_split: Validation split ratio
        target_size: Target image size
        num_workers: Number of data loading workers
        augment: Whether to apply augmentation
    
    Returns:
        train_loader, val_loader, dataset_info
    """
    
    dataset_path = Path(dataset_path)
    
    # Load dataset metadata
    metadata_path = dataset_path / 'dataset_metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"📂 Loading dataset: {metadata['name']}")
    print(f"   Classes: {metadata['num_classes']}")
    print(f"   Total images: {metadata['total_images']}")
    
    # Collect all image paths and labels
    class_to_idx = {name: idx for idx, name in enumerate(metadata['class_names'])}
    
    all_image_paths = []
    all_labels = []
    
    for class_name in metadata['class_names']:
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            continue
        
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                all_image_paths.append(img_path)
                all_labels.append(class_to_idx[class_name])
    
    print(f"   Loaded {len(all_image_paths)} images")
    
    # Create transforms
    train_transform, val_transform = create_data_transforms(
        target_size=target_size,
        augment=augment
    )
    
    # Create full dataset
    full_dataset = ImageClassificationDataset(
        image_paths=all_image_paths,
        labels=all_labels,
        transform=train_transform
    )
    
    # Split into train and validation
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    print(f"   Train set: {len(train_dataset)} images")
    print(f"   Val set: {len(val_dataset)} images")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    dataset_info = {
        'num_classes': metadata['num_classes'],
        'class_names': metadata['class_names'],
        'class_to_idx': class_to_idx,
        'train_size': train_size,
        'val_size': val_size
    }
    
    return train_loader, val_loader, dataset_info


class ModelForgePipeline:
    """
    Complete end-to-end pipeline for ModelForge-CV
    Integrates all components: dataset processing, model creation, training
    """
    
    def __init__(self, project_name: str, workspace_dir: str = "./workspaces"):
        self.project_name = project_name
        self.workspace_dir = Path(workspace_dir)
        self.project_dir = self.workspace_dir / project_name
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.datasets_dir = self.project_dir / "datasets"
        self.experiments_dir = self.project_dir / "experiments"
        self.models_dir = self.project_dir / "models"
        
        for dir_path in [self.datasets_dir, self.experiments_dir, self.models_dir]:
            dir_path.mkdir(exist_ok=True)
        
        print(f"✅ ModelForge Pipeline initialized")
        print(f"   Project: {project_name}")
        print(f"   Workspace: {self.project_dir}")
    
    def process_dataset(self, dataset_zip_path: str) -> Dict:
        """
        Step 1: Process and validate uploaded dataset
        
        Args:
            dataset_zip_path: Path to uploaded ZIP file
        
        Returns:
            Dataset metadata dict
        """
        
        
        print("\n" + "="*60)
        print("STEP 1: Dataset Processing")
        print("="*60)
        
        processor = ImageDatasetProcessor(workspace_path=str(self.datasets_dir))
        
        # Extract dataset
        dataset_path = processor.extract_dataset(
            dataset_zip_path,
            extract_to=str(self.datasets_dir / Path(dataset_zip_path).stem)
        )
        
        # Analyze dataset
        metadata = processor.analyze_dataset(dataset_path)
        
        if not metadata.is_valid:
            print("\n❌ Dataset validation failed:")
            for error in metadata.validation_errors:
                print(f"   - {error}")
            raise ValueError("Invalid dataset")
        
        print("\n✅ Dataset processing complete!")
        print(f"   Task: {metadata.task_type}")
        print(f"   Classes: {metadata.num_classes}")
        print(f"   Images: {metadata.total_images}")
        
        return {
            'metadata': metadata,
            'dataset_path': str(dataset_path)
        }
    
    def create_model(
        self,
        num_classes: int,
        backbone: str = "vit_base_patch16_224",
        finetuning_mode: str = "lora"
    ) -> torch.nn.Module:
        """
        Step 2: Create and configure model
        
        Args:
            num_classes: Number of classes
            backbone: Model backbone name
            finetuning_mode: Fine-tuning strategy
        
        Returns:
            Configured PyTorch model
        """
        
        
        print("\n" + "="*60)
        print("STEP 2: Model Creation")
        print("="*60)
        
        manager = ModelManager()
        
        config = ModelConfig(
            backbone=ModelBackbone(backbone),
            num_classes=num_classes,
            finetuning_mode=FineTuningMode(finetuning_mode),
            pretrained=True
        )
        
        model = manager.create_model(config)
        
        print("\n✅ Model created successfully!")
        
        return model
    
    def train_model(
        self,
        model: torch.nn.Module,
        dataset_path: str,
        experiment_name: str,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        early_stopping_patience: int = 10
    ) -> Dict:
        """
        Step 3: Train the model
        
        Args:
            model: PyTorch model to train
            dataset_path: Path to processed dataset
            experiment_name: Name for this training run
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            early_stopping_patience: Patience for early stopping
        
        Returns:
            Training summary dict
        """
        
        
        print("\n" + "="*60)
        print("STEP 3: Model Training")
        print("="*60)
        
        # Prepare dataloaders
        train_loader, val_loader, dataset_info = prepare_dataloaders(
            dataset_path=dataset_path,
            batch_size=batch_size,
            val_split=0.2,
            target_size=224,
            num_workers=4,
            augment=True
        )
        
        # Training configuration
        config = TrainingConfig(
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            use_amp=True,
            optimizer='adamw',
            scheduler='cosine'
        )
        
        # Create training engine
        experiment_dir = self.experiments_dir / experiment_name
        engine = TrainingEngine(
            model=model,
            config=config,
            save_dir=str(experiment_dir)
        )
        
        # Train
        summary = engine.train(train_loader, val_loader)
        
        # Save final model to models directory
        final_model_path = self.models_dir / f"{experiment_name}_final.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_names': dataset_info['class_names'],
            'num_classes': dataset_info['num_classes'],
            'summary': summary
        }, final_model_path)
        
        print(f"\n💾 Final model saved to: {final_model_path}")
        
        return {
            'summary': summary,
            'model_path': str(final_model_path),
            'experiment_dir': str(experiment_dir)
        }
    
    def run_complete_pipeline(
        self,
        dataset_zip_path: str,
        backbone: str = "vit_base_patch16_224",
        finetuning_mode: str = "lora",
        experiment_name: str = "exp_001",
        **training_kwargs
    ) -> Dict:
        """
        Run the complete end-to-end pipeline
        
        Args:
            dataset_zip_path: Path to dataset ZIP
            backbone: Model backbone
            finetuning_mode: Fine-tuning strategy
            experiment_name: Name for this experiment
            **training_kwargs: Additional training arguments
        
        Returns:
            Complete pipeline results
        """
        
        print("\n" + "🚀"*30)
        print("MODELFORGE-CV: COMPLETE PIPELINE")
        print("🚀"*30 + "\n")
        
        # Step 1: Process dataset
        dataset_result = self.process_dataset(dataset_zip_path)
        metadata = dataset_result['metadata']
        dataset_path = dataset_result['dataset_path']
        
        # Step 2: Create model
        model = self.create_model(
            num_classes=metadata.num_classes,
            backbone=backbone,
            finetuning_mode=finetuning_mode
        )
        
        # Step 3: Train model
        training_result = self.train_model(
            model=model,
            dataset_path=dataset_path,
            experiment_name=experiment_name,
            **training_kwargs
        )
        
        # Compile results
        results = {
            'project_name': self.project_name,
            'dataset_metadata': metadata.to_dict(),
            'model_config': {
                'backbone': backbone,
                'finetuning_mode': finetuning_mode,
                'num_classes': metadata.num_classes
            },
            'training_summary': training_result['summary'],
            'model_path': training_result['model_path'],
            'experiment_dir': training_result['experiment_dir']
        }
        
        # Save pipeline results
        results_path = self.project_dir / f"{experiment_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\n" + "🎉"*30)
        print("PIPELINE COMPLETE!")
        print("🎉"*30)
        print(f"\n📊 Results Summary:")
        print(f"   Best Val Accuracy: {results['training_summary']['best_val_acc']:.4f}")
        print(f"   Total Epochs: {results['training_summary']['total_epochs']}")
        print(f"   Model saved: {results['model_path']}")
        print(f"   Results saved: {results_path}")
        
        return results


# Example usage and demo
if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║              🧠 MODELFORGE-CV PIPELINE 🧠                  ║
    ║                                                            ║
    ║         End-to-End AutoML for Computer Vision             ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize pipeline
    pipeline = ModelForgePipeline(
        project_name="my_image_classifier",
        workspace_dir="./workspaces"
    )
    
    # Example: Run complete pipeline
    # Uncomment to run with your dataset
    """
    results = pipeline.run_complete_pipeline(
        dataset_zip_path="./datasets/my_dataset.zip",
        backbone="vit_base_patch16_224",
        finetuning_mode="lora",
        experiment_name="exp_001",
        epochs=30,
        batch_size=32,
        learning_rate=1e-4,
        early_stopping_patience=10
    )
    """
    
    # Or run step by step for more control:
    print("\n📝 Example: Step-by-Step Pipeline\n")
    
    print("Step 1: Process Dataset")
    print("   pipeline.process_dataset('dataset.zip')")
    
    print("\nStep 2: Create Model")
    print("   model = pipeline.create_model(")
    print("       num_classes=10,")
    print("       backbone='vit_base_patch16_224',")
    print("       finetuning_mode='lora'")
    print("   )")
    
    print("\nStep 3: Train Model")
    print("   results = pipeline.train_model(")
    print("       model=model,")
    print("       dataset_path='./datasets/processed',")
    print("       experiment_name='exp_001',")
    print("       epochs=30,")
    print("       batch_size=32")
    print("   )")
    
    print("\n" + "="*60)
    print("💡 Pipeline Features:")
    print("="*60)
    print("✅ Automatic dataset validation")
    print("✅ Smart preprocessing based on dataset stats")
    print("✅ Multiple pretrained backbones (ViT, EfficientNet, ConvNeXt, etc.)")
    print("✅ Parameter-efficient fine-tuning with LoRA")
    print("✅ Mixed precision training (faster + less memory)")
    print("✅ Early stopping to prevent overfitting")
    print("✅ Automatic checkpointing")
    print("✅ Real-time metrics tracking")
    print("✅ Comprehensive logging")
    
    print("\n" + "="*60)
    print("🎯 Supported Model Backbones:")
    print("="*60)
    print("• Vision Transformers (ViT): vit_base, vit_large")
    print("• EfficientNet: efficientnet_b0 to efficientnet_b7")
    print("• ConvNeXt: convnext_tiny, convnext_base")
    print("• Swin Transformer: swin_tiny, swin_base")
    print("• ResNet: resnet50, resnet101")
    
    print("\n" + "="*60)
    print("🔧 Fine-Tuning Modes:")
    print("="*60)
    print("• full: Train all parameters (best for large datasets)")
    print("• lora: Low-rank adaptation (recommended - 10x faster, 90% less memory)")
    print("• linear_probe: Freeze backbone, train head only (fastest)")
    print("• gradual_unfreezing: Progressive unfreezing (advanced)")
    
    print("\n" + "="*60)
    print("📦 Ready to use! Upload your dataset and start training!")
    print("="*60)