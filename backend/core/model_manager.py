"""
ModelForge-CV: Model Manager
Handles pretrained model loading, configuration, and LoRA adaptation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import timm
from peft import LoraConfig, get_peft_model, TaskType


class ModelBackbone(Enum):
    """Supported pretrained backbones"""
    VIT_BASE = "vit_base_patch16_224"
    VIT_LARGE = "vit_large_patch16_224"
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B3 = "efficientnet_b3"
    CONVNEXT_TINY = "convnext_tiny"
    CONVNEXT_BASE = "convnext_base"
    RESNET50 = "resnet50"
    SWIN_TINY = "swin_tiny_patch4_window7_224"
    SWIN_BASE = "swin_base_patch4_window7_224"


class FineTuningMode(Enum):
    """Fine-tuning strategies"""
    FULL = "full"  # Train all parameters
    LORA = "lora"  # Low-Rank Adaptation
    LINEAR_PROBE = "linear_probe"  # Freeze backbone, train head only
    GRADUAL_UNFREEZING = "gradual_unfreezing"  # Progressive unfreezing


@dataclass
class ModelConfig:
    """Model configuration"""
    backbone: ModelBackbone
    num_classes: int
    pretrained: bool = True
    finetuning_mode: FineTuningMode = FineTuningMode.LORA
    
    # LoRA specific parameters
    lora_r: int = 8  # Rank
    lora_alpha: int = 16  # Scaling factor
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None
    
    # Training parameters
    dropout: float = 0.1
    freeze_bn: bool = True  # Freeze batch norm layers
    
    def __post_init__(self):
        if isinstance(self.backbone, str):
            self.backbone = ModelBackbone(self.backbone)
        if isinstance(self.finetuning_mode, str):
            self.finetuning_mode = FineTuningMode(self.finetuning_mode)


class ModelManager:
    """Manages model loading, configuration, and adaptation"""
    
    # Model metadata
    MODEL_INFO = {
        ModelBackbone.VIT_BASE: {
            'params': '86M',
            'input_size': 224,
            'speed': 'medium',
            'accuracy': 'high',
            'description': 'Vision Transformer - Good all-around performance'
        },
        ModelBackbone.VIT_LARGE: {
            'params': '304M',
            'input_size': 224,
            'speed': 'slow',
            'accuracy': 'very_high',
            'description': 'Large Vision Transformer - Best accuracy, slower'
        },
        ModelBackbone.EFFICIENTNET_B0: {
            'params': '5M',
            'input_size': 224,
            'speed': 'fast',
            'accuracy': 'good',
            'description': 'EfficientNet B0 - Fast and efficient'
        },
        ModelBackbone.EFFICIENTNET_B3: {
            'params': '12M',
            'input_size': 300,
            'speed': 'medium',
            'accuracy': 'high',
            'description': 'EfficientNet B3 - Balanced speed and accuracy'
        },
        ModelBackbone.CONVNEXT_TINY: {
            'params': '28M',
            'input_size': 224,
            'speed': 'fast',
            'accuracy': 'high',
            'description': 'ConvNeXt Tiny - Modern CNN architecture'
        },
        ModelBackbone.CONVNEXT_BASE: {
            'params': '89M',
            'input_size': 224,
            'speed': 'medium',
            'accuracy': 'very_high',
            'description': 'ConvNeXt Base - Excellent performance'
        },
        ModelBackbone.RESNET50: {
            'params': '25M',
            'input_size': 224,
            'speed': 'fast',
            'accuracy': 'good',
            'description': 'ResNet50 - Classic reliable architecture'
        },
        ModelBackbone.SWIN_TINY: {
            'params': '28M',
            'input_size': 224,
            'speed': 'medium',
            'accuracy': 'high',
            'description': 'Swin Transformer - Hierarchical vision transformer'
        },
        ModelBackbone.SWIN_BASE: {
            'params': '88M',
            'input_size': 224,
            'speed': 'slow',
            'accuracy': 'very_high',
            'description': 'Swin Base - Top-tier performance'
        }
    }
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def create_model(self, config: ModelConfig) -> nn.Module:
        """
        Create and configure model based on config
        Returns: Configured PyTorch model
        """
        print(f"🔧 Creating model: {config.backbone.value}")
        print(f"   Fine-tuning mode: {config.finetuning_mode.value}")
        
        # Load pretrained model using timm
        model = timm.create_model(
            config.backbone.value,
            pretrained=config.pretrained,
            num_classes=config.num_classes,
            drop_rate=config.dropout
        )
        
        # Apply fine-tuning strategy
        if config.finetuning_mode == FineTuningMode.FULL:
            model = self._setup_full_finetuning(model, config)
        
        elif config.finetuning_mode == FineTuningMode.LORA:
            model = self._setup_lora_finetuning(model, config)
        
        elif config.finetuning_mode == FineTuningMode.LINEAR_PROBE:
            model = self._setup_linear_probe(model, config)
        
        elif config.finetuning_mode == FineTuningMode.GRADUAL_UNFREEZING:
            model = self._setup_gradual_unfreezing(model, config)
        
        model = model.to(self.device)
        
        # Print trainable parameters
        self._print_trainable_params(model)
        
        return model
    
    def _setup_full_finetuning(self, model: nn.Module, config: ModelConfig) -> nn.Module:
        """Full fine-tuning: all parameters trainable"""
        for param in model.parameters():
            param.requires_grad = True
        
        # Optionally freeze batch norm
        if config.freeze_bn:
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
        
        return model
    
    def _setup_lora_finetuning(self, model: nn.Module, config: ModelConfig) -> nn.Module:
        """
        LoRA fine-tuning: Parameter-efficient adaptation
        Only trains low-rank matrices, massively reduces trainable params
        """
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        # Determine target modules based on architecture
        if config.lora_target_modules is None:
            target_modules = self._get_lora_target_modules(config.backbone)
        else:
            target_modules = config.lora_target_modules
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            #task_type=TaskType.IMAGE_CLASSIFICATION
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        return model
    
    def _setup_linear_probe(self, model: nn.Module, config: ModelConfig) -> nn.Module:
        """
        Linear probe: Freeze backbone, only train classification head
        Fastest training, good for quick experiments
        """
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier head
        # Different architectures have different classifier names
        classifier_names = ['head', 'fc', 'classifier']
        
        for name in classifier_names:
            if hasattr(model, name):
                classifier = getattr(model, name)
                for param in classifier.parameters():
                    param.requires_grad = True
                break
        
        return model
    
    def _setup_gradual_unfreezing(self, model: nn.Module, config: ModelConfig) -> nn.Module:
        """
        Gradual unfreezing: Start with frozen backbone, unfreeze progressively
        This is set up for the training loop to handle
        """
        # Initially freeze all except head (like linear probe)
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier
        classifier_names = ['head', 'fc', 'classifier']
        for name in classifier_names:
            if hasattr(model, name):
                classifier = getattr(model, name)
                for param in classifier.parameters():
                    param.requires_grad = True
                break
        
        # Training loop will progressively unfreeze layers
        return model
    
    def _get_lora_target_modules(self, backbone: ModelBackbone) -> List[str]:
        """Get appropriate LoRA target modules for each architecture"""
        
        # Vision Transformers - target attention projections
        if backbone in [ModelBackbone.VIT_BASE, ModelBackbone.VIT_LARGE]:
            return ["qkv", "proj"]
        
        # Swin Transformers
        elif backbone in [ModelBackbone.SWIN_TINY, ModelBackbone.SWIN_BASE]:
            return ["qkv", "proj"]
        
        # ConvNext - target depthwise and pointwise convs
        elif backbone in [ModelBackbone.CONVNEXT_TINY, ModelBackbone.CONVNEXT_BASE]:
            return ["dwconv", "pwconv1", "pwconv2"]
        
        # EfficientNet - target depthwise and project convs
        elif backbone in [ModelBackbone.EFFICIENTNET_B0, ModelBackbone.EFFICIENTNET_B3]:
            return ["conv_dw", "conv_pw", "project_conv"]
        
        # ResNet - target conv layers
        elif backbone == ModelBackbone.RESNET50:
            return ["conv1", "conv2", "conv3"]
        
        # Default fallback
        return ["attn", "mlp"]
    
    def _print_trainable_params(self, model: nn.Module):
        """Print trainable parameter statistics"""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_pct = 100 * trainable_params / total_params if total_params > 0 else 0
        
        print(f"\n📊 Parameter Statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Trainable percentage: {trainable_pct:.2f}%")
    
    def get_model_recommendations(
        self, 
        dataset_size: int, 
        num_classes: int,
        time_constraint: str = 'medium'  # 'fast', 'medium', 'slow'
    ) -> List[Tuple[ModelBackbone, FineTuningMode]]:
        """
        Recommend model and fine-tuning strategy based on dataset characteristics
        """
        recommendations = []
        
        # Small dataset (< 500 images)
        if dataset_size < 500:
            recommendations.append((
                ModelBackbone.EFFICIENTNET_B0,
                FineTuningMode.LORA,
                "Lightweight model with efficient fine-tuning"
            ))
            recommendations.append((
                ModelBackbone.RESNET50,
                FineTuningMode.LINEAR_PROBE,
                "Fast training with frozen backbone"
            ))
        
        # Medium dataset (500-5000 images)
        elif dataset_size < 5000:
            if time_constraint == 'fast':
                recommendations.append((
                    ModelBackbone.EFFICIENTNET_B3,
                    FineTuningMode.LORA,
                    "Good balance of speed and accuracy"
                ))
            else:
                recommendations.append((
                    ModelBackbone.VIT_BASE,
                    FineTuningMode.LORA,
                    "High accuracy with parameter efficiency"
                ))
                recommendations.append((
                    ModelBackbone.CONVNEXT_BASE,
                    FineTuningMode.LORA,
                    "Excellent modern architecture"
                ))
        
        # Large dataset (> 5000 images)
        else:
            if time_constraint == 'fast':
                recommendations.append((
                    ModelBackbone.CONVNEXT_TINY,
                    FineTuningMode.FULL,
                    "Fast training with full fine-tuning"
                ))
            elif time_constraint == 'medium':
                recommendations.append((
                    ModelBackbone.VIT_BASE,
                    FineTuningMode.FULL,
                    "Strong all-around performance"
                ))
            else:
                recommendations.append((
                    ModelBackbone.VIT_LARGE,
                    FineTuningMode.LORA,
                    "Best possible accuracy"
                ))
                recommendations.append((
                    ModelBackbone.SWIN_BASE,
                    FineTuningMode.LORA,
                    "State-of-the-art hierarchical transformer"
                ))
        
        return recommendations


# Example usage
if __name__ == "__main__":
    manager = ModelManager()
    
    # Create a model configuration
    config = ModelConfig(
        backbone=ModelBackbone.VIT_BASE,
        num_classes=10,
        finetuning_mode=FineTuningMode.LORA,
        lora_r=8,
        lora_alpha=16
    )
    
    # Create the model
    model = manager.create_model(config)
    
    # Get recommendations
    print("\n" + "="*60)
    print("💡 MODEL RECOMMENDATIONS")
    print("="*60)
    
    recs = manager.get_model_recommendations(
        dataset_size=1000,
        num_classes=10,
        time_constraint='medium'
    )
    
    for i, (backbone, mode, reason) in enumerate(recs, 1):
        info = manager.MODEL_INFO[backbone]
        print(f"\n{i}. {backbone.value}")
        print(f"   Mode: {mode.value}")
        print(f"   Params: {info['params']}")
        print(f"   Speed: {info['speed']}")
        print(f"   Reason: {reason}")