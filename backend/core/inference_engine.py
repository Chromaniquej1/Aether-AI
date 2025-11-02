"""
ModelForge-CV: Inference Service (Definitive - Matches Training)
Works with ModelManager and handles all finetuning modes
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os
from typing import Dict, List, Optional
from pathlib import Path
import logging
import timm
from peft import PeftModel

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Service for running inference on trained models
    Properly handles models created by ModelManager with timm + peft
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize inference service
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = None
        self.num_classes = None
        self.backbone = None
        self.finetuning_mode = None
        
        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self._load_model()
    
    def _load_model(self):
        """Load model from checkpoint"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")
            
            logger.info(f"Loading model from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract metadata
            self.class_names = checkpoint.get('class_names', [])
            self.num_classes = len(self.class_names) if self.class_names else checkpoint.get('num_classes', 2)
            self.backbone = checkpoint.get('backbone', 'resnet50')
            self.finetuning_mode = checkpoint.get('finetuning_mode', 'full')
            
            logger.info(f"   - Backbone: {self.backbone}")
            logger.info(f"   - Finetuning mode: {self.finetuning_mode}")
            logger.info(f"   - Classes: {self.num_classes} - {self.class_names}")
            
            # Recreate model using timm (same as training)
            logger.info("⚙️  Creating base model with timm...")
            base_model = timm.create_model(
                self.backbone,
                pretrained=False,  # Don't load pretrained weights
                num_classes=self.num_classes
            )
            
            # Load state dict
            state_dict = checkpoint.get('model_state_dict', {})
            
            # Check if it's a PEFT/LoRA model
            is_peft_model = any('lora' in key.lower() or 'base_model' in key for key in state_dict.keys())
            
            if is_peft_model and self.finetuning_mode == 'lora':
                logger.info("🔀 Detected LoRA model, loading with PEFT...")
                
                # For LoRA models, we need to:
                # 1. Create base model
                # 2. Apply PEFT config
                # 3. Load merged weights OR merge after loading
                
                try:
                    # Try to load the full PEFT model
                    # The state dict has keys like: base_model.model.layer.weight
                    base_model.load_state_dict(state_dict, strict=False)
                    
                    # Extract the actual model if it's wrapped
                    if hasattr(base_model, 'base_model'):
                        if hasattr(base_model.base_model, 'model'):
                            model = base_model.base_model.model
                        else:
                            model = base_model.base_model
                    else:
                        model = base_model
                    
                    logger.info("✅ LoRA model loaded")
                    
                except Exception as e:
                    logger.warning(f"Direct LoRA loading failed: {e}")
                    logger.info("Attempting to reconstruct and merge LoRA weights...")
                    
                    # Alternative: Recreate PEFT model and load
                    from peft import LoraConfig, get_peft_model
                    
                    lora_config = LoraConfig(
                        r=checkpoint.get('lora_r', 8),
                        lora_alpha=checkpoint.get('lora_alpha', 16),
                        lora_dropout=0.1,
                        target_modules=self._get_lora_targets(self.backbone),
                        bias="none",
                    )
                    
                    # Apply PEFT
                    peft_model = get_peft_model(base_model, lora_config)
                    
                    # Load state dict
                    peft_model.load_state_dict(state_dict, strict=False)
                    
                    # Merge LoRA weights into base model for faster inference
                    model = peft_model.merge_and_unload()
                    
                    logger.info("✅ LoRA weights merged successfully")
                
                self.model = model
                
            else:
                # For full finetuning or linear probe
                logger.info(f"📥 Loading {self.finetuning_mode} model...")
                
                # Clean up state dict keys if necessary
                new_state_dict = {}
                for key, value in state_dict.items():
                    # Remove any wrapper prefixes
                    new_key = key
                    if key.startswith('module.'):
                        new_key = key.replace('module.', '')
                    if key.startswith('base_model.model.'):
                        new_key = key.replace('base_model.model.', '')
                    
                    # Skip LoRA-specific keys if they exist
                    if 'lora_' not in new_key and 'base_layer' not in new_key:
                        new_state_dict[new_key] = value
                
                # Load cleaned state dict
                missing_keys, unexpected_keys = base_model.load_state_dict(new_state_dict, strict=False)
                
                if missing_keys:
                    logger.warning(f"Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
                
                self.model = base_model
                logger.info("✅ Model loaded successfully")
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ Model ready for inference on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _get_lora_targets(self, backbone: str) -> List[str]:
        """Get LoRA target modules based on backbone"""
        if 'vit' in backbone.lower():
            return ["qkv", "proj"]
        elif 'swin' in backbone.lower():
            return ["qkv", "proj"]
        elif 'convnext' in backbone.lower():
            return ["dwconv", "pwconv1", "pwconv2"]
        elif 'efficientnet' in backbone.lower():
            return ["conv_dw", "conv_pw", "project_conv"]
        elif 'resnet' in backbone.lower():
            return ["conv1", "conv2", "conv3"]
        else:
            return ["attn", "mlp"]
    
    def preprocess_image(self, image_input) -> torch.Tensor:
        """
        Preprocess image for inference
        
        Args:
            image_input: Can be PIL Image, bytes, or file path
        
        Returns:
            Preprocessed tensor ready for model
        """
        # Handle different input types
        if isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input)).convert('RGB')
        elif isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        # Apply transforms and add batch dimension
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict(self, image_input, top_k: int = None) -> Dict:
        """
        Run inference on a single image
        
        Args:
            image_input: Image to classify (PIL Image, bytes, or file path)
            top_k: Return top K predictions (default: all classes)
        
        Returns:
            Dictionary containing predictions and metadata
        """
        try:
            # Preprocess
            img_tensor = self.preprocess_image(image_input)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get predictions
            probs = probabilities[0].cpu().numpy()
            
            # Create predictions dictionary
            predictions = {
                self.class_names[i]: float(probs[i])
                for i in range(len(self.class_names))
            }
            
            # Sort by confidence
            sorted_predictions = dict(
                sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            )
            
            # Get top prediction
            predicted_idx = probs.argmax()
            predicted_class = self.class_names[predicted_idx]
            confidence = float(probs[predicted_idx])
            
            # Limit to top_k if specified
            if top_k:
                sorted_predictions = dict(list(sorted_predictions.items())[:top_k])
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'predictions': sorted_predictions,
                'all_predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {str(e)}")
            raise
    
    def predict_batch(self, image_inputs: List, batch_size: int = 32) -> List[Dict]:
        """
        Run inference on multiple images
        
        Args:
            image_inputs: List of images (PIL Images, bytes, or file paths)
            batch_size: Batch size for processing
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(image_inputs), batch_size):
            batch = image_inputs[i:i + batch_size]
            
            for img in batch:
                try:
                    result = self.predict(img)
                    results.append({
                        'success': True,
                        **result
                    })
                except Exception as e:
                    results.append({
                        'success': False,
                        'error': str(e)
                    })
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get model metadata
        
        Returns:
            Dictionary with model information
        """
        return {
            'backbone': self.backbone,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'finetuning_mode': self.finetuning_mode,
            'device': self.device,
            'model_path': self.model_path
        }


class InferenceServiceManager:
    """
    Manager for handling multiple inference services (model caching)
    """
    
    def __init__(self, cache_size: int = 3):
        """
        Initialize inference service manager
        
        Args:
            cache_size: Maximum number of models to keep in memory
        """
        self.cache_size = cache_size
        self.services: Dict[str, InferenceService] = {}
        self.access_order: List[str] = []
    
    def get_service(self, model_id: str, model_path: str) -> InferenceService:
        """
        Get or create inference service for a model
        
        Args:
            model_id: Unique identifier for the model
            model_path: Path to model checkpoint
        
        Returns:
            InferenceService instance
        """
        # Return cached service if available
        if model_id in self.services:
            # Update access order (LRU)
            self.access_order.remove(model_id)
            self.access_order.append(model_id)
            return self.services[model_id]
        
        # Create new service
        logger.info(f"Creating new inference service for model {model_id}")
        service = InferenceService(model_path)
        
        # Add to cache
        self.services[model_id] = service
        self.access_order.append(model_id)
        
        # Evict oldest if cache is full
        if len(self.services) > self.cache_size:
            oldest = self.access_order.pop(0)
            logger.info(f"Evicting model {oldest} from cache")
            del self.services[oldest]
        
        return service
    
    def clear_cache(self):
        """Clear all cached models"""
        self.services.clear()
        self.access_order.clear()
        logger.info("Inference service cache cleared")
    
    def get_cache_info(self) -> Dict:
        """Get cache statistics"""
        return {
            'cached_models': len(self.services),
            'cache_size': self.cache_size,
            'model_ids': list(self.services.keys())
        }


# Global inference service manager
inference_manager = InferenceServiceManager(cache_size=3)


def get_inference_service(model_id: str, model_path: str) -> InferenceService:
    """
    Helper function to get inference service
    
    Args:
        model_id: Model identifier
        model_path: Path to model checkpoint
    
    Returns:
        InferenceService instance
    """
    return inference_manager.get_service(model_id, model_path)