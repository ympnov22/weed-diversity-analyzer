"""LoRA fine-tuning system for Hokkaido field adaptation."""

# import numpy as np  # Removed for minimal deployment
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import time

from .inatag_model import iNatAgModel
from ..utils.logger import LoggerMixin


class LoRATrainer(LoggerMixin):
    """LoRA fine-tuning trainer for regional adaptation."""
    
    def __init__(self, base_model: iNatAgModel, lora_config: Optional[Dict] = None):
        """Initialize LoRA trainer.
        
        Args:
            base_model: Base iNatAg model to adapt
            lora_config: LoRA configuration parameters
        """
        self.base_model = base_model
        self.lora_config = lora_config or {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'target_modules': ['qkv', 'proj']
        }
        self.peft_model = None
        self.training_history = []
        
    def setup_lora_model(self) -> bool:
        """Setup LoRA adapter on base model."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            if not self.base_model.is_loaded:
                self.logger.error("Base model must be loaded before setting up LoRA")
                return False
            
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,  # Use FEATURE_EXTRACTION instead
                r=self.lora_config['r'],
                lora_alpha=self.lora_config['lora_alpha'],
                lora_dropout=self.lora_config['lora_dropout'],
                target_modules=self.lora_config['target_modules']
            )
            
            self.peft_model = get_peft_model(self.base_model.model, lora_config)
            self.peft_model.print_trainable_parameters()
            
            self.logger.info(f"Setup LoRA adapter with rank={self.lora_config['r']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup LoRA model: {e}")
            return False
    
    def prepare_training_data(self, image_paths: List[Path], labels: List[int]) -> Tuple[Any, Any]:
        """Prepare training data for LoRA fine-tuning.
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            
        Returns:
            Tuple of (images_tensor, labels_tensor)
        """
        images = []
        valid_labels = []
        
        for img_path, label in zip(image_paths, labels):
            try:
                import cv2
                image = cv2.imread(str(img_path))
                if image is not None:
                    preprocessed = self.base_model.preprocess_image(image)
                    images.append(preprocessed)
                    valid_labels.append(label)
            except Exception as e:
                self.logger.warning(f"Failed to load image {img_path}: {e}")
        
        if not images:
            raise ValueError("No valid images found for training")
        
        import torch
        images_tensor = torch.stack([torch.from_numpy(img) for img in images]).float()
        labels_tensor = torch.tensor(valid_labels, dtype=torch.long)
        
        return images_tensor, labels_tensor
    
    def train_lora(self, 
                   train_images: Any, 
                   train_labels: Any,
                   val_images: Optional[Any] = None,
                   val_labels: Optional[Any] = None,
                   epochs: int = 10,
                   learning_rate: float = 1e-4,
                   batch_size: int = 16) -> Dict[str, Any]:
        """Train LoRA adapter.
        
        Args:
            train_images: Training images tensor
            train_labels: Training labels tensor
            val_images: Validation images tensor (optional)
            val_labels: Validation labels tensor (optional)
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            
        Returns:
            Training history dictionary
        """
        import torch
        
        if self.peft_model is None:
            if not self.setup_lora_model():
                raise RuntimeError("Failed to setup LoRA model")
        
        device = self.base_model._get_device()
        self.peft_model = self.peft_model.to(device)
        self.peft_model.train()
        
        optimizer = torch.optim.AdamW(self.peft_model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        
        train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        val_loader = None
        if val_images is not None and val_labels is not None:
            val_dataset = torch.utils.data.TensorDataset(val_images, val_labels)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizer, criterion, device
            )
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            if val_loader is not None:
                val_loss, val_acc = self._validate_epoch(val_loader, criterion, device)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
                    f"time={time.time()-epoch_start:.2f}s"
                )
            else:
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"time={time.time()-epoch_start:.2f}s"
                )
        
        self.training_history = history
        return history
    
    def _train_epoch(self, train_loader, optimizer, criterion, device) -> Tuple[float, float]:
        """Train one epoch."""
        self.peft_model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = self.peft_model(batch_images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            import torch
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader, criterion, device) -> Tuple[float, float]:
        """Validate one epoch."""
        self.peft_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        import torch
        with torch.no_grad():
            for batch_images, batch_labels in val_loader:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = self.peft_model(batch_images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = criterion(outputs, batch_labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_lora_adapter(self, save_path: Path) -> bool:
        """Save LoRA adapter weights.
        
        Args:
            save_path: Path to save adapter
            
        Returns:
            True if saved successfully
        """
        try:
            if self.peft_model is None:
                self.logger.error("No LoRA model to save")
                return False
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.peft_model.save_pretrained(str(save_path))
            
            self.logger.info(f"Saved LoRA adapter to {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save LoRA adapter: {e}")
            return False
    
    def load_lora_adapter(self, load_path: Path) -> bool:
        """Load LoRA adapter weights.
        
        Args:
            load_path: Path to load adapter from
            
        Returns:
            True if loaded successfully
        """
        try:
            if not self.setup_lora_model():
                return False
            
            from peft import PeftModel
            self.peft_model = PeftModel.from_pretrained(
                self.base_model.model, 
                str(load_path)
            )
            
            self.logger.info(f"Loaded LoRA adapter from {load_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load LoRA adapter: {e}")
            return False
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        if not self.training_history:
            return {}
        
        return {
            'epochs': len(self.training_history['train_loss']),
            'final_train_loss': self.training_history['train_loss'][-1],
            'final_train_acc': self.training_history['train_acc'][-1],
            'final_val_loss': self.training_history['val_loss'][-1] if self.training_history['val_loss'] else None,
            'final_val_acc': self.training_history['val_acc'][-1] if self.training_history['val_acc'] else None,
            'best_val_acc': max(self.training_history['val_acc']) if self.training_history['val_acc'] else None,
            'lora_config': self.lora_config
        }
