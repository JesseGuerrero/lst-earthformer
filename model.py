import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
import numpy as np
from typing import Dict, Any, Tuple
import os
from pathlib import Path

class LandsatLSTPredictor(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        max_epochs: int = 100,
        **model_kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Default Landsat-optimized config
        self.model_config = {
            'input_shape': (3, 128, 128, 9),  # 3 timesteps, 128x128, 9 bands
            'target_shape': (3, 128, 128, 1), # 3 timesteps, LST only
            'base_units': 96,
            'num_heads': 6,
            'enc_depth': [2, 2],
            'dec_depth': [1, 1],
            'attn_drop': 0.1,
            'proj_drop': 0.1,
            'ffn_drop': 0.1,
            'num_global_vectors': 8,
            'use_dec_self_global': True,
            'use_dec_cross_global': True,
            'pos_embed_type': 't+hw',
            'use_relative_pos': True,
            'ffn_activation': 'gelu',
            'enc_cuboid_size': [(2, 4, 4), (2, 4, 4)],
            'enc_cuboid_strategy': [('l', 'l', 'l'), ('d', 'd', 'd')],
            'dec_cross_cuboid_hw': [(4, 4), (4, 4)],
            'dec_cross_n_temporal': [1, 2],
        }
        
        # Update with any provided kwargs
        self.model_config.update(model_kwargs)
        
        # Initialize model
        self.model = CuboidTransformerModel(**self.model_config)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        return self.model(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        inputs, targets = batch
        
        # Forward pass
        predictions = self.forward(inputs)
        
        # Calculate loss
        loss = self.criterion(predictions, targets)
        
        # Calculate metrics
        mae = F.l1_loss(predictions, targets)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True)
        
        # Calculate temperature-specific metrics (LST in Fahrenheit)
        with torch.no_grad():
            # Convert from scaled values back to Fahrenheit for interpretability
            pred_temp = predictions.detach()
            true_temp = targets.detach()
            
            temp_mae = F.l1_loss(pred_temp, true_temp)
            temp_rmse = torch.sqrt(F.mse_loss(pred_temp, true_temp))
            
            self.log('train_temp_mae_scaled', temp_mae, on_step=False, on_epoch=True)
            self.log('train_temp_rmse_scaled', temp_rmse, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        inputs, targets = batch
        
        # Forward pass
        predictions = self.forward(inputs)
        
        # Calculate loss
        loss = self.criterion(predictions, targets)
        
        # Calculate metrics
        mae = F.l1_loss(predictions, targets)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True)
        
        # Temperature-specific metrics
        with torch.no_grad():
            pred_temp = predictions.detach()
            true_temp = targets.detach()
            
            temp_mae = F.l1_loss(pred_temp, true_temp)
            temp_rmse = torch.sqrt(F.mse_loss(pred_temp, true_temp))
            
            # Calculate correlation coefficient
            pred_flat = pred_temp.flatten()
            true_flat = true_temp.flatten()
            
            # Remove any NaN or extreme values
            mask = torch.isfinite(pred_flat) & torch.isfinite(true_flat)
            if mask.sum() > 0:
                pred_clean = pred_flat[mask]
                true_clean = true_flat[mask]
                
                correlation = torch.corrcoef(torch.stack([pred_clean, true_clean]))[0, 1]
                if torch.isfinite(correlation):
                    self.log('val_correlation', correlation, on_step=False, on_epoch=True)
            
            self.log('val_temp_mae_scaled', temp_mae, on_step=False, on_epoch=True)
            self.log('val_temp_rmse_scaled', temp_rmse, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step"""
        inputs, targets = batch
        predictions = self.forward(inputs)
        loss = self.criterion(predictions, targets)
        
        # Calculate comprehensive test metrics
        mae = F.l1_loss(predictions, targets)
        rmse = torch.sqrt(F.mse_loss(predictions, targets))
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_mae', mae, on_step=False, on_epoch=True)
        self.log('test_rmse', rmse, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers - simplified without scheduler for now"""
        # Just return the optimizer without any scheduler
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999)
        )
        
        return optimizer
        
        # Alternative: If you want to try a scheduler later, uncomment this:
        """
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    mode='min',
                    factor=0.5,
                    patience=5,
                    verbose=True
                ),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }
        """