import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
import numpy as np
from typing import Dict, Any, Tuple, Optional
import os
from pathlib import Path
import wandb
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle

class LandsatLSTPredictor(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        max_epochs: int = 100,
        log_images_every_n_epochs: int = 5,
        max_images_to_log: int = 4,
        **model_kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Image logging parameters
        self.log_images_every_n_epochs = log_images_every_n_epochs
        self.max_images_to_log = max_images_to_log
        
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
        
        # Band names for visualization (updated with proper scaling info)
        self.band_names = ['DEM (+10k offset)', 'LST (°F)', 'Red (×10k)', 'Green (×10k)', 'Blue (×10k)', 
                          'NDVI (×10k)', 'NDWI (×10k)', 'NDBI (×10k)', 'Albedo (×10k)']
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        return self.model(x)
    
    def create_landsat_visualization(self, inputs: torch.Tensor, targets: torch.Tensor, 
                                   predictions: Optional[torch.Tensor] = None, 
                                   batch_idx: int = 0, max_samples: int = 2) -> plt.Figure:
        """Create comprehensive visualization of Landsat data with proper int16 scaling"""
        # Move to CPU and convert to numpy
        inputs_np = inputs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        if predictions is not None:
            predictions_np = predictions.detach().cpu().numpy()
        
        batch_size = min(inputs_np.shape[0], max_samples)
        
        # Create figure
        n_cols = 4 if predictions is not None else 3
        fig, axes = plt.subplots(batch_size, n_cols, figsize=(4*n_cols, 4*batch_size))
        
        if batch_size == 1:
            axes = axes.reshape(1, -1)
        
        for sample_idx in range(batch_size):
            # Get middle timestep for visualization (t=1 out of 0,1,2)
            middle_t = 1
            
            # Extract data for this sample and timestep
            sample_input = inputs_np[sample_idx, middle_t]  # (H, W, C)
            sample_target = targets_np[sample_idx, middle_t, :, :, 0]  # (H, W)
            
            # Unscale RGB bands (indices 2, 3, 4) from int16 to reflectance [0, 1]
            red_scaled = sample_input[:, :, 2] / 10000.0    # Red band
            green_scaled = sample_input[:, :, 3] / 10000.0  # Green band  
            blue_scaled = sample_input[:, :, 4] / 10000.0   # Blue band
            
            # Create RGB composite and clip to valid range
            rgb = np.stack([red_scaled, green_scaled, blue_scaled], axis=-1)
            rgb = np.clip(rgb, 0, 1)  # Ensure valid reflectance range
            
            # Handle nodata values (set to black)
            nodata_mask = (sample_input[:, :, 2] == 0) | (sample_input[:, :, 3] == 0) | (sample_input[:, :, 4] == 0)
            rgb[nodata_mask] = 0
            
            # NDVI visualization - unscale from int16 to [-1, 1]
            ndvi_scaled = sample_input[:, :, 5] / 10000.0  # NDVI band
            ndvi_scaled = np.clip(ndvi_scaled, -1, 1)
            ndvi_masked = np.where(sample_input[:, :, 5] == 0, np.nan, ndvi_scaled)
            
            # LST target (already in Fahrenheit)
            lst_target = sample_target
            lst_target_masked = np.where(lst_target == 0, np.nan, lst_target)
            
            # Plot RGB composite
            axes[sample_idx, 0].imshow(rgb)
            axes[sample_idx, 0].set_title(f'Sample {sample_idx}: RGB Composite\n(t={middle_t}, reflectance)')
            axes[sample_idx, 0].axis('off')
            
            # Plot NDVI with proper scaling
            ndvi_im = axes[sample_idx, 1].imshow(ndvi_masked, cmap='RdYlGn', vmin=-1, vmax=1)
            axes[sample_idx, 1].set_title(f'NDVI\n(t={middle_t}, -1 to 1)')
            axes[sample_idx, 1].axis('off')
            cbar1 = plt.colorbar(ndvi_im, ax=axes[sample_idx, 1], fraction=0.046, pad=0.04)
            cbar1.set_label('NDVI Index')
            
            # Plot LST target with Fahrenheit scale
            temp_min, temp_max = np.nanpercentile(lst_target_masked, [5, 95]) if not np.all(np.isnan(lst_target_masked)) else (32, 100)
            lst_im = axes[sample_idx, 2].imshow(lst_target_masked, cmap='coolwarm', vmin=temp_min, vmax=temp_max)
            axes[sample_idx, 2].set_title(f'LST Target\n(t={middle_t}, °F)')
            axes[sample_idx, 2].axis('off')
            cbar2 = plt.colorbar(lst_im, ax=axes[sample_idx, 2], fraction=0.046, pad=0.04)
            cbar2.set_label('Temperature (°F)')
            
            # Plot prediction if available
            if predictions is not None:
                sample_pred = predictions_np[sample_idx, middle_t, :, :, 0]  # (H, W)
                sample_pred_masked = np.where(sample_pred == 0, np.nan, sample_pred)
                pred_im = axes[sample_idx, 3].imshow(sample_pred_masked, cmap='coolwarm', vmin=temp_min, vmax=temp_max)
                axes[sample_idx, 3].set_title(f'LST Prediction\n(t={middle_t}, °F)')
                axes[sample_idx, 3].axis('off')
                cbar3 = plt.colorbar(pred_im, ax=axes[sample_idx, 3], fraction=0.046, pad=0.04)
                cbar3.set_label('Temperature (°F)')
        
        plt.tight_layout()
        return fig
    
    def create_temporal_sequence_viz(self, inputs: torch.Tensor, targets: torch.Tensor,
                                   predictions: Optional[torch.Tensor] = None,
                                   sample_idx: int = 0) -> plt.Figure:
        """Create visualization showing temporal sequence with proper scaling"""
        inputs_np = inputs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        if predictions is not None:
            predictions_np = predictions.detach().cpu().numpy()
        
        # Get single sample
        sample_input = inputs_np[sample_idx]  # (T, H, W, C)
        sample_target = targets_np[sample_idx]  # (T, H, W, 1)
        
        n_timesteps = sample_input.shape[0]
        n_cols = 3 if predictions is not None else 2
        
        fig, axes = plt.subplots(n_timesteps, n_cols, figsize=(4*n_cols, 3*n_timesteps))
        
        if n_timesteps == 1:
            axes = axes.reshape(1, -1)
        
        # Calculate global temperature range for consistent scaling
        all_targets = targets_np[sample_idx].flatten()
        all_targets_clean = all_targets[all_targets != 0]
        if len(all_targets_clean) > 0:
            global_temp_min, global_temp_max = np.percentile(all_targets_clean, [5, 95])
        else:
            global_temp_min, global_temp_max = 32, 100
        
        for t in range(n_timesteps):
            # Input LST (band index 1) - already in Fahrenheit
            input_lst = sample_input[t, :, :, 1]
            target_lst = sample_target[t, :, :, 0]
            
            # Mask nodata values
            input_lst_masked = np.where(input_lst == 0, np.nan, input_lst)
            target_lst_masked = np.where(target_lst == 0, np.nan, target_lst)
            
            # Plot input LST
            im1 = axes[t, 0].imshow(input_lst_masked, cmap='coolwarm', 
                                   vmin=global_temp_min, vmax=global_temp_max)
            axes[t, 0].set_title(f't={t}: Input LST (°F)')
            axes[t, 0].axis('off')
            cbar1 = plt.colorbar(im1, ax=axes[t, 0], fraction=0.046, pad=0.04)
            cbar1.set_label('Temperature (°F)')
            
            # Plot target LST
            im2 = axes[t, 1].imshow(target_lst_masked, cmap='coolwarm',
                                   vmin=global_temp_min, vmax=global_temp_max)
            axes[t, 1].set_title(f't={t}: Target LST (°F)')
            axes[t, 1].axis('off')
            cbar2 = plt.colorbar(im2, ax=axes[t, 1], fraction=0.046, pad=0.04)
            cbar2.set_label('Temperature (°F)')
            
            # Plot predicted LST if available
            if predictions is not None:
                pred_lst = predictions_np[sample_idx, t, :, :, 0]
                pred_lst_masked = np.where(pred_lst == 0, np.nan, pred_lst)
                im3 = axes[t, 2].imshow(pred_lst_masked, cmap='coolwarm',
                                       vmin=global_temp_min, vmax=global_temp_max)
                axes[t, 2].set_title(f't={t}: Predicted LST (°F)')
                axes[t, 2].axis('off')
                cbar3 = plt.colorbar(im3, ax=axes[t, 2], fraction=0.046, pad=0.04)
                cbar3.set_label('Temperature (°F)')
        
        plt.tight_layout()
        return fig
    
    def create_difference_visualization(self, targets: torch.Tensor, predictions: torch.Tensor,
                                      sample_idx: int = 0) -> plt.Figure:
        """Create visualization showing prediction errors in Fahrenheit"""
        targets_np = targets.detach().cpu().numpy()
        predictions_np = predictions.detach().cpu().numpy()
        
        # Calculate difference in Fahrenheit
        diff = predictions_np - targets_np
        
        sample_target = targets_np[sample_idx]  # (T, H, W, 1)
        sample_pred = predictions_np[sample_idx]  # (T, H, W, 1)
        sample_diff = diff[sample_idx]  # (T, H, W, 1)
        
        n_timesteps = sample_target.shape[0]
        fig, axes = plt.subplots(1, n_timesteps, figsize=(4*n_timesteps, 4))
        
        if n_timesteps == 1:
            axes = [axes]
        
        for t in range(n_timesteps):
            diff_t = sample_diff[t, :, :, 0]
            
            # Mask nodata values
            target_t = sample_target[t, :, :, 0]
            pred_t = sample_pred[t, :, :, 0]
            valid_mask = (target_t != 0) & (pred_t != 0)
            diff_t_masked = np.where(valid_mask, diff_t, np.nan)
            
            # Plot difference with symmetric colormap
            if not np.all(np.isnan(diff_t_masked)):
                vmax = min(np.nanpercentile(np.abs(diff_t_masked), 95), 15)
            else:
                vmax = 10
            
            im = axes[t].imshow(diff_t_masked, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[t].set_title(f't={t}: Prediction Error\n(Pred - Target, °F)')
            axes[t].axis('off')
            cbar = plt.colorbar(im, ax=axes[t], fraction=0.046, pad=0.04)
            cbar.set_label('Temperature Error (°F)')
            
            # Add statistics
            if not np.all(np.isnan(diff_t_masked)):
                mae = np.nanmean(np.abs(diff_t_masked))
                rmse = np.sqrt(np.nanmean(diff_t_masked**2))
                axes[t].text(0.02, 0.98, f'MAE: {mae:.1f}°F\nRMSE: {rmse:.1f}°F', 
                           transform=axes[t].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def log_images_to_wandb(self, inputs: torch.Tensor, targets: torch.Tensor,
                           predictions: Optional[torch.Tensor] = None,
                           stage: str = "train", batch_idx: int = 0):
        """Log images to Weights & Biases"""
        if not isinstance(self.logger, pl.loggers.WandbLogger):
            return
        
        try:
            # Create main visualization
            fig1 = self.create_landsat_visualization(
                inputs, targets, predictions, batch_idx, self.max_images_to_log
            )
            
            # Log main visualization
            self.logger.experiment.log({
                f"{stage}_landsat_tiles": wandb.Image(fig1, caption=f"{stage} - Epoch {self.current_epoch}")
            })
            plt.close(fig1)
            
            # Create temporal sequence for first sample
            if inputs.shape[0] > 0:
                fig2 = self.create_temporal_sequence_viz(inputs, targets, predictions, sample_idx=0)
                self.logger.experiment.log({
                    f"{stage}_temporal_sequence": wandb.Image(fig2, caption=f"{stage} Temporal Sequence - Epoch {self.current_epoch}")
                })
                plt.close(fig2)
                
                # Create difference visualization if we have predictions
                if predictions is not None:
                    fig3 = self.create_difference_visualization(targets, predictions, sample_idx=0)
                    self.logger.experiment.log({
                        f"{stage}_prediction_errors": wandb.Image(fig3, caption=f"{stage} Prediction Errors - Epoch {self.current_epoch}")
                    })
                    plt.close(fig3)
        
        except Exception as e:
            print(f"Warning: Failed to log images to wandb: {e}")
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with image logging"""
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
        
        # Calculate temperature-specific metrics
        with torch.no_grad():
            pred_temp = predictions.detach()
            true_temp = targets.detach()
            
            temp_mae = F.l1_loss(pred_temp, true_temp)
            temp_rmse = torch.sqrt(F.mse_loss(pred_temp, true_temp))
            
            self.log('train_temp_mae_scaled', temp_mae, on_step=False, on_epoch=True)
            self.log('train_temp_rmse_scaled', temp_rmse, on_step=False, on_epoch=True)
        
        # Log images periodically
        if (batch_idx == 0 and 
            self.current_epoch % self.log_images_every_n_epochs == 0):
            self.log_images_to_wandb(inputs, targets, predictions, "train", batch_idx)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step with image logging"""
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
        
        # Log images for first batch every N epochs
        if (batch_idx == 0 and 
            self.current_epoch % self.log_images_every_n_epochs == 0):
            self.log_images_to_wandb(inputs, targets, predictions, "val", batch_idx)
        
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
        
        # Log test images for first few batches
        if batch_idx < 3:
            self.log_images_to_wandb(inputs, targets, predictions, "test", batch_idx)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999)
        )
        
        return optimizer