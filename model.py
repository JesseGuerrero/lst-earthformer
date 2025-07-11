import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path

class LandsatLSTPredictor(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        max_epochs: int = 100,
        log_images_every_n_epochs: int = 5,
        max_images_to_log: int = 4,
        input_sequence_length: int = 3,     # New parameter
        output_sequence_length: int = 3,    # New parameter
        **model_kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store sequence lengths
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        
        # Image logging parameters
        self.log_images_every_n_epochs = log_images_every_n_epochs
        self.max_images_to_log = max_images_to_log
        
        # Store metadata for each batch during training
        self.current_batch_metadata = {}
        
        # Default Landsat-optimized config
        self.model_config = {
            'input_shape': (input_sequence_length, 128, 128, 9),  # input_sequence timesteps, 128x128, 9 bands
            'target_shape': (output_sequence_length, 128, 128, 1), # output_sequence timesteps, LST only
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
        
        # Initialize model - Import here to avoid issues
        from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
        self.model = CuboidTransformerModel(**self.model_config)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Band names for visualization
        self.band_names = ['DEM (+10k offset)', 'LST (°F)', 'Red (×10k)', 'Green (×10k)', 'Blue (×10k)', 
                          'NDVI (×10k)', 'NDWI (×10k)', 'NDBI (×10k)', 'Albedo (×10k)']
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        return self.model(x)    

    def extract_batch_metadata(self, batch_info: Any, batch_idx: int) -> Dict[str, Any]:
        """
        Extract metadata from the current batch.
        This method should be called during the training/validation step.
        """
        # Get the dataset to extract metadata
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'train_dataset'):
            dataset = self.trainer.datamodule.train_dataset
            
            # Calculate actual sample indices from batch
            batch_size = self.trainer.datamodule.batch_size
            start_idx = batch_idx * batch_size
            
            metadata = {
                'batch_idx': batch_idx,
                'epoch': self.current_epoch,
                'batch_size': batch_size,
                'start_sample_idx': start_idx,
                'samples_metadata': []
            }
            
            # Extract metadata for each sample in the batch
            for i in range(min(batch_size, self.max_images_to_log)):
                sample_idx = start_idx + i
                if sample_idx < len(dataset):
                    # Get the tile sequence info from dataset
                    if hasattr(dataset, 'tile_sequences') and sample_idx < len(dataset.tile_sequences):
                        city, tile_row, tile_col, input_months, output_months = dataset.tile_sequences[sample_idx]
                        
                        sample_metadata = {
                            'sample_idx': sample_idx,
                            'city': city,
                            'tile_position': f"row_{tile_row:03d}_col_{tile_col:03d}",
                            'tile_row': tile_row,
                            'tile_col': tile_col,
                            'input_months': input_months,
                            'output_months': output_months,
                            'input_date_range': f"{input_months[0]} to {input_months[-1]}",
                            'output_date_range': f"{output_months[0]} to {output_months[-1]}",
                            'sequence_length': len(input_months),
                            'file_paths': self._get_file_paths(city, tile_row, tile_col, input_months + output_months)
                        }
                        metadata['samples_metadata'].append(sample_metadata)
            
            return metadata
        
        # Fallback metadata if dataset info not available
        return {
            'batch_idx': batch_idx,
            'epoch': self.current_epoch,
            'note': 'Limited metadata - dataset info not accessible'
        }
    
    def _get_file_paths(self, city: str, tile_row: int, tile_col: int, months: List[str]) -> Dict[str, List[str]]:
        """Get file paths for the tiles used in this sequence"""
        if not hasattr(self.trainer, 'datamodule'):
            return {}
            
        dataset_root = Path(self.trainer.datamodule.dataset_root)
        
        file_paths = {
            'dem_path': str(dataset_root / "DEM_2014_Tiles" / city / f"DEM_row_{tile_row:03d}_col_{tile_col:03d}.tif"),
            'monthly_scenes': {}
        }
        
        # Get paths for each month's tiles
        for month in months:
            monthly_scenes = self._get_monthly_scenes_for_city(city)
            if month in monthly_scenes:
                scene_dir = Path(monthly_scenes[month])
                month_paths = {}
                
                band_names = ['LST', 'red', 'green', 'blue', 'ndvi', 'ndwi', 'ndbi', 'albedo']
                for band in band_names:
                    tile_path = scene_dir / f"{band}_row_{tile_row:03d}_col_{tile_col:03d}.tif"
                    month_paths[band] = str(tile_path)
                
                file_paths['monthly_scenes'][month] = month_paths
        
        return file_paths
    
    def _get_monthly_scenes_for_city(self, city: str) -> Dict[str, str]:
        """Helper to get monthly scenes for a city (simplified version of dataset method)"""
        if not hasattr(self.trainer, 'datamodule'):
            return {}
            
        dataset_root = Path(self.trainer.datamodule.dataset_root)
        city_dir = dataset_root / "Cities_Tiles" / city
        
        if not city_dir.exists():
            return {}
        
        monthly_scenes = {}
        scene_dirs = [d for d in city_dir.iterdir() if d.is_dir()]
        
        for scene_dir in scene_dirs:
            try:
                from datetime import datetime
                date_str = scene_dir.name
                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                month_key = f"{date_obj.year}-{date_obj.month:02d}"
                
                if month_key not in monthly_scenes:
                    monthly_scenes[month_key] = str(scene_dir)
            except:
                continue
        
        return monthly_scenes
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with enhanced image logging"""
        inputs, targets = batch
        
        # Forward pass
        predictions = self.forward(inputs)
        
        # Calculate loss
        loss = self.criterion(predictions, targets)
        
        # Calculate metrics
        mae = torch.nn.functional.l1_loss(predictions, targets)
        
        # Log metrics
        self.log('train_loss', torch.sqrt(loss), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True)
        
        # Calculate temperature-specific metrics
        with torch.no_grad():
            pred_temp = predictions.detach()
            true_temp = targets.detach()
            
            temp_mae = torch.nn.functional.l1_loss(pred_temp, true_temp)
            temp_rmse = torch.sqrt(torch.nn.functional.mse_loss(pred_temp, true_temp))
            
            self.log('train_temp_mae_scaled', temp_mae, on_step=False, on_epoch=True)
            self.log('train_temp_rmse_scaled', temp_rmse, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step with enhanced image logging"""
        inputs, targets = batch
        
        # Forward pass
        predictions = self.forward(inputs)
        
        # Calculate loss
        loss = self.criterion(predictions, targets)
        
        # Calculate metrics
        mae = torch.nn.functional.l1_loss(predictions, targets)
        
        # Log metrics
        self.log('val_loss', torch.sqrt(loss), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True)
        
        # Temperature-specific metrics
        with torch.no_grad():
            pred_temp = predictions.detach()
            true_temp = targets.detach()
            
            temp_mae = torch.nn.functional.l1_loss(pred_temp, true_temp)
            temp_rmse = torch.sqrt(torch.nn.functional.mse_loss(pred_temp, true_temp))
            
            # Calculate correlation coefficient
            pred_flat = pred_temp.flatten()
            true_flat = true_temp.flatten()
            
            # Remove any NaN or extreme values
            mask = torch.isfinite(pred_flat) & torch.isfinite(true_flat)
            if mask.sum() > 0:
                pred_clean = pred_flat[mask]
                true_clean = true_flat[mask]
                
                if len(pred_clean) > 1:  # Need at least 2 points for correlation
                    correlation = torch.corrcoef(torch.stack([pred_clean, true_clean]))[0, 1]
                    if torch.isfinite(correlation):
                        self.log('val_correlation', correlation, on_step=False, on_epoch=True)
            
            self.log('val_temp_mae_scaled', temp_mae, on_step=False, on_epoch=True)
            self.log('val_temp_rmse_scaled', temp_rmse, on_step=False, on_epoch=True)
        
        return loss
    
    def log_simple_sequence_visualization(self, inputs: torch.Tensor, targets: torch.Tensor, 
                                    predictions: torch.Tensor, stage: str = "val", 
                                    batch_idx: int = 0, max_samples: int = 2):
        """
        Create a simple visualization showing input sequences, target sequences, and predictions
        
        Args:
            inputs: Input tensor [batch, time, height, width, channels]
            targets: Target tensor [batch, time, height, width, 1] (LST only)
            predictions: Prediction tensor [batch, time, height, width, 1] (LST only)
            stage: Training stage ("train", "val", "test")
            batch_idx: Current batch index
            max_samples: Maximum number of samples to visualize from batch
        """
        if not isinstance(self.logger, pl.loggers.WandbLogger):
            return
        
        # Convert tensors to float32 for visualization to avoid precision issues
        inputs = inputs.float().cpu()
        targets = targets.float().cpu()
        predictions = predictions.float().cpu()
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Only log every N epochs during training/validation
            if stage in ["train", "val"] and self.current_epoch % self.log_images_every_n_epochs != 0:
                return
            
            # Limit number of samples to visualize
            batch_size = min(inputs.shape[0], max_samples)
            
            for sample_idx in range(batch_size):
                # Extract single sample (already converted to float32 and CPU)
                input_seq = inputs[sample_idx].numpy()    # [time, H, W, channels]
                target_seq = targets[sample_idx].numpy()  # [time, H, W, 1]
                pred_seq = predictions[sample_idx].numpy() # [time, H, W, 1]
                
                # Get sequence lengths
                input_len = input_seq.shape[0]
                output_len = target_seq.shape[0]
                
                # Create figure: 3 rows x max timesteps columns
                max_timesteps = max(input_len, output_len)
                fig, axes = plt.subplots(3, max_timesteps, figsize=(4*max_timesteps, 12))
                
                # Handle case where we only have one timestep
                if max_timesteps == 1:
                    axes = axes.reshape(3, 1)
                
                # Row 0: Input sequences (show LST band - index 1)
                for t in range(input_len):
                    ax = axes[0, t]
                    lst_input = input_seq[t, :, :, 1]  # LST is band index 1
                    # Denormalize from [0,1] back to Fahrenheit
                    lst_input_fahrenheit = lst_input * (211.0 - (-189.0)) + (-189.0)
                    # Use actual min/max of this specific image for better color contrast
                    vmin_input = lst_input_fahrenheit.min()
                    vmax_input = lst_input_fahrenheit.max()
                    im = ax.imshow(lst_input_fahrenheit, cmap='RdYlBu_r', vmin=vmin_input, vmax=vmax_input)
                    ax.set_title(f'Input T={t+1}\n({vmin_input:.1f}°F - {vmax_input:.1f}°F)', fontsize=10)
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, label='°F')
                
                # Fill remaining input columns if needed
                for t in range(input_len, max_timesteps):
                    axes[0, t].axis('off')
                    axes[0, t].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[0, t].transAxes)
                
                # Row 1: Target sequences
                for t in range(output_len):
                    ax = axes[1, t]
                    lst_target = target_seq[t, :, :, 0]  # LST target
                    # Denormalize from [0,1] back to Fahrenheit
                    lst_target_fahrenheit = lst_target * (211.0 - (-189.0)) + (-189.0)
                    # Use actual min/max of this specific image for better color contrast
                    vmin_target = lst_target_fahrenheit.min()
                    vmax_target = lst_target_fahrenheit.max()
                    im = ax.imshow(lst_target_fahrenheit, cmap='RdYlBu_r', vmin=vmin_target, vmax=vmax_target)
                    ax.set_title(f'Target T={input_len+t+1}\n({vmin_target:.1f}°F - {vmax_target:.1f}°F)', fontsize=10)
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, label='°F')
                
                # Fill remaining target columns if needed
                for t in range(output_len, max_timesteps):
                    axes[1, t].axis('off')
                    axes[1, t].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[1, t].transAxes)
                
                # Row 2: Prediction sequences
                for t in range(output_len):
                    ax = axes[2, t]
                    lst_pred = pred_seq[t, :, :, 0]  # LST prediction
                    # Denormalize from [0,1] back to Fahrenheit
                    lst_pred_fahrenheit = lst_pred * (211.0 - (-189.0)) + (-189.0)
                    # Use actual min/max of this specific image for better color contrast
                    vmin_pred = lst_pred_fahrenheit.min()
                    vmax_pred = lst_pred_fahrenheit.max()
                    im = ax.imshow(lst_pred_fahrenheit, cmap='RdYlBu_r', vmin=vmin_pred, vmax=vmax_pred)
                    ax.set_title(f'Prediction T={input_len+t+1}\n({vmin_pred:.1f}°F - {vmax_pred:.1f}°F)', fontsize=10)
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, label='°F')
                
                # Fill remaining prediction columns if needed
                for t in range(output_len, max_timesteps):
                    axes[2, t].axis('off')
                    axes[2, t].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[2, t].transAxes)
                
                # Add row labels (which will appear as sections in WandB)
                axes[0, 0].text(-0.2, 0.5, 'INPUT SEQUENCE', rotation=90, ha='center', va='center',
                            transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold')
                axes[1, 0].text(-0.2, 0.5, 'TARGET SEQUENCE', rotation=90, ha='center', va='center',
                            transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold')
                axes[2, 0].text(-0.2, 0.5, 'PREDICTION SEQUENCE', rotation=90, ha='center', va='center',
                            transform=axes[2, 0].transAxes, fontsize=12, fontweight='bold')
                
                plt.suptitle(f'{stage.upper()} - Batch {batch_idx}, Sample {sample_idx+1}\n'
                            f'Input Length: {input_len}, Output Length: {output_len}', fontsize=14)
                plt.tight_layout()
                
                # Log to wandb
                self.logger.experiment.log({
                    f"{stage}_sequence_visualization_batch{batch_idx}_sample{sample_idx}": wandb.Image(fig)
                })
                
                plt.close(fig)
                
        except Exception as e:
            print(f"Warning: Failed to create sequence visualization: {e}")


    def on_validation_epoch_end(self):
        """Called at the end of validation epoch - good place to log visualizations"""
        super().on_validation_epoch_end()
        
        # Get a sample batch for visualization
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'val_dataloader'):
            try:
                val_dataloader = self.trainer.datamodule.val_dataloader()
                sample_batch = next(iter(val_dataloader))
                
                # Move to device if needed
                if len(sample_batch) == 2:
                    inputs, targets = sample_batch
                    if self.device != inputs.device:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                    
                    # Get predictions with proper precision handling
                    with torch.no_grad():
                        # Ensure inputs are in the same precision as model
                        if self.dtype != inputs.dtype:
                            inputs = inputs.to(dtype=self.dtype)
                        predictions = self.forward(inputs)
                    
                    # Create visualization
                    self.log_simple_sequence_visualization(
                        inputs, targets, predictions, 
                        stage="val", batch_idx=0, max_samples=2
                    )
                    
            except Exception as e:
                print(f"Warning: Could not create validation visualization: {e}")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step"""
        inputs, targets = batch
        predictions = self.forward(inputs)
        loss = self.criterion(predictions, targets)
        
        # Calculate comprehensive test metrics
        mae = torch.nn.functional.l1_loss(predictions, targets)
        rmse = torch.sqrt(torch.nn.functional.mse_loss(predictions, targets))
        
        self.log('test_loss', torch.sqrt(loss), on_step=False, on_epoch=True)
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