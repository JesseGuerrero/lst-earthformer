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
        log_images_every_n_epochs: int = 1,
        max_images_to_log: int = 4,
        input_sequence_length: int = 3,
        output_sequence_length: int = 3,
        model_size: str = "small",  # NEW: "tiny", "small", "medium", "large"
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
        
        # Model size configurations
        model_configs = {
            "tiny": {
                'base_units': 64,
                'num_heads': 4,
                'enc_depth': [1, 1],
                'dec_depth': [1, 1],
                'enc_cuboid_size': [(2, 4, 4), (2, 4, 4)],
                'num_global_vectors': 4,
            },
            "small": {
                'base_units': 96,
                'num_heads': 6,
                'enc_depth': [2, 2],
                'dec_depth': [1, 1],
                'enc_cuboid_size': [(2, 4, 4), (2, 4, 4)],
                'num_global_vectors': 8,
            },
                "medium": {
                'base_units': 128,      # Keep same
                'num_heads': 8,         # Keep same  
                'enc_depth': [2, 2],    # REDUCED from [3, 3] 
                'dec_depth': [1, 1],    # REDUCED from [2, 2]
                'enc_cuboid_size': [(2, 4, 4), (2, 4, 4)],  # REDUCED from [(2, 4, 4), (2, 8, 8)]
                'num_global_vectors': 12,  # REDUCED from 16
            },
            "large": {
                'base_units': 144,      # REDUCED from 192
                'num_heads': 8,         # REDUCED from 12 (must divide base_units)
                'enc_depth': [2, 2],    # REDUCED from [4, 4]  
                'dec_depth': [1, 1],    # REDUCED from [3, 3]
                'enc_cuboid_size': [(2, 4, 4), (2, 4, 4)],  # REDUCED from [(2, 8, 8), (2, 8, 8)]
                'num_global_vectors': 16,  # REDUCED from 32
            }
        }
        
        # Get base config for selected model size
        selected_config = model_configs.get(model_size, model_configs["small"])
        
        # Default Landsat-optimized config (shared across all sizes)
        self.model_config = {
            'input_shape': (input_sequence_length, 128, 128, 9),
            'target_shape': (output_sequence_length, 128, 128, 1),
            'attn_drop': 0.1,
            'proj_drop': 0.1,
            'ffn_drop': 0.1,
            'use_dec_self_global': True,
            'use_dec_cross_global': True,
            'pos_embed_type': 't+hw',
            'use_relative_pos': True,
            'ffn_activation': 'gelu',
            'enc_cuboid_strategy': [('l', 'l', 'l'), ('d', 'd', 'd')],
            'dec_cross_cuboid_hw': [(4, 4), (4, 4)],
            'dec_cross_n_temporal': [1, 2],
        }
        
        # Update with size-specific config
        self.model_config.update(selected_config)
        
        # Update with any provided kwargs (allows override)
        self.model_config.update(model_kwargs)
        
        # Initialize model
        from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
        self.model = CuboidTransformerModel(**self.model_config)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Band names for visualization
        self.band_names = ['DEM (+10k offset)', 'LST (°F)', 'Red (×10k)', 'Green (×10k)', 'Blue (×10k)', 
                        'NDVI (×10k)', 'NDWI (×10k)', 'NDBI (×10k)', 'Albedo (×10k)']
        
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Model '{model_size}' initialized with {param_count:,} parameters")
    
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
        """Training step with Fahrenheit temperature metrics"""
        inputs, targets = batch
        
        # Forward pass
        predictions = self.forward(inputs)
        
        # Calculate loss
        loss = self.criterion(predictions, targets)
        
        # Calculate metrics
        mae = torch.nn.functional.l1_loss(predictions, targets)
        
        # Log metrics
        self.log('train_loss', torch.sqrt(loss), on_step=False, on_epoch=True)
        self.log('train_mae', mae, on_step=False, on_epoch=True)
        
        # Calculate temperature-specific metrics in Fahrenheit
        with torch.no_grad():
            # Denormalize to Fahrenheit: value * (max - min) + min
            pred_fahrenheit = predictions.detach() * (211.0 - (-189.0)) + (-189.0)
            true_fahrenheit = targets.detach() * (211.0 - (-189.0)) + (-189.0)
            
            temp_mae_f = torch.nn.functional.l1_loss(pred_fahrenheit, true_fahrenheit)
            temp_rmse_f = torch.sqrt(torch.nn.functional.mse_loss(pred_fahrenheit, true_fahrenheit))
            
            self.log('train_mae_F', temp_mae_f, on_step=False, on_epoch=True, prog_bar=True)
            self.log('train_rmse_F', temp_rmse_f, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def debug_image_logging_in_sweep(self, inputs, targets, predictions, stage="debug"):
        """
        Debug version of image logging to identify sweep issues
        """
        print(f"\n🔍 DEBUG: Image logging called for {stage}")
        print(f"📊 Logger type: {type(self.logger)}")
        print(f"🔗 Wandb available: {wandb.run is not None}")
        
        if wandb.run:
            print(f"🏃 Wandb run name: {wandb.run.name}")
            print(f"📁 Wandb run id: {wandb.run.id}")
        
        # Check if logger is WandB
        if not isinstance(self.logger, pl.loggers.WandbLogger):
            print(f"❌ Logger is not WandbLogger: {type(self.logger)}")
            return
        
        # Check tensor shapes and types
        print(f"📐 Input shape: {inputs.shape}, dtype: {inputs.dtype}")
        print(f"📐 Target shape: {targets.shape}, dtype: {targets.dtype}")
        print(f"📐 Prediction shape: {predictions.shape}, dtype: {predictions.dtype}")
        
        # Check if we're in the right epoch
        print(f"📅 Current epoch: {self.current_epoch}")
        print(f"📅 Log every N epochs: {self.log_images_every_n_epochs}")
        print(f"✅ Should log images: {self.current_epoch % self.log_images_every_n_epochs == 0}")
        
        try:
            # Create a simple test image
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.imshow(np.random.random((64, 64)), cmap='viridis')
            ax.set_title(f"Debug Image - Epoch {self.current_epoch}")
            ax.axis('off')
            
            # Try to log the test image
            self.logger.experiment.log({
                f"debug_test_image_{stage}": wandb.Image(fig)
            })
            
            plt.close(fig)
            print("✅ Successfully logged debug image!")
            
        except Exception as e:
            print(f"❌ Failed to log debug image: {e}")
            import traceback
            traceback.print_exc()
        
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step with direct image logging that works in sweeps"""
        inputs, targets = batch
        predictions = self.forward(inputs)
        
        # Calculate loss in normalized space
        loss = self.criterion(predictions, targets)
        mae = torch.nn.functional.l1_loss(predictions, targets)
        
        # Log normalized metrics
        self.log('val_loss', torch.sqrt(loss), on_step=False, on_epoch=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True)
        
        # Calculate metrics in Fahrenheit
        with torch.no_grad():
            # Denormalize to Fahrenheit: value * (max - min) + min
            pred_fahrenheit = predictions.detach() * (211.0 - (-189.0)) + (-189.0)
            true_fahrenheit = targets.detach() * (211.0 - (-189.0)) + (-189.0)
            
            temp_mae_f = torch.nn.functional.l1_loss(pred_fahrenheit, true_fahrenheit)
            temp_rmse_f = torch.sqrt(torch.nn.functional.mse_loss(pred_fahrenheit, true_fahrenheit))
            
            # Log ACTUAL temperature metrics
            self.log('val_mae_F', temp_mae_f, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_rmse_F', temp_rmse_f, on_step=False, on_epoch=True, prog_bar=True)
            
            # Correlation (same in both spaces)
            pred_flat = pred_fahrenheit.flatten()
            true_flat = true_fahrenheit.flatten()
            mask = torch.isfinite(pred_flat) & torch.isfinite(true_flat)
            if mask.sum() > 1:
                correlation = torch.corrcoef(torch.stack([pred_flat[mask], true_flat[mask]]))[0, 1]
                if torch.isfinite(correlation):
                    self.log('val_correlation', correlation, on_step=False, on_epoch=True)
        
        # DIRECT IMAGE LOGGING IN VALIDATION STEP
        # This works in both normal runs and sweeps
        if (batch_idx == 0 and  # Only first batch
            self.current_epoch % self.log_images_every_n_epochs == 0 and  # Every N epochs
            wandb.run is not None):  # Only if wandb is available
            
            try:
                print(f"🖼️ Attempting to log images at epoch {self.current_epoch}")
                
                # Convert to CPU and numpy
                inputs_cpu = inputs[0:1].float().cpu().numpy()  # Take only first sample
                targets_cpu = targets[0:1].float().cpu().numpy()
                predictions_cpu = predictions[0:1].detach().float().cpu().numpy()
                
                # Extract sequences
                input_seq = inputs_cpu[0]    # [time, H, W, channels]
                target_seq = targets_cpu[0]  # [time, H, W, 1]
                pred_seq = predictions_cpu[0] # [time, H, W, 1]
                
                input_len = input_seq.shape[0]
                output_len = target_seq.shape[0]
                max_timesteps = max(input_len, output_len)
                
                # Create the visualization
                fig, axes = plt.subplots(3, max_timesteps, figsize=(4*max_timesteps, 12))
                
                # Handle single timestep case
                if max_timesteps == 1:
                    axes = axes.reshape(3, 1)
                
                fig.patch.set_facecolor('lightgray')
                
                # Row 0: Input sequences (LST band - index 1)
                for t in range(input_len):
                    ax = axes[0, t]
                    ax.set_facecolor('lightgray')
                    
                    lst_input = input_seq[t, :, :, 1]  # LST band
                    lst_input_fahrenheit = lst_input * (211.0 - (-189.0)) + (-189.0)
                    
                    # Create mask for NODATA
                    nodata_mask = np.abs(lst_input_fahrenheit - (-189.0)) < 0.1
                    lst_masked = np.ma.masked_where(nodata_mask, lst_input_fahrenheit)
                    
                    if not lst_masked.mask.all():
                        vmin_input = lst_masked.min()
                        vmax_input = lst_masked.max()
                        im = ax.imshow(lst_masked, cmap='RdYlBu_r', vmin=vmin_input, vmax=vmax_input, alpha=0.9)
                        ax.set_title(f'Input T={t+1}\n({vmin_input:.1f}°F - {vmax_input:.1f}°F)', fontsize=10)
                        plt.colorbar(im, ax=ax, fraction=0.046, label='°F')
                    else:
                        ax.imshow(np.zeros_like(lst_input_fahrenheit), cmap='RdYlBu_r', alpha=0)
                        ax.set_title(f'Input T={t+1}\n(No Valid Data)', fontsize=10)
                    
                    ax.axis('off')
                
                # Fill remaining input columns
                for t in range(input_len, max_timesteps):
                    axes[0, t].set_facecolor('lightgray')
                    axes[0, t].axis('off')
                    axes[0, t].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[0, t].transAxes)
                
                # Row 1: Target sequences
                for t in range(output_len):
                    ax = axes[1, t]
                    ax.set_facecolor('lightgray')
                    
                    lst_target = target_seq[t, :, :, 0]
                    lst_target_fahrenheit = lst_target * (211.0 - (-189.0)) + (-189.0)
                    
                    nodata_mask = np.abs(lst_target_fahrenheit - (-189.0)) < 0.1
                    lst_masked = np.ma.masked_where(nodata_mask, lst_target_fahrenheit)
                    
                    if not lst_masked.mask.all():
                        vmin_target = lst_masked.min()
                        vmax_target = lst_masked.max()
                        im = ax.imshow(lst_masked, cmap='RdYlBu_r', vmin=vmin_target, vmax=vmax_target, alpha=0.9)
                        ax.set_title(f'Target T={input_len+t+1}\n({vmin_target:.1f}°F - {vmax_target:.1f}°F)', fontsize=10)
                        plt.colorbar(im, ax=ax, fraction=0.046, label='°F')
                    else:
                        ax.imshow(np.zeros_like(lst_target_fahrenheit), cmap='RdYlBu_r', alpha=0)
                        ax.set_title(f'Target T={input_len+t+1}\n(No Valid Data)', fontsize=10)
                    
                    ax.axis('off')
                
                # Fill remaining target columns
                for t in range(output_len, max_timesteps):
                    axes[1, t].set_facecolor('lightgray')
                    axes[1, t].axis('off')
                    axes[1, t].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[1, t].transAxes)
                
                # Row 2: Prediction sequences
                for t in range(output_len):
                    ax = axes[2, t]
                    ax.set_facecolor('lightgray')
                    
                    lst_pred = pred_seq[t, :, :, 0]
                    lst_pred_fahrenheit = lst_pred * (211.0 - (-189.0)) + (-189.0)
                    
                    # Use target's mask for predictions
                    target_lst = target_seq[t, :, :, 0] * (211.0 - (-189.0)) + (-189.0)
                    nodata_mask = np.abs(target_lst - (-189.0)) < 0.1
                    lst_masked = np.ma.masked_where(nodata_mask, lst_pred_fahrenheit)
                    
                    if not lst_masked.mask.all():
                        vmin_pred = lst_masked.min()
                        vmax_pred = lst_masked.max()
                        im = ax.imshow(lst_masked, cmap='RdYlBu_r', vmin=vmin_pred, vmax=vmax_pred, alpha=0.9)
                        ax.set_title(f'Prediction T={input_len+t+1}\n({vmin_pred:.1f}°F - {vmax_pred:.1f}°F)', fontsize=10)
                        plt.colorbar(im, ax=ax, fraction=0.046, label='°F')
                    else:
                        ax.imshow(np.zeros_like(lst_pred_fahrenheit), cmap='RdYlBu_r', alpha=0)
                        ax.set_title(f'Prediction T={input_len+t+1}\n(No Valid Data)', fontsize=10)
                    
                    ax.axis('off')
                
                # Fill remaining prediction columns
                for t in range(output_len, max_timesteps):
                    axes[2, t].set_facecolor('lightgray')
                    axes[2, t].axis('off')
                    axes[2, t].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[2, t].transAxes)
                
                # Add row labels
                axes[0, 0].text(-0.2, 0.5, 'INPUT LST', rotation=90, ha='center', va='center',
                            transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold')
                axes[1, 0].text(-0.2, 0.5, 'TARGET LST', rotation=90, ha='center', va='center',
                            transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold')
                axes[2, 0].text(-0.2, 0.5, 'PREDICTED LST', rotation=90, ha='center', va='center',
                            transform=axes[2, 0].transAxes, fontsize=12, fontweight='bold')
                
                # Add title
                plt.suptitle(f'Validation - Epoch {self.current_epoch}, Batch {batch_idx}\n'
                            f'Input Length: {input_len}, Output Length: {output_len}', fontsize=12)
                plt.tight_layout()
                
                # Log directly to wandb (this should work in sweeps)
                wandb.log({
                    "validation_predictions": wandb.Image(fig)
                }, step=self.global_step)
                
                plt.close(fig)
                print(f"✅ Successfully logged validation image at epoch {self.current_epoch}")
                
            except Exception as e:
                print(f"❌ Image logging failed in validation_step: {e}")
                import traceback
                traceback.print_exc()
        
        return loss
    
    # def log_simple_sequence_visualization(self, inputs: torch.Tensor, targets: torch.Tensor, 
    #                     predictions: torch.Tensor, stage: str = "val", 
    #                     batch_idx: int = 0, max_samples: int = 2):
    #     """
    #     FIXED VERSION: Create visualization that works in sweeps
    #     """
    #     # CRITICAL FIX 1: Check if wandb is properly initialized
    #     if not wandb.run:
    #         print(f"⚠️ Wandb run not initialized - skipping image logging for {stage}")
    #         return
        
    #     # CRITICAL FIX 2: For sweeps, use wandb.log directly instead of self.logger.experiment.log
    #     # The logger might not be properly connected to the sweep run
        
    #     # Convert tensors to float32 for visualization to avoid precision issues
    #     inputs = inputs.float().cpu()
    #     targets = targets.float().cpu()
    #     predictions = predictions.float().cpu()
        
    #     try:                        
    #         # Limit number of samples to visualize
    #         batch_size = min(inputs.shape[0], max_samples)
            
    #         for sample_idx in range(batch_size):
    #             # Extract single sample (already converted to float32 and CPU)
    #             input_seq = inputs[sample_idx].numpy()    # [time, H, W, channels]
    #             target_seq = targets[sample_idx].numpy()  # [time, H, W, 1]
    #             pred_seq = predictions[sample_idx].numpy() # [time, H, W, 1]
                
    #             # Get sequence lengths
    #             input_len = input_seq.shape[0]
    #             output_len = target_seq.shape[0]
                
    #             # Get metadata for this sample
    #             metadata_info = self._get_sample_metadata(batch_idx, sample_idx)
                
    #             # Create figure: 3 rows x max timesteps columns
    #             max_timesteps = max(input_len, output_len)
    #             fig, axes = plt.subplots(3, max_timesteps, figsize=(4*max_timesteps, 12))
                
    #             # Handle case where we only have one timestep
    #             if max_timesteps == 1:
    #                 axes = axes.reshape(3, 1)
                
    #             # Set background color for better contrast with transparent pixels
    #             fig.patch.set_facecolor('lightgray')
                
    #             # Row 0: Input sequences (show LST band - index 1)
    #             for t in range(input_len):
    #                 ax = axes[0, t]
    #                 ax.set_facecolor('lightgray')  # Set axes background
                    
    #                 lst_input = input_seq[t, :, :, 1]  # LST is band index 1
    #                 # Denormalize from [0,1] back to Fahrenheit
    #                 lst_input_fahrenheit = lst_input * (211.0 - (-189.0)) + (-189.0)
                    
    #                 # Create mask for NODATA pixels (originally 0 before normalization)
    #                 # In normalized space, NODATA (0) becomes -189°F after denormalization
    #                 nodata_mask = np.abs(lst_input_fahrenheit - (-189.0)) < 0.1  # Small tolerance
                    
    #                 # Create masked array for transparency
    #                 lst_masked = np.ma.masked_where(nodata_mask, lst_input_fahrenheit)
                    
    #                 # Use actual min/max of valid data for better color contrast
    #                 if not lst_masked.mask.all():  # Check if we have valid data
    #                     vmin_input = lst_masked.min()
    #                     vmax_input = lst_masked.max()
    #                     im = ax.imshow(lst_masked, cmap='RdYlBu_r', vmin=vmin_input, vmax=vmax_input, alpha=0.9)
    #                     ax.set_title(f'Input T={t+1}\n({vmin_input:.1f}°F - {vmax_input:.1f}°F)', fontsize=10)
    #                 else:
    #                     # All data is NODATA
    #                     ax.imshow(np.zeros_like(lst_input_fahrenheit), cmap='RdYlBu_r', alpha=0)
    #                     ax.set_title(f'Input T={t+1}\n(No Valid Data)', fontsize=10)
    #                     im = None
                    
    #                 ax.axis('off')
    #                 if im is not None:
    #                     plt.colorbar(im, ax=ax, fraction=0.046, label='°F')
                
    #             # Fill remaining input columns if needed
    #             for t in range(input_len, max_timesteps):
    #                 axes[0, t].set_facecolor('lightgray')
    #                 axes[0, t].axis('off')
    #                 axes[0, t].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[0, t].transAxes)
                
    #             # Row 1: Target sequences (LST only)
    #             for t in range(output_len):
    #                 ax = axes[1, t]
    #                 ax.set_facecolor('lightgray')  # Set axes background
                    
    #                 lst_target = target_seq[t, :, :, 0]  # LST target (only channel)
    #                 # Denormalize from [0,1] back to Fahrenheit
    #                 lst_target_fahrenheit = lst_target * (211.0 - (-189.0)) + (-189.0)
                    
    #                 # Create mask for NODATA pixels
    #                 nodata_mask = np.abs(lst_target_fahrenheit - (-189.0)) < 0.1
                    
    #                 # Create masked array for transparency
    #                 lst_masked = np.ma.masked_where(nodata_mask, lst_target_fahrenheit)
                    
    #                 # Use actual min/max of valid data for better color contrast
    #                 if not lst_masked.mask.all():  # Check if we have valid data
    #                     vmin_target = lst_masked.min()
    #                     vmax_target = lst_masked.max()
    #                     im = ax.imshow(lst_masked, cmap='RdYlBu_r', vmin=vmin_target, vmax=vmax_target, alpha=0.9)
    #                     ax.set_title(f'Target T={input_len+t+1}\n({vmin_target:.1f}°F - {vmax_target:.1f}°F)', fontsize=10)
    #                 else:
    #                     # All data is NODATA
    #                     ax.imshow(np.zeros_like(lst_target_fahrenheit), cmap='RdYlBu_r', alpha=0)
    #                     ax.set_title(f'Target T={input_len+t+1}\n(No Valid Data)', fontsize=10)
    #                     im = None
                    
    #                 ax.axis('off')
    #                 if im is not None:
    #                     plt.colorbar(im, ax=ax, fraction=0.046, label='°F')
                
    #             # Fill remaining target columns if needed
    #             for t in range(output_len, max_timesteps):
    #                 axes[1, t].set_facecolor('lightgray')
    #                 axes[1, t].axis('off')
    #                 axes[1, t].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[1, t].transAxes)
                
    #             # Row 2: Prediction sequences (LST only)
    #             for t in range(output_len):
    #                 ax = axes[2, t]
    #                 ax.set_facecolor('lightgray')  # Set axes background
                    
    #                 lst_pred = pred_seq[t, :, :, 0]  # LST prediction (only channel)
    #                 # Denormalize from [0,1] back to Fahrenheit
    #                 lst_pred_fahrenheit = lst_pred * (211.0 - (-189.0)) + (-189.0)
                    
    #                 # For predictions, we might want to mask based on the target's valid pixels
    #                 # or use the same NODATA threshold
    #                 target_lst = target_seq[t, :, :, 0] * (211.0 - (-189.0)) + (-189.0)
    #                 nodata_mask = np.abs(target_lst - (-189.0)) < 0.1
                    
    #                 # Create masked array for transparency
    #                 lst_masked = np.ma.masked_where(nodata_mask, lst_pred_fahrenheit)
                    
    #                 # Use actual min/max of valid data for better color contrast
    #                 if not lst_masked.mask.all():  # Check if we have valid data
    #                     vmin_pred = lst_masked.min()
    #                     vmax_pred = lst_masked.max()
    #                     im = ax.imshow(lst_masked, cmap='RdYlBu_r', vmin=vmin_pred, vmax=vmax_pred, alpha=0.9)
    #                     ax.set_title(f'Prediction T={input_len+t+1}\n({vmin_pred:.1f}°F - {vmax_pred:.1f}°F)', fontsize=10)
    #                 else:
    #                     # All data is NODATA
    #                     ax.imshow(np.zeros_like(lst_pred_fahrenheit), cmap='RdYlBu_r', alpha=0)
    #                     ax.set_title(f'Prediction T={input_len+t+1}\n(No Valid Data)', fontsize=10)
    #                     im = None
                    
    #                 ax.axis('off')
    #                 if im is not None:
    #                     plt.colorbar(im, ax=ax, fraction=0.046, label='°F')
                
    #             # Fill remaining prediction columns if needed
    #             for t in range(output_len, max_timesteps):
    #                 axes[2, t].set_facecolor('lightgray')
    #                 axes[2, t].axis('off')
    #                 axes[2, t].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[2, t].transAxes)
                
    #             # Add row labels
    #             axes[0, 0].text(-0.2, 0.5, 'INPUT LST', rotation=90, ha='center', va='center',
    #                         transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold')
    #             axes[1, 0].text(-0.2, 0.5, 'TARGET LST', rotation=90, ha='center', va='center',
    #                         transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold')
    #             axes[2, 0].text(-0.2, 0.5, 'PREDICTED LST', rotation=90, ha='center', va='center',
    #                         transform=axes[2, 0].transAxes, fontsize=12, fontweight='bold')
                
    #             # Create title with metadata
    #             title_parts = [f'{stage.upper()} - Batch {batch_idx}, Sample {sample_idx+1}']
    #             if metadata_info:
    #                 title_parts.append(f'City: {metadata_info["city"]}, Tile: {metadata_info["tile_position"]}')
    #                 if "input_date_range" in metadata_info:
    #                     title_parts.append(f'Dates: {metadata_info["input_date_range"]} → {metadata_info["output_date_range"]}')
    #             title_parts.append(f'Input Length: {input_len}, Output Length: {output_len}')
    #             title_parts.append('Gray areas = NODATA pixels (transparent)')
                
    #             plt.suptitle('\n'.join(title_parts), fontsize=12)
    #             plt.tight_layout()
                
    #             # CRITICAL FIX 3: Use wandb.log directly instead of self.logger.experiment.log
    #             # This ensures compatibility with sweeps
    #             image_key = f"{stage}_lst_sequence_batch{batch_idx}_sample{sample_idx}"
                
    #             try:
    #                 # Method 1: Direct wandb logging (most reliable for sweeps)
    #                 wandb.log({image_key: wandb.Image(fig)})
    #                 print(f"✅ Successfully logged image: {image_key}")
                    
    #             except Exception as log_error:
    #                 print(f"❌ Failed to log image {image_key}: {log_error}")
                    
    #                 # Method 2: Fallback to logger if direct logging fails
    #                 try:
    #                     if hasattr(self, 'logger') and hasattr(self.logger, 'experiment'):
    #                         self.logger.experiment.log({image_key: wandb.Image(fig)})
    #                         print(f"✅ Successfully logged image via logger: {image_key}")
    #                 except Exception as fallback_error:
    #                     print(f"❌ Fallback logging also failed: {fallback_error}")
                
    #             plt.close(fig)
                
    #     except Exception as e:
    #         print(f"Warning: Failed to create LST temperature visualization: {e}")
    #         import traceback
    #         traceback.print_exc()

    def _get_sample_metadata(self, batch_idx: int, sample_idx: int) -> dict:
        """Get metadata for a specific sample in the batch"""
        try:
            if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'train_dataset'):
                # Determine which dataset to use based on current stage
                if hasattr(self.trainer, 'state') and hasattr(self.trainer.state, 'stage'):
                    if self.trainer.state.stage.name == 'VALIDATING':
                        dataset = self.trainer.datamodule.val_dataset
                    elif self.trainer.state.stage.name == 'TESTING':
                        dataset = getattr(self.trainer.datamodule, 'test_dataset', None)
                    else:
                        dataset = self.trainer.datamodule.train_dataset
                else:
                    # Fallback to train dataset
                    dataset = self.trainer.datamodule.train_dataset
                
                if dataset is None:
                    return {}
                
                # Calculate actual sample index from batch info
                batch_size = self.trainer.datamodule.batch_size
                actual_sample_idx = batch_idx * batch_size + sample_idx
                
                # Get the tile sequence info from dataset
                if hasattr(dataset, 'tile_sequences') and actual_sample_idx < len(dataset.tile_sequences):
                    city, tile_row, tile_col, input_months, output_months = dataset.tile_sequences[actual_sample_idx]
                    
                    return {
                        'city': city,
                        'tile_position': f"row_{tile_row:03d}_col_{tile_col:03d}",
                        'tile_row': tile_row,
                        'tile_col': tile_col,
                        'input_months': input_months,
                        'output_months': output_months,
                        'input_date_range': f"{input_months[0]} to {input_months[-1]}",
                        'output_date_range': f"{output_months[0]} to {output_months[-1]}"
                    }
            
            return {}
            
        except Exception as e:
            print(f"Warning: Could not get sample metadata: {e}")
            return {}


    # def on_validation_epoch_end(self):
    #     """Called at the end of validation epoch - good place to log visualizations"""
    #     super().on_validation_epoch_end()
        
    #     # Get a sample batch for visualization
    #     if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'val_dataloader'):
    #         try:
    #             val_dataloader = self.trainer.datamodule.val_dataloader()
    #             sample_batch = next(iter(val_dataloader))
                
    #             # Move to device if needed
    #             if len(sample_batch) == 2:
    #                 inputs, targets = sample_batch
    #                 if self.device != inputs.device:
    #                     inputs = inputs.to(self.device)
    #                     targets = targets.to(self.device)
                    
    #                 # Get predictions with proper precision handling
    #                 with torch.no_grad():
    #                     # Ensure inputs are in the same precision as model
    #                     if self.dtype != inputs.dtype:
    #                         inputs = inputs.to(dtype=self.dtype)
    #                     predictions = self.forward(inputs)
                    
    #                 # Create visualization
    #                 # self.debug_image_logging_in_sweep(inputs, targets, predictions, stage="val", batch_idx=0, max_samples=2)
    #                 self.log_simple_sequence_visualization(
    #                     inputs, targets, predictions, 
    #                     stage="val", batch_idx=0, max_samples=2
    #                 )

                    
    #         except Exception as e:
    #             print(f"Warning: Could not create validation visualization: {e}")

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