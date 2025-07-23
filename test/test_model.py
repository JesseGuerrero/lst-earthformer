import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
from sklearn.metrics import r2_score
from datetime import datetime

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
        model_size: str = "small",  # NEW: "tiny", "small", "medium", "large", "earthnet"
        **model_kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_root = Path("../Data/ML")
        # Store sequence lengths
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length

        # Image logging parameters
        self.log_images_every_n_epochs = log_images_every_n_epochs
        self.max_images_to_log = max_images_to_log

        # For Scatter Plot
        self.train_predictions = []
        self.train_targets = []
        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []

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
            },
            "earthnet": {
                'base_units': 256,
                'num_heads': 4,
                'enc_depth': [1, 1],
                'dec_depth': [1, 1],
                'enc_cuboid_size': [(2, 4, 4), (2, 4, 4)],
                'num_global_vectors': 8,
                'use_dec_self_global': False,  # Earthnet disables this
                'use_dec_cross_global': False,  # Earthnet disables this
                'initial_downsample_type': 'stack_conv',
                'initial_downsample_stack_conv_num_layers': 2,
                'initial_downsample_stack_conv_dim_list': [64, 256],
                'initial_downsample_stack_conv_downscale_list': [2, 2],
                'initial_downsample_stack_conv_num_conv_list': [2, 2],
                'initial_downsample_activation': 'leaky',
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
        self.criterion = nn.MSELoss(reduction='none')

        # Band names for visualization
        self.band_names = ['DEM (+10k offset)', 'LST (°F)', 'Red (×10k)', 'Green (×10k)', 'Blue (×10k)',
                        'NDVI (×10k)', 'NDWI (×10k)', 'NDBI (×10k)', 'Albedo (×10k)']

        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Model '{model_size}' initialized with {param_count:,} parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        return self.model(x)

    def create_correlation_plot(self, all_predictions, all_targets, split_name, epoch):
        """Create research-quality correlation scatter plot for WandB"""
        try:
            # Convert and flatten
            pred_flat = torch.cat(all_predictions).detach().cpu().numpy().flatten()
            true_flat = torch.cat(all_targets).detach().cpu().numpy().flatten()

            # Remove NODATA values
            valid_mask = (true_flat != 0) & (pred_flat != 0)
            pred_clean = pred_flat[valid_mask]
            true_clean = true_flat[valid_mask]

            if len(pred_clean) < 100:  # Need minimum points for meaningful plot
                return None

            # Convert to Fahrenheit for interpretation
            pred_fahrenheit = pred_clean * (211.0 - (-189.0)) + (-189.0)
            true_fahrenheit = true_clean * (211.0 - (-189.0)) + (-189.0)

            # Calculate metrics
            correlation = np.corrcoef(true_fahrenheit, pred_fahrenheit)[0, 1]
            r2 = r2_score(true_fahrenheit, pred_fahrenheit)
            mae = np.mean(np.abs(true_fahrenheit - pred_fahrenheit))
            rmse = np.sqrt(np.mean((true_fahrenheit - pred_fahrenheit)**2))

            # Create plot
            fig, ax = plt.subplots(figsize=(8, 8))

            # Scatter plot with transparency
            ax.scatter(true_fahrenheit, pred_fahrenheit, alpha=0.6, s=8, color='gray', edgecolors='none')

            # Perfect prediction line (diagonal)
            min_temp = min(true_fahrenheit.min(), pred_fahrenheit.min())
            max_temp = max(true_fahrenheit.max(), pred_fahrenheit.max())
            ax.plot([min_temp, max_temp], [min_temp, max_temp], 'k--', linewidth=2, label='Perfect Prediction')

            # Labels and title
            ax.set_xlabel('Ground Truth Mean (Background Temperature) (F)', fontsize=12)
            ax.set_ylabel('Mean (Background Temperature) (F)', fontsize=12)
            ax.set_title(f'{split_name.title()} - Epoch {epoch}\n'
                        f'Correlation: {correlation:.3f}, R²: {r2:.3f}, MAE: {mae:.1f}°F, RMSE: {rmse:.1f}°F',
                        fontsize=12)

            # Equal aspect ratio and clean appearance
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            return fig

        except Exception as e:
            print(f"Error creating correlation plot: {e}")
            return None

    def masked_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = (targets != 0).float()
        loss_elements = self.criterion(predictions, targets)
        masked_loss = loss_elements * valid_mask
        valid_count = valid_mask.sum()

        if valid_count > 0:
            return masked_loss.sum() / valid_count
        else:
            # Return a small loss that won't cause NaN gradients
            return torch.tensor(1e-8, device=predictions.device, dtype=predictions.dtype)

    def masked_mae(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate MAE with NODATA masking"""
        valid_mask = (targets != 0).float()
        mae_elements = torch.abs(predictions - targets)
        masked_mae = mae_elements * valid_mask
        valid_count = valid_mask.sum()

        if valid_count > 0:
            return masked_mae.sum() / valid_count
        else:
            return torch.tensor(0.0, device=predictions.device)

    def extract_batch_metadata(self, batch_idx: int) -> Dict[str, Any]:
        """
        Extract metadata from the current batch.
        This method should be called during the training/validation step.
        """
        # Get the dataset to extract metadata
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'test_dataset'):
            dataset = self.trainer.datamodule.test_dataset

            # Calculate actual sample indices from batch
            batch_size = self.trainer.datamodule.batch_size
            start_idx = batch_idx * batch_size # Start idx is the sample id

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

        file_paths = {
            'dem_path': str(self.dataset_root / "DEM_2014_Tiles" / city / f"DEM_row_{tile_row:03d}_col_{tile_col:03d}.tif"),
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

    def _get_monthly_scenes(self, cluster: str, city: str) -> Dict[str, Dict[str, str]]:
        city_dir = self.dataset_root / "Clustered" / cluster / "Cities_Tiles" / city
        if not city_dir.exists():
            return {}

        monthly_scenes = {}
        scene_dirs = [d for d in city_dir.iterdir() if d.is_dir()]
        for scene_dir in scene_dirs:
            try:
                date_str = scene_dir.name  # e.g., "2016-12-26T18:10:25Z"
                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))

                # Filter by years for this split
                if date_obj.year not in self.years:
                    continue

                # Additional filtering for debug monthly split
                if self.debug_monthly_split and self.allowed_months is not None:
                    if date_obj.month not in self.allowed_months:
                        continue

                month_key = f"{date_obj.year}-{date_obj.month:02d}"

                # Only keep first scene per month
                if month_key not in monthly_scenes:
                    if self._validate_tiled_scene(scene_dir):
                        monthly_scenes[month_key] = str(scene_dir)

            except Exception as e:
                print(f"Warning: Could not parse date from {scene_dir.name}: {e}")
                continue
        '''
        {
            "1": {
                "San Antonio": {
                    "2013-04": "scene_path",
                    "2013-05": "scene_path",
                    "2013-06": "scene_path",
                    etc...
                    "2025-04": "scene_path",
                    "2025-05": "scene_path",
                    "2025-06": "scene_path",
                }
            }
        }
        '''
        return monthly_scenes

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with Fahrenheit temperature metrics"""
        inputs, targets = batch

        # Forward pass
        predictions = self.forward(inputs)

        # Calculate loss
        loss = self.masked_loss(predictions, targets)

        # Calculate metrics
        mae = self.masked_mae(predictions, targets)

        # Log metrics
        self.log('train_loss', torch.sqrt(loss), on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_mae', mae, on_step=False, on_epoch=True, sync_dist=True)

        # Calculate temperature-specific metrics in Fahrenheit
        with torch.no_grad():
            # Denormalize to Fahrenheit: value * (max - min) + min
            pred_fahrenheit = predictions.detach() * (211.0 - (-189.0)) + (-189.0)
            true_fahrenheit = targets.detach() * (211.0 - (-189.0)) + (-189.0)

            temp_mae_f = self.masked_mae(pred_fahrenheit, true_fahrenheit)
            temp_rmse_f = torch.sqrt(self.masked_loss(pred_fahrenheit, true_fahrenheit))

            self.log('train_mae_F', temp_mae_f, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('train_rmse_F', temp_rmse_f, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Store predictions for correlation plot (limit to avoid memory issues)
        if self.trainer.is_global_zero and len(self.train_predictions) < 50:  # Limit to ~50 batches per epoch
            self.train_predictions.append(predictions.detach().cpu())
            self.train_targets.append(targets.detach().cpu())

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step with direct image logging that works in sweeps"""
        inputs, targets = batch
        predictions = self.forward(inputs)

        # Calculate loss in normalized space
        loss = self.masked_loss(predictions, targets)
        mae = self.masked_mae(predictions, targets)

        # Log normalized metrics
        self.log('val_loss', torch.sqrt(loss), on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True, sync_dist=True)

        # Calculate metrics in Fahrenheit
        with torch.no_grad():
            # Denormalize to Fahrenheit: value * (max - min) + min
            pred_fahrenheit = predictions.detach() * (211.0 - (-189.0)) + (-189.0)
            true_fahrenheit = targets.detach() * (211.0 - (-189.0)) + (-189.0)

            temp_mae_f = self.masked_mae(pred_fahrenheit, true_fahrenheit)
            temp_rmse_f = torch.sqrt(self.masked_loss(pred_fahrenheit, true_fahrenheit))

            # Log ACTUAL temperature metrics
            self.log('val_mae_F', temp_mae_f, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val_rmse_F', temp_rmse_f, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

            # Correlation (same in both spaces)
            pred_flat = pred_fahrenheit.flatten()
            true_flat = true_fahrenheit.flatten()
            # Add NODATA masking to correlation
            valid_data_mask = (true_flat != 0) & (pred_flat != 0)  # Exclude NODATA
            finite_mask = torch.isfinite(pred_flat) & torch.isfinite(true_flat)
            mask = valid_data_mask & finite_mask
            if mask.sum() > 1:
                correlation = torch.corrcoef(torch.stack([pred_flat[mask], true_flat[mask]]))[0, 1]
                if torch.isfinite(correlation):
                    self.log('val_correlation', correlation, on_step=False, on_epoch=True, sync_dist=True)

        # Store predictions for correlation plot (limit to avoid memory issues)
        if self.trainer.is_global_zero and len(self.val_predictions) < 50:  # Limit to ~50 batches per epoch
            self.val_predictions.append(predictions.detach().cpu())
            self.val_targets.append(targets.detach().cpu())

        # DIRECT IMAGE LOGGING IN VALIDATION STEP
        # This works in both normal runs and sweeps
        if (batch_idx == 0 and  # Only first batch
            self.current_epoch % self.log_images_every_n_epochs == 0 and  # Every N epochs
            wandb.run is not None):  # Only if wandb is available

            try:
                print(f"🖼️ Attempting to log images at epoch {self.current_epoch}")

                # Determine how many samples to log (up to 4, limited by batch size)
                batch_size = inputs.shape[0]
                num_samples_to_log = min(4, batch_size, self.max_images_to_log)

                # Convert to CPU and numpy for all samples we want to log
                inputs_cpu = inputs[0:num_samples_to_log].float().cpu().numpy()
                targets_cpu = targets[0:num_samples_to_log].float().cpu().numpy()
                predictions_cpu = predictions[0:num_samples_to_log].detach().float().cpu().numpy()

                # Create a list to store all the figures
                wandb_images = []

                # Process each sample
                for sample_idx in range(num_samples_to_log):
                    # Extract sequences for this sample (keeping original indexing logic)
                    input_seq = inputs_cpu[sample_idx]    # [time, H, W, channels]
                    target_seq = targets_cpu[sample_idx]  # [time, H, W, 1]
                    pred_seq = predictions_cpu[sample_idx] # [time, H, W, 1]

                    input_len = input_seq.shape[0]
                    output_len = target_seq.shape[0]
                    max_timesteps = max(input_len, output_len)

                    # Create the visualization for this sample
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

                    # Add title with sample information
                    plt.suptitle(f'Validation Sample {sample_idx+1}/{num_samples_to_log} - Epoch {self.current_epoch}, Batch {batch_idx}\n'
                                f'Input Length: {input_len}, Output Length: {output_len}', fontsize=12)
                    plt.tight_layout()

                    # Add this figure to our list
                    wandb_images.append(wandb.Image(fig))
                    plt.close(fig)

                    # Log all images at once
                    wandb.log({
                        "validation_predictions": wandb_images
                    }, step=self.global_step)

                    print(f"✅ Successfully logged validation images at epoch {self.current_epoch}")

            except Exception as e:
                print(f"❌ Image logging failed in validation_step: {e}")
                import traceback
                traceback.print_exc()

        return loss



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

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step with proper masked loss usage and image logging similar to validation"""
        inputs, targets = batch
        metadata = self.extract_batch_metadata(batch_idx)
        # print(metadata)
        predictions = self.forward(inputs)

        # Calculate masked loss (MSE) - this is what we return for optimization
        loss = self.masked_loss(predictions, targets)

        # Calculate additional masked metrics
        mae = self.masked_mae(predictions, targets)
        rmse = torch.sqrt(loss)  # RMSE is just sqrt of the MSE loss

        # Log metrics - loss is already MSE, rmse is sqrt(MSE)
        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=True)  # Log raw MSE loss
        self.log('test_mae', mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_rmse', rmse, on_step=False, on_epoch=True, sync_dist=True)  # Log RMSE

        # Calculate metrics in Fahrenheit (similar to validation_step)
        with torch.no_grad():
            # Denormalize to Fahrenheit: value * (max - min) + min
            pred_fahrenheit = predictions.detach() * (211.0 - (-189.0)) + (-189.0)
            true_fahrenheit = targets.detach() * (211.0 - (-189.0)) + (-189.0)

            temp_mae_f = self.masked_mae(pred_fahrenheit, true_fahrenheit)
            temp_rmse_f = torch.sqrt(self.masked_loss(pred_fahrenheit, true_fahrenheit))

            # Log temperature metrics in Fahrenheit
            self.log('test_mae_F', temp_mae_f, on_step=False, on_epoch=True, sync_dist=True)
            self.log('test_rmse_F', temp_rmse_f, on_step=False, on_epoch=True, sync_dist=True)

            # Calculate correlation (excluding NODATA)
            pred_flat = pred_fahrenheit.flatten()
            true_flat = true_fahrenheit.flatten()
            valid_data_mask = (true_flat != 0) & (pred_flat != 0)  # Exclude NODATA
            finite_mask = torch.isfinite(pred_flat) & torch.isfinite(true_flat)
            mask = valid_data_mask & finite_mask

            if mask.sum() > 1:
                correlation = torch.corrcoef(torch.stack([pred_flat[mask], true_flat[mask]]))[0, 1]
                if torch.isfinite(correlation):
                    self.log('test_correlation', correlation, on_step=False, on_epoch=True, sync_dist=True)

        # DIRECT IMAGE LOGGING IN TEST STEP (similar to validation_step)
        # Log images for first few batches to see model performance on test data
        if (batch_idx < 3 and  # Only first 3 batches
            wandb.run is not None):  # Only if wandb is available

            try:
                print(f"🖼️ Attempting to log test images at batch {batch_idx}")

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

                # Add title with test-specific information
                plt.suptitle(f'Test - Batch {batch_idx}\n'
                            f'Input Length: {input_len}, Output Length: {output_len}', fontsize=12)
                plt.tight_layout()

                # FIXED: Log directly to wandb WITHOUT specifying step
                wandb.log({
                    "test_predictions": wandb.Image(fig)
                })  # Removed step parameter to let wandb auto-increment

                plt.close(fig)
                print(f"✅ Successfully logged test image at batch {batch_idx}")

            except Exception as e:
                print(f"❌ Test image logging failed at batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()

        # Store predictions for correlation plot
        if self.trainer.is_global_zero and len(self.test_predictions) < 20:
            self.test_predictions.append(predictions.detach().cpu())
            self.test_targets.append(targets.detach().cpu())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999)
        )

        # More aggressive clipping for earthnet
        model_size = getattr(self.hparams, 'model_size', 'small')
        if model_size == "earthnet":
            grad_clip_val = 1.0
        elif model_size == "large":
            grad_clip_val = 1.0
        else:
            grad_clip_val = 2.0

        return {
            "optimizer": optimizer,
            "gradient_clip_val": grad_clip_val,
            "gradient_clip_algorithm": "norm"
        }

    def on_train_epoch_end(self):
        print("train epoch end ran")
        """Create correlation plot at end of training epoch"""
        # Only log from rank 0 in distributed training
        if self.trainer.is_global_zero and len(self.train_predictions) > 0 and wandb.run is not None:
            try:
                fig = self.create_correlation_plot(
                    self.train_predictions, self.train_targets,
                    "training", self.current_epoch
                )
                if fig is not None:
                    wandb.log({
                        "train_correlation_plot": wandb.Image(fig),
                        "epoch": self.current_epoch
                    })
                    plt.close(fig)
                    print(f"✅ Logged training correlation plot for epoch {self.current_epoch}")
            except Exception as e:
                print(f"❌ Failed to create training correlation plot: {e}")

        # Clear stored data on all ranks
        self.train_predictions = []
        self.train_targets = []

    def on_validation_epoch_end(self):
        print("validation epoch end ran")
        """Create correlation plot at end of validation epoch"""
        # Only log from rank 0 in distributed training
        if self.trainer.is_global_zero and len(self.val_predictions) > 0 and wandb.run is not None:
            try:
                fig = self.create_correlation_plot(
                    self.val_predictions, self.val_targets,
                    "validation", self.current_epoch
                )
                if fig is not None:
                    wandb.log({
                        "val_correlation_plot": wandb.Image(fig),
                        "epoch": self.current_epoch
                    })
                    plt.close(fig)
                    print(f"✅ Logged validation correlation plot for epoch {self.current_epoch}")
            except Exception as e:
                print(f"❌ Failed to create validation correlation plot: {e}")

        # Clear stored data on all ranks
        self.val_predictions = []
        self.val_targets = []

    def on_test_epoch_end(self):
        print("test epoch end ran")
        """Create correlation plot at end of test epoch"""
        # Only log from rank 0 in distributed training
        if self.trainer.is_global_zero and len(self.test_predictions) > 0 and wandb.run is not None:
            try:
                fig = self.create_correlation_plot(
                    self.test_predictions, self.test_targets,
                    "test", self.current_epoch
                )
                if fig is not None:
                    wandb.log({
                        "test_correlation_plot": wandb.Image(fig),
                        "epoch": self.current_epoch
                    })
                    plt.close(fig)
                    print(f"✅ Logged test correlation plot for epoch {self.current_epoch}")
            except Exception as e:
                print(f"❌ Failed to create test correlation plot: {e}")

        # Clear stored data on all ranks
        self.test_predictions = []
        self.test_targets = []