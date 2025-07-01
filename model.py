# Enhanced model.py with rich metadata logging

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
        **model_kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Image logging parameters
        self.log_images_every_n_epochs = log_images_every_n_epochs
        self.max_images_to_log = max_images_to_log
        
        # Store metadata for each batch during training
        self.current_batch_metadata = {}
        
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
        
        # Initialize model - Import here to avoid issues
        try:
            from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
            self.model = CuboidTransformerModel(**self.model_config)
        except ImportError:
            print("Warning: Could not import CuboidTransformerModel. Using dummy model for testing.")
            self.model = self._create_dummy_model()
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Band names for visualization
        self.band_names = ['DEM (+10k offset)', 'LST (°F)', 'Red (×10k)', 'Green (×10k)', 'Blue (×10k)', 
                          'NDVI (×10k)', 'NDWI (×10k)', 'NDBI (×10k)', 'Albedo (×10k)']
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def _create_dummy_model(self):
        """Create a dummy model for testing when EarthFormer is not available"""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Simple CNN for testing
                self.conv1 = nn.Conv3d(9, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1))
                self.conv2 = nn.Conv3d(32, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1))
                self.conv3 = nn.Conv3d(16, 1, kernel_size=(1, 1, 1))
                
            def forward(self, x):
                # x shape: (B, T, H, W, C) -> need to convert to (B, C, T, H, W)
                x = x.permute(0, 4, 1, 2, 3)
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.conv3(x)
                # Convert back to (B, T, H, W, C)
                x = x.permute(0, 2, 3, 4, 1)
                return x
        
        return DummyModel()
    
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
            
            # LST target - already in Fahrenheit, no scaling needed
            lst_target = sample_target.copy()  # Keep original values
            lst_target_masked = np.where(lst_target == 0, np.nan, lst_target)
            
            # Debug: Print actual LST values to verify
            valid_lst = lst_target[lst_target != 0]
            if len(valid_lst) > 0:
                print(f"Sample {sample_idx}: LST range {np.min(valid_lst):.1f}°F to {np.max(valid_lst):.1f}°F")
            
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
            
            # Plot LST target - use actual data range, not percentiles
            if not np.all(np.isnan(lst_target_masked)):
                # Use actual min/max from the data for accurate color mapping
                temp_min = np.nanmin(lst_target_masked)
                temp_max = np.nanmax(lst_target_masked)
            else:
                temp_min, temp_max = 32, 100  # fallback
                
            lst_im = axes[sample_idx, 2].imshow(lst_target_masked, cmap='coolwarm', vmin=temp_min, vmax=temp_max)
            axes[sample_idx, 2].set_title(f'LST Target\n(t={middle_t}, {temp_min:.1f}-{temp_max:.1f}°F)')
            axes[sample_idx, 2].axis('off')
            cbar2 = plt.colorbar(lst_im, ax=axes[sample_idx, 2], fraction=0.046, pad=0.04)
            cbar2.set_label('Temperature (°F)')
            
            # Plot prediction if available
            if predictions is not None:
                sample_pred = predictions_np[sample_idx, middle_t, :, :, 0]  # (H, W)
                sample_pred_masked = np.where(sample_pred == 0, np.nan, sample_pred)
                
                # Use same temperature range as target for comparison
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
        
        # DEBUG: Print actual data ranges to understand scaling
        print("=== LST DATA ANALYSIS ===")
        for t in range(n_timesteps):
            input_lst = sample_input[t, :, :, 1]  # LST band index 1
            target_lst = sample_target[t, :, :, 0]
            
            # Analyze non-zero values
            input_nonzero = input_lst[input_lst != 0]
            target_nonzero = target_lst[target_lst != 0]
            
            print(f"Timestep {t}:")
            print(f"  Input LST  - Min: {input_nonzero.min():.1f}, Max: {input_nonzero.max():.1f}, Mean: {input_nonzero.mean():.1f}")
            print(f"  Target LST - Min: {target_nonzero.min():.1f}, Max: {target_nonzero.max():.1f}, Mean: {target_nonzero.mean():.1f}")
            
            if predictions is not None:
                pred_lst = predictions_np[sample_idx, t, :, :, 0]
                pred_nonzero = pred_lst[pred_lst != 0]
                print(f"  Pred LST   - Min: {pred_nonzero.min():.1f}, Max: {pred_nonzero.max():.1f}, Mean: {pred_nonzero.mean():.1f}")
        
        # Calculate global temperature range for consistent scaling
        # Use ALL timesteps and both input and target data
        all_input_lst = sample_input[:, :, :, 1].flatten()  # All input LST values
        all_target_lst = sample_target[:, :, :, 0].flatten()  # All target LST values
        
        # Combine input and target for global range (exclude zeros)
        all_lst_values = np.concatenate([
            all_input_lst[all_input_lst != 0],
            all_target_lst[all_target_lst != 0]
        ])
        
        if predictions is not None:
            all_pred_lst = predictions_np[sample_idx, :, :, :, 0].flatten()
            all_lst_values = np.concatenate([
                all_lst_values,
                all_pred_lst[all_pred_lst != 0]
            ])
        
        if len(all_lst_values) > 0:
            # Use wider percentile range for better contrast
            global_temp_min = np.percentile(all_lst_values, 1)  # 1st percentile
            global_temp_max = np.percentile(all_lst_values, 99)  # 99th percentile
            print(f"Global temp range: {global_temp_min:.1f}°F to {global_temp_max:.1f}°F")
        else:
            global_temp_min, global_temp_max = 32, 100
            print("Warning: No valid temperature data found, using default range")
        
        for t in range(n_timesteps):
            # Input LST (band index 1)
            input_lst = sample_input[t, :, :, 1]
            target_lst = sample_target[t, :, :, 0]
            
            # Mask nodata values properly
            input_lst_masked = np.where(input_lst == 0, np.nan, input_lst)
            target_lst_masked = np.where(target_lst == 0, np.nan, target_lst)
            
            # Plot input LST
            im1 = axes[t, 0].imshow(input_lst_masked, cmap='coolwarm', 
                                   vmin=global_temp_min, vmax=global_temp_max)
            axes[t, 0].set_title(f't={t}: Input LST (°F)\nRange: {np.nanmin(input_lst_masked):.1f}-{np.nanmax(input_lst_masked):.1f}°F')
            axes[t, 0].axis('off')
            cbar1 = plt.colorbar(im1, ax=axes[t, 0], fraction=0.046, pad=0.04)
            cbar1.set_label('Temperature (°F)')
            
            # Plot target LST
            im2 = axes[t, 1].imshow(target_lst_masked, cmap='coolwarm',
                                   vmin=global_temp_min, vmax=global_temp_max)
            axes[t, 1].set_title(f't={t}: Target LST (°F)\nRange: {np.nanmin(target_lst_masked):.1f}-{np.nanmax(target_lst_masked):.1f}°F')
            axes[t, 1].axis('off')
            cbar2 = plt.colorbar(im2, ax=axes[t, 1], fraction=0.046, pad=0.04)
            cbar2.set_label('Temperature (°F)')
            
            # Plot predicted LST if available
            if predictions is not None:
                pred_lst = predictions_np[sample_idx, t, :, :, 0]
                pred_lst_masked = np.where(pred_lst == 0, np.nan, pred_lst)
                im3 = axes[t, 2].imshow(pred_lst_masked, cmap='coolwarm',
                                       vmin=global_temp_min, vmax=global_temp_max)
                axes[t, 2].set_title(f't={t}: Predicted LST (°F)\nRange: {np.nanmin(pred_lst_masked):.1f}-{np.nanmax(pred_lst_masked):.1f}°F')
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
    
    def create_enhanced_wandb_table(self, metadata: Dict[str, Any], predictions: Optional[torch.Tensor] = None) -> wandb.Table:
        """Create a detailed WandB table with all metadata"""
        
        # Define table columns
        columns = [
            "sample_idx", "city", "tile_position", "tile_row", "tile_col",
            "input_date_range", "output_date_range", "sequence_length",
            "dem_file_path", "input_months", "output_months"
        ]
        
        if predictions is not None:
            columns.extend(["pred_temp_min", "pred_temp_max", "pred_temp_mean", "pred_temp_std"])
        
        # Create table data
        table_data = []
        
        for sample_meta in metadata.get('samples_metadata', []):
            row = [
                sample_meta['sample_idx'],
                sample_meta['city'],
                sample_meta['tile_position'],
                sample_meta['tile_row'],
                sample_meta['tile_col'],
                sample_meta['input_date_range'],
                sample_meta['output_date_range'],
                sample_meta['sequence_length'],
                sample_meta['file_paths'].get('dem_path', 'N/A'),
                ', '.join(sample_meta['input_months']),
                ', '.join(sample_meta['output_months'])
            ]
            
            # Add prediction statistics if available
            if predictions is not None:
                sample_idx = sample_meta['sample_idx'] % predictions.shape[0]
                pred_sample = predictions[sample_idx].detach().cpu().numpy()
                pred_sample_clean = pred_sample[pred_sample != 0]  # Remove nodata
                
                if len(pred_sample_clean) > 0:
                    row.extend([
                        float(np.min(pred_sample_clean)),
                        float(np.max(pred_sample_clean)),
                        float(np.mean(pred_sample_clean)),
                        float(np.std(pred_sample_clean))
                    ])
                else:
                    row.extend([None, None, None, None])
            
            table_data.append(row)
        
        return wandb.Table(columns=columns, data=table_data)
    
    def calculate_image_statistics(self, inputs: torch.Tensor, targets: torch.Tensor, 
                                 predictions: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Calculate comprehensive statistics for the batch"""
        
        # Convert to numpy
        inputs_np = inputs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        stats = {
            'batch_size': inputs.shape[0],
            'sequence_length': inputs.shape[1],
            'spatial_dimensions': [inputs.shape[2], inputs.shape[3]],
            'num_input_bands': inputs.shape[4],
            'target_bands': targets.shape[4],
        }
        
        # Input statistics (for each band)
        band_names = ['DEM', 'LST', 'Red', 'Green', 'Blue', 'NDVI', 'NDWI', 'NDBI', 'Albedo']
        for i, band_name in enumerate(band_names):
            if i < inputs.shape[4]:
                band_data = inputs_np[:, :, :, :, i]
                band_data_clean = band_data[band_data != 0]  # Remove nodata
                
                if len(band_data_clean) > 0:
                    stats[f'{band_name.lower()}_min'] = float(np.min(band_data_clean))
                    stats[f'{band_name.lower()}_max'] = float(np.max(band_data_clean))
                    stats[f'{band_name.lower()}_mean'] = float(np.mean(band_data_clean))
                    stats[f'{band_name.lower()}_std'] = float(np.std(band_data_clean))
        
        # Target statistics
        target_data = targets_np[:, :, :, :, 0]  # LST is the only target band
        target_clean = target_data[target_data != 0]
        
        if len(target_clean) > 0:
            stats.update({
                'target_lst_min': float(np.min(target_clean)),
                'target_lst_max': float(np.max(target_clean)),
                'target_lst_mean': float(np.mean(target_clean)),
                'target_lst_std': float(np.std(target_clean))
            })
        
        # Prediction statistics
        if predictions is not None:
            pred_np = predictions.detach().cpu().numpy()
            pred_data = pred_np[:, :, :, :, 0]
            pred_clean = pred_data[pred_data != 0]
            
            if len(pred_clean) > 0:
                stats.update({
                    'pred_lst_min': float(np.min(pred_clean)),
                    'pred_lst_max': float(np.max(pred_clean)),
                    'pred_lst_mean': float(np.mean(pred_clean)),
                    'pred_lst_std': float(np.std(pred_clean))
                })
                
                # Error statistics - ensure same shape
                if len(target_clean) > 0 and len(pred_clean) == len(target_clean):
                    error = pred_clean - target_clean
                    stats.update({
                        'mae': float(np.mean(np.abs(error))),
                        'rmse': float(np.sqrt(np.mean(error**2))),
                        'bias': float(np.mean(error)),
                        'error_std': float(np.std(error))
                    })
                elif len(target_clean) > 0 and len(pred_clean) > 0:
                    # If shapes don't match, calculate on whole arrays
                    pred_whole = pred_np.flatten()
                    target_whole = targets_np.flatten()
                    valid_mask = (pred_whole != 0) & (target_whole != 0)
                    if valid_mask.sum() > 0:
                        pred_valid = pred_whole[valid_mask]
                        target_valid = target_whole[valid_mask]
                        error = pred_valid - target_valid
                        stats.update({
                            'mae': float(np.mean(np.abs(error))),
                            'rmse': float(np.sqrt(np.mean(error**2))),
                            'bias': float(np.mean(error)),
                            'error_std': float(np.std(error))
                        })
        
        return stats
    
    def log_images_to_wandb(self, inputs: torch.Tensor, targets: torch.Tensor,
                           predictions: Optional[torch.Tensor] = None,
                           stage: str = "train", batch_idx: int = 0):
        """Enhanced image logging with comprehensive metadata"""
        if not isinstance(self.logger, pl.loggers.WandbLogger):
            return
        
        try:
            # Extract metadata for this batch
            metadata = self.extract_batch_metadata(None, batch_idx)
            
            # Calculate statistics with error handling
            try:
                stats = self.calculate_image_statistics(inputs, targets, predictions)
            except Exception as stats_error:
                print(f"Warning: Failed to calculate statistics: {stats_error}")
                stats = {
                    'batch_size': inputs.shape[0],
                    'sequence_length': inputs.shape[1],
                    'spatial_dimensions': [inputs.shape[2], inputs.shape[3]],
                    'error': str(stats_error)
                }
            
            # Create visualizations with error handling
            try:
                fig1 = self.create_landsat_visualization(
                    inputs, targets, predictions, batch_idx, self.max_images_to_log
                )
            except Exception as viz_error:
                print(f"Warning: Failed to create main visualization: {viz_error}")
                fig1 = None
            
            if fig1 is not None:
                # Create enhanced caption with metadata
                caption_parts = [
                    f"{stage.title()} - Epoch {self.current_epoch}, Batch {batch_idx}",
                    f"Samples: {len(metadata.get('samples_metadata', []))}",
                ]
                
                # Add sample-specific info to caption
                if metadata.get('samples_metadata'):
                    sample_info = []
                    for sample_meta in metadata['samples_metadata'][:2]:  # First 2 samples
                        sample_info.append(f"{sample_meta['city']} ({sample_meta['tile_position']})")
                    caption_parts.append(f"Cities/Tiles: {', '.join(sample_info)}")
                    
                    if len(metadata['samples_metadata']) > 2:
                        caption_parts.append(f"+ {len(metadata['samples_metadata']) - 2} more")
                
                enhanced_caption = "\n".join(caption_parts)
                
                # Log main visualization with metadata
                log_dict = {
                    f"{stage}_landsat_tiles": wandb.Image(
                        fig1, 
                        caption=enhanced_caption
                    ),
                    f"{stage}_batch_metadata": metadata,
                    f"{stage}_batch_statistics": stats,
                }
                
                # Create and log metadata table with error handling
                try:
                    if metadata.get('samples_metadata'):
                        table = self.create_enhanced_wandb_table(metadata, predictions)
                        log_dict[f"{stage}_sample_details"] = table
                except Exception as table_error:
                    print(f"Warning: Failed to create metadata table: {table_error}")
                
                # Log everything
                self.logger.experiment.log(log_dict, step=self.global_step)
                plt.close(fig1)
                
                # Create temporal sequence for first sample with metadata
                try:
                    if inputs.shape[0] > 0 and metadata.get('samples_metadata'):
                        fig2 = self.create_temporal_sequence_viz(inputs, targets, predictions, sample_idx=0)
                        
                        first_sample = metadata['samples_metadata'][0]
                        temporal_caption = (
                            f"{stage} Temporal Sequence - Epoch {self.current_epoch}\n"
                            f"City: {first_sample['city']}, Tile: {first_sample['tile_position']}\n"
                            f"Input: {first_sample['input_date_range']}\n"
                            f"Output: {first_sample['output_date_range']}"
                        )
                        
                        self.logger.experiment.log({
                            f"{stage}_temporal_sequence": wandb.Image(fig2, caption=temporal_caption),
                            f"{stage}_temporal_metadata": first_sample
                        }, step=self.global_step)
                        
                        plt.close(fig2)
                        
                        # Create difference visualization if we have predictions
                        if predictions is not None:
                            try:
                                fig3 = self.create_difference_visualization(targets, predictions, sample_idx=0)
                                error_caption = (
                                    f"{stage} Prediction Errors - Epoch {self.current_epoch}\n"
                                    f"City: {first_sample['city']}, Tile: {first_sample['tile_position']}\n"
                                    f"MAE: {stats.get('mae', 'N/A'):.2f}°F, "
                                    f"RMSE: {stats.get('rmse', 'N/A'):.2f}°F"
                                )
                                
                                self.logger.experiment.log({
                                    f"{stage}_prediction_errors": wandb.Image(fig3, caption=error_caption)
                                }, step=self.global_step)
                                
                                plt.close(fig3)
                            except Exception as diff_error:
                                print(f"Warning: Failed to create difference visualization: {diff_error}")
                                
                except Exception as temporal_error:
                    print(f"Warning: Failed to create temporal visualizations: {temporal_error}")
        
        except Exception as e:
            print(f"Warning: Failed to log images to wandb: {e}")
            # Log basic information at least
            try:
                if isinstance(self.logger, pl.loggers.WandbLogger):
                    self.logger.experiment.log({
                        f"{stage}_logging_error": {
                            "error": str(e),
                            "epoch": self.current_epoch,
                            "batch_idx": batch_idx,
                            "input_shape": list(inputs.shape),
                            "target_shape": list(targets.shape)
                        }
                    }, step=self.global_step)
            except:
                pass  # Fail silently if even basic logging fails
    
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
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True)
        
        # Calculate temperature-specific metrics
        with torch.no_grad():
            pred_temp = predictions.detach()
            true_temp = targets.detach()
            
            temp_mae = torch.nn.functional.l1_loss(pred_temp, true_temp)
            temp_rmse = torch.sqrt(torch.nn.functional.mse_loss(pred_temp, true_temp))
            
            self.log('train_temp_mae_scaled', temp_mae, on_step=False, on_epoch=True)
            self.log('train_temp_rmse_scaled', temp_rmse, on_step=False, on_epoch=True)
        
        # Enhanced image logging with metadata
        if (batch_idx == 0 and 
            self.current_epoch % self.log_images_every_n_epochs == 0):
            self.log_images_to_wandb(inputs, targets, predictions, "train", batch_idx)
        
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
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
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
        
        # Enhanced image logging with metadata
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
        mae = torch.nn.functional.l1_loss(predictions, targets)
        rmse = torch.sqrt(torch.nn.functional.mse_loss(predictions, targets))
        
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

# Additional utility function to log dataset overview
def log_dataset_overview_to_wandb(data_module, logger):
    """Log comprehensive dataset overview with file paths and metadata"""
    if not isinstance(logger, pl.loggers.WandbLogger):
        return
    
    try:
        # Get dataset information
        data_module.setup("fit")
        train_dataset = data_module.train_dataset
        val_dataset = data_module.val_dataset
        
        # Create dataset overview table
        columns = [
            "split", "city", "tile_count", "sequence_count", 
            "available_months", "date_range", "dataset_path"
        ]
        
        table_data = []
        
        # Analyze train dataset
        train_cities = {}
        for city, tile_row, tile_col, input_months, output_months in train_dataset.tile_sequences:
            if city not in train_cities:
                train_cities[city] = {
                    'tiles': set(),
                    'months': set(),
                    'sequences': 0
                }
            train_cities[city]['tiles'].add((tile_row, tile_col))
            train_cities[city]['months'].update(input_months + output_months)
            train_cities[city]['sequences'] += 1
        
        for city, info in train_cities.items():
            months_sorted = sorted(list(info['months']))
            table_data.append([
                "train", city, len(info['tiles']), info['sequences'],
                len(info['months']), f"{months_sorted[0]} to {months_sorted[-1]}",
                str(train_dataset.dataset_root)
            ])
        
        # Analyze val dataset
        val_cities = {}
        for city, tile_row, tile_col, input_months, output_months in val_dataset.tile_sequences:
            if city not in val_cities:
                val_cities[city] = {
                    'tiles': set(),
                    'months': set(),
                    'sequences': 0
                }
            val_cities[city]['tiles'].add((tile_row, tile_col))
            val_cities[city]['months'].update(input_months + output_months)
            val_cities[city]['sequences'] += 1
        
        for city, info in val_cities.items():
            months_sorted = sorted(list(info['months']))
            table_data.append([
                "val", city, len(info['tiles']), info['sequences'],
                len(info['months']), f"{months_sorted[0]} to {months_sorted[-1]}",
                str(val_dataset.dataset_root)
            ])
        
        # Create and log table
        dataset_table = wandb.Table(columns=columns, data=table_data)
        
        logger.experiment.log({
            "dataset_overview": dataset_table,
            "dataset_summary": {
                "total_train_sequences": len(train_dataset.tile_sequences),
                "total_val_sequences": len(val_dataset.tile_sequences),
                "train_cities": len(train_cities),
                "val_cities": len(val_cities),
                "dataset_root": str(train_dataset.dataset_root),
                "band_names": train_dataset.band_names,
                "sequence_length": train_dataset.sequence_length
            }
        })
        
        print("✅ Dataset overview logged to WandB!")
        
    except Exception as e:
        print(f"Warning: Failed to log dataset overview: {e}")