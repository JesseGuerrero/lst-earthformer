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