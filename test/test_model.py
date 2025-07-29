import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
from sklearn.metrics import r2_score
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from model import LandsatLSTPredictor

class PersonalizedLandsatLSTPredictor(pl.LightningModule):
    def __init__(
            self,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-5,
            warmup_steps: int = 1000,
            max_epochs: int = 100,
            checkpoint_dir: str = "/root/lst-earthformer/Personalized/Earthnet_No_Aux",
            use_all: bool = False,
            log_images_every_n_epochs: int = 1,
            max_images_to_log: int = 4,
            input_sequence_length: int = 3,
            output_sequence_length: int = 3,
            model_size: str = "small",  # NEW: "tiny", "small", "medium", "large", "earthnet"
            **model_kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.use_all = use_all # If true use one model for all clusters, otherwise if false use individual cluster models.

        # Store sequence lengths
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length

        # Image logging parameters
        self.log_images_every_n_epochs = log_images_every_n_epochs
        self.max_images_to_log = max_images_to_log

        self.train_predictions = []
        self.train_targets = []
        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []
        self.cluster_models = {}
        self.checkpoint_dir = Path(checkpoint_dir)
        self._load_cluster_models()
        self.cluster_predictions = {"1": [], "2": [], "3": [], "4": []}
        self.cluster_targets = {"1": [], "2": [], "3": [], "4": []}

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
                'base_units': 128,  # Keep same
                'num_heads': 8,  # Keep same
                'enc_depth': [2, 2],  # REDUCED from [3, 3]
                'dec_depth': [1, 1],  # REDUCED from [2, 2]
                'enc_cuboid_size': [(2, 4, 4), (2, 4, 4)],  # REDUCED from [(2, 4, 4), (2, 8, 8)]
                'num_global_vectors': 12,  # REDUCED from 16
            },
            "large": {
                'base_units': 144,  # REDUCED from 192
                'num_heads': 8,  # REDUCED from 12 (must divide base_units)
                'enc_depth': [2, 2],  # REDUCED from [4, 4]
                'dec_depth': [1, 1],  # REDUCED from [3, 3]
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

    def _load_cluster_models(self):
        """Load all 4 cluster-specific models"""
        for cluster_id in ["1", "2", "3", "4", "all"]:
            checkpoint_path = self.checkpoint_dir / f"{cluster_id}.ckpt"
            if checkpoint_path.exists():
                try:
                    model = LandsatLSTPredictor.load_from_checkpoint(
                        str(checkpoint_path),
                        strict=False
                    )
                    model.eval()
                    model.freeze()
                    self.cluster_models[cluster_id] = model
                    print(f"✅ Loaded cluster {cluster_id} model")
                except Exception as e:
                    print(f"❌ Failed to load cluster {cluster_id}: {e}")

    def _get_cluster_ids(self, batch_idx: int, batch_size: int) -> List[str]:
        """Extract cluster IDs for current batch"""
        dataset = self.trainer.datamodule.test_dataset
        cluster_ids = []
        for i in range(batch_size):
            sample_idx = batch_idx * batch_size + i
            if sample_idx < len(dataset.tile_sequences):
                cluster = dataset.tile_sequences[sample_idx][0]  # First element is cluster
                cluster_ids.append(cluster)
        if len(cluster_ids) == 0:
            raise RuntimeError("No cluster IDs found")
        return cluster_ids


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        return self.model(x)

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
                        cluster, city, tile_row, tile_col, input_months, output_months = dataset.tile_sequences[sample_idx]

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
                            'cluster': cluster
                            # 'file_paths': self._get_file_paths(city, tile_row, tile_col, input_months + output_months)
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
        """Test step with cluster-specific model routing and proper cluster metrics aggregation"""
        self.eval()
        inputs, targets = batch
        batch_size = 1  # batch size is always one to keep cluster calculations simple

        # Get cluster IDs for this batch
        cluster_ids = self._get_cluster_ids(batch_idx, batch_size)

        # Route predictions through cluster-specific models
        predictions = torch.zeros_like(targets)

        for i, cluster_id in enumerate(cluster_ids):
            if self.use_all:
                cluster_id = "all"  # just use the one all model
            if cluster_id in self.cluster_models:
                sample_input = inputs[i:i + 1]  # Single sample
                with torch.no_grad():
                    sample_pred = self.cluster_models[cluster_id](sample_input)
                predictions[i:i + 1] = sample_pred
        # Calculate metrics in Fahrenheit
        with torch.no_grad():
            # Denormalize to Fahrenheit: value * (max - min) + min
            pred_fahrenheit = predictions.detach() * (211.0 - (-189.0)) + (-189.0)
            true_fahrenheit = targets.detach() * (211.0 - (-189.0)) + (-189.0)

            # Store cluster-specific predictions and targets for end-of-epoch aggregation
            current_cluster = cluster_ids[0]
            if current_cluster in ["1", "2", "3", "4"]:
                # Store predictions and targets for this cluster
                self.cluster_predictions[current_cluster].append(pred_fahrenheit.detach().cpu())
                self.cluster_targets[current_cluster].append(true_fahrenheit.detach().cpu())
                self.val_predictions.append(pred_fahrenheit.detach().cpu())
                self.val_targets.append(true_fahrenheit.detach().cpu())
        loss = self.masked_loss(predictions, targets)
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
        """Test step with cluster-specific model routing and proper cluster metrics aggregation"""
        self.eval()
        inputs, targets = batch
        batch_size = 1  # batch size is always one to keep cluster calculations simple

        # Get cluster IDs for this batch
        cluster_ids = self._get_cluster_ids(batch_idx, batch_size)

        # Route predictions through cluster-specific models
        predictions = torch.zeros_like(targets)

        for i, cluster_id in enumerate(cluster_ids):
            if self.use_all:
                cluster_id = "all"  # just use the one all model
            if cluster_id in self.cluster_models:
                sample_input = inputs[i:i + 1]  # Single sample
                with torch.no_grad():
                    sample_pred = self.cluster_models[cluster_id](sample_input)
                predictions[i:i + 1] = sample_pred
        # Calculate metrics in Fahrenheit
        with torch.no_grad():
            # Denormalize to Fahrenheit: value * (max - min) + min
            pred_fahrenheit = predictions.detach() * (211.0 - (-189.0)) + (-189.0)
            true_fahrenheit = targets.detach() * (211.0 - (-189.0)) + (-189.0)

            # Store cluster-specific predictions and targets for end-of-epoch aggregation
            current_cluster = cluster_ids[0]
            if current_cluster in ["1", "2", "3", "4"]:
                # Store predictions and targets for this cluster
                self.cluster_predictions[current_cluster].append(pred_fahrenheit.detach().cpu())
                self.cluster_targets[current_cluster].append(true_fahrenheit.detach().cpu())
                self.test_predictions.append(pred_fahrenheit.detach().cpu())
                self.test_targets.append(true_fahrenheit.detach().cpu())
        loss = self.masked_loss(predictions, targets)
        return loss

    def on_test_epoch_end(self):
        """Calculate cluster-specific metrics"""
        # Only calculate and log from rank 0 in distributed training
        if self.trainer.is_global_zero:
            # Calculate cluster-specific metrics
            for cluster_id in ["1", "2", "3", "4"]:
                if len(self.cluster_predictions[cluster_id]) > 0 and len(self.cluster_targets[cluster_id]) > 0:
                    # Concatenate all predictions and targets for this cluster
                    cluster_preds = torch.cat(self.cluster_predictions[cluster_id])
                    cluster_targets = torch.cat(self.cluster_targets[cluster_id])
                    cluster_mae = self.masked_mae(cluster_preds, cluster_targets)
                    cluster_rmse = torch.sqrt(self.masked_loss(cluster_preds, cluster_targets))
                    if wandb.run is not None:
                        wandb.log({
                            f'test_mae_F_C{cluster_id}': cluster_mae.item(),
                            f'test_rmse_F_C{cluster_id}': cluster_rmse.item()
                        })

            # Calculate overall test metrics
            if len(self.test_predictions) > 0 and len(self.test_targets) > 0:
                all_preds = torch.cat(self.test_predictions)
                all_targets = torch.cat(self.test_targets)
                test_mae_F = self.masked_mae(all_preds, all_targets)
                test_rmse_F = torch.sqrt(self.masked_loss(all_preds, all_targets))

                if wandb.run is not None:
                    wandb.log({
                        'test_mae_F': test_mae_F.item(),
                        'test_rmse_F': test_rmse_F.item()
                    })

        # Clear stored data on all ranks
        self.test_predictions = []
        self.test_targets = []
        # Clear cluster-specific data
        for cluster_id in ["1", "2", "3", "4"]:
            self.cluster_predictions[cluster_id] = []
            self.cluster_targets[cluster_id] = []

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
        """Calculate cluster-specific metrics"""
        # Only calculate and log from rank 0 in distributed training
        if self.trainer.is_global_zero:
            # Calculate cluster-specific metrics
            for cluster_id in ["1", "2", "3", "4"]:
                if len(self.cluster_predictions[cluster_id]) > 0 and len(self.cluster_targets[cluster_id]) > 0:
                    # Concatenate all predictions and targets for this cluster
                    cluster_preds = torch.cat(self.cluster_predictions[cluster_id])
                    cluster_targets = torch.cat(self.cluster_targets[cluster_id])
                    cluster_mae = self.masked_mae(cluster_preds, cluster_targets)
                    cluster_rmse = torch.sqrt(self.masked_loss(cluster_preds, cluster_targets))
                    if wandb.run is not None:
                        wandb.log({
                            f'val_mae_F_C{cluster_id}': cluster_mae.item(),
                            f'val_rmse_F_C{cluster_id}': cluster_rmse.item()
                        })

            # Calculate overall test metrics
            if len(self.test_predictions) > 0 and len(self.test_targets) > 0:
                all_preds = torch.cat(self.test_predictions)
                all_targets = torch.cat(self.test_targets)
                val_mae_F = self.masked_mae(all_preds, all_targets)
                val_rmse_F = torch.sqrt(self.masked_loss(all_preds, all_targets))

                if wandb.run is not None:
                    wandb.log({
                        'val_mae_F': val_mae_F.item(),
                        'val_rmse_F': val_rmse_F.item()
                    })

        # Clear stored data on all ranks
        self.val_predictions = []
        self.val_targets = []
        # Clear cluster-specific data
        for cluster_id in ["1", "2", "3", "4"]:
            self.cluster_predictions[cluster_id] = []
            self.cluster_targets[cluster_id] = []
