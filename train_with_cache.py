#!/usr/bin/env python3
"""
Training script with cache-aware argument parsing.

This script uses the same arguments as setup_data.py to ensure perfect cache matching.
All arguments are optional and have sensible defaults.

Usage:
    python train_with_cache.py
    python train_with_cache.py --cluster 1 --debug
    python train_with_cache.py --input_length 6 --output_length 1 --gpus 2
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import LandsatLSTPredictor
from dataset import LandsatDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import wandb
from typing import List, Optional

def train_landsat_model(
    # Dataset parameters (matching setup_data.py)
    dataset_root: str = "./Data/ML",
    cluster: str = "all",
    input_sequence_length: int = 3,
    output_sequence_length: int = 1,
    train_years: Optional[List[int]] = None,
    val_years: Optional[List[int]] = None,
    test_years: Optional[List[int]] = None,
    debug_monthly_split: bool = False,
    debug_year: int = 2014,
    max_input_nodata_pct: float = 0.95,
    
    # Training parameters
    wandb_project: str = "AAAI-Project-final-tests",
    learning_rate: float = 0.001,
    batch_size: int = 2,
    max_epochs: int = 3,
    num_workers: int = 8,
    gpus: int = 1,
    precision: int = 16,
    model_size: str = "medium",
    limit_train_batches: float = 1.0,
    limit_val_batches: float = 1.0,
    limit_test_batches: float = 1.0,
):
    """
    Train Landsat LST prediction model with cache-aware configuration.
    
    Args:
        Dataset parameters match setup_data.py exactly for cache compatibility.
        Training parameters control the actual training process.
    """
    
    # Generate appropriate tags
    if debug_monthly_split:
        wandb_tags = [
            "landsat", "lst-prediction", "earthformer", 
            "debug-monthly-split", f"model-{model_size}",
            f"cluster-{cluster}"
        ]
    else:
        wandb_tags = [
            "landsat", "lst-prediction", "earthformer", 
            "year-based-split", f"model-{model_size}",
            f"cluster-{cluster}"
        ]
    
    # Create directories
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    print(f"🔧 Training Configuration:")
    print(f"   Dataset: {dataset_root}")
    print(f"   Cluster: {cluster}")
    print(f"   Sequence: {input_sequence_length} → {output_sequence_length}")
    print(f"   Model: {model_size}")
    print(f"   GPUs: {gpus}")
    print(f"   Batch size: {batch_size}")
    print(f"   Max epochs: {max_epochs}")
    if debug_monthly_split:
        print(f"   Debug mode: Monthly splits in {debug_year}")
    else:
        print(f"   Year splits: train={train_years}, val={val_years}, test={test_years}")
    print()
    
    # Initialize data module with exact cache-matching parameters
    data_module = LandsatDataModule(
        dataset_root=dataset_root,
        cluster=cluster,
        batch_size=batch_size,
        num_workers=num_workers,
        input_sequence_length=input_sequence_length,
        output_sequence_length=output_sequence_length,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        debug_monthly_split=debug_monthly_split,
        debug_year=debug_year,
        interpolated_scenes_file=os.path.join(dataset_root, "interpolated.txt"),
        max_input_nodata_pct=max_input_nodata_pct,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches
    )
    
    # Initialize Weights & Biases logger
    logger = WandbLogger(
        project=wandb_project,
        tags=wandb_tags,
        save_dir="./logs",
        log_model=True,
    )
    
    # Create run-specific checkpoint directory
    run_name = logger.experiment.name if hasattr(logger.experiment, 'name') else f"run_{logger.version}"
    checkpoint_dir = f"./checkpoints/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"🏷️ Run name: {run_name}")
    print(f"📁 Checkpoints: {checkpoint_dir}")
    print()
    
    # Initialize model
    model = LandsatLSTPredictor(
        learning_rate=learning_rate,
        weight_decay=1e-5,
        warmup_steps=1000,
        max_epochs=max_epochs,
        input_sequence_length=input_sequence_length,
        output_sequence_length=output_sequence_length,
        model_size=model_size
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch:02d}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15 if not debug_monthly_split else 10,
        mode='min',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Configure strategy for multi-GPU
    if gpus > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            static_graph=True,
            gradient_as_bucket_view=True
        )
    else:
        strategy = None  # Let PyTorch Lightning choose automatically
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if gpus > 0 else 'cpu',
        strategy=strategy,
        devices=gpus if gpus > 0 else None,
        precision=precision,
        accumulate_grad_batches=1,
        val_check_interval=0.5,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=logger,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
        benchmark=True,
    )
    
    try:
        # Start training
        print("🚀 Starting training...")
        trainer.fit(model, data_module)
        
        # Run test
        print("\n🧪 Running final test...")
        try:
            test_results = trainer.test(model, data_module, ckpt_path='best')
            print(f"✅ Test completed: {test_results}")
        except Exception as e:
            print(f"⚠️ Test failed: {e}")
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Best model: {checkpoint_callback.best_model_path}")
        
        if wandb.run is not None:
            print(f"🔗 View experiment: {wandb.run.url}")
        
        # Save checkpoint to WandB
        if checkpoint_callback.best_model_path and wandb.run is not None:
            try:
                wandb.save(checkpoint_callback.best_model_path)
                print(f"✅ Model saved to WandB")
            except Exception as e:
                print(f"⚠️ Failed to save to WandB: {e}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print(f"📁 Last checkpoint: {checkpoint_callback.last_model_path}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        if wandb.run is not None:
            try:
                wandb.log({"error": str(e)})
            except:
                pass
        raise
    
    finally:
        try:
            if wandb.run is not None:
                wandb.finish()
        except:
            pass
    
    return trainer, model, data_module

def main():
    parser = argparse.ArgumentParser(description="Train Landsat LST prediction model with cache awareness")
    
    # Dataset parameters (EXACT same as setup_data.py)
    parser.add_argument("--dataset_root", type=str, default="./Data/ML",
                        help="Root directory of the tiled dataset")
    parser.add_argument("--cluster", type=str, default="all", 
                        help="Cluster to use (1, 2, 3, 4, or 'all')")
    
    # Sequence parameters
    parser.add_argument("--input_length", type=int, default=3,
                        help="Input sequence length")
    parser.add_argument("--output_length", type=int, default=1,
                        help="Output sequence length")
    
    # Year configuration
    parser.add_argument("--debug", action="store_true",
                        help="Use debug monthly splits within a single year")
    parser.add_argument("--debug_year", type=int, default=2014,
                        help="Year to use for debug monthly splits")
    
    # Custom year splits (ignored if --debug is used)
    parser.add_argument("--train_years", type=int, nargs="*",
                        default=[2013,2014,2015],
                        help="Years for training split")
    parser.add_argument("--val_years", type=int, nargs="*", 
                        default=[2016],
                        help="Years for validation split")
    parser.add_argument("--test_years", type=int, nargs="*",
                        default=[2017], 
                        help="Years for test split")
    
    # Quality filtering
    parser.add_argument("--max_nodata", type=float, default=0.95,
                        help="Maximum input NODATA percentage (0.0-1.0)")
    
    # Training parameters
    parser.add_argument("--wandb_project", type=str, default="AAAI-Project-final-tests",
                        help="Weights & Biases project name")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=3,
                        help="Maximum number of epochs")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of dataloader workers")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to use")
    parser.add_argument("--precision", type=int, default=16, choices=[16, 32],
                        help="Training precision")
    parser.add_argument("--model_size", type=str, default="medium",
                        choices=["tiny", "small", "medium", "large"],
                        help="Model size")
    
    # Data limiting (for debugging)
    parser.add_argument("--limit_train_batches", type=float, default=1.0,
                        help="Fraction of training data to use")
    parser.add_argument("--limit_val_batches", type=float, default=1.0,
                        help="Fraction of validation data to use")
    parser.add_argument("--limit_test_batches", type=float, default=1.0,
                        help="Fraction of test data to use")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0.0 <= args.max_nodata <= 1.0:
        parser.error("--max_nodata must be between 0.0 and 1.0")
    
    if not 0.0 < args.limit_train_batches <= 1.0:
        parser.error("--limit_train_batches must be between 0.0 and 1.0")
    
    # Convert debug flag to debug_monthly_split
    debug_monthly_split = args.debug
    
    # Set years to None if using debug mode
    train_years = None if debug_monthly_split else args.train_years
    val_years = None if debug_monthly_split else args.val_years
    test_years = None if debug_monthly_split else args.test_years
    
    print("=" * 60)
    print("🚀 LANDSAT LST PREDICTION TRAINING")
    print("=" * 60)
    
    try:
        trainer, model, data_module = train_landsat_model(
            # Dataset parameters (cache-compatible)
            dataset_root=args.dataset_root,
            cluster=args.cluster,
            input_sequence_length=args.input_length,
            output_sequence_length=args.output_length,
            train_years=train_years,
            val_years=val_years,
            test_years=test_years,
            debug_monthly_split=debug_monthly_split,
            debug_year=args.debug_year,
            max_input_nodata_pct=args.max_nodata,
            
            # Training parameters
            wandb_project=args.wandb_project,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            num_workers=args.num_workers,
            gpus=args.gpus,
            precision=args.precision,
            model_size=args.model_size,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
            limit_test_batches=args.limit_test_batches,
        )
        
        print("\n🎉 All done! Check your WandB dashboard for results.")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()