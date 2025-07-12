import os
import torch
from model import LandsatLSTPredictor
from dataset import LandsatDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from typing import List, Optional

def train_landsat_model(wandb_project: str, dataset_root: str, config: dict):
    if config["debug_monthly_split"]:
        wandb_tags = [
            "landsat", "lst-prediction", "earthformer", 
            "debug-monthly-split", f"model-{config.get('model_size', 'small')}"
        ]
    else:
        wandb_tags = [
            "landsat", "lst-prediction", "earthformer", 
            "year-based-split", f"model-{config.get('model_size', 'small')}"
        ]
    
    # Create directories
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    checkpoint_dir = "./checkpoints"
    log_dir = "./logs"
    
    print(f"✅ Found tiled dataset at {dataset_root}")
    
    # Initialize data module
    data_module = LandsatDataModule(
        dataset_root=dataset_root,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        input_sequence_length=config["input_sequence_length"],
        output_sequence_length=config["output_sequence_length"],
        train_years=config["train_years"],
        val_years=config["val_years"],
        test_years=config["test_years"],
        debug_monthly_split=config["debug_monthly_split"],
        debug_year=config["debug_year"],
        interpolated_scenes_file="./Data/ML/interpolated.txt",
        max_input_nodata_pct=config["max_input_nodata_pct"]
    )
    
    # Initialize Weights & Biases logger
    logger = WandbLogger(
        project=wandb_project,
        tags=wandb_tags,
        config=config,
        save_dir=log_dir,
        log_model=True,
    )
    
    # Initialize model with configurable size
    model = LandsatLSTPredictor(
        learning_rate=config["learning_rate"],
        weight_decay=1e-5,
        warmup_steps=1000,
        max_epochs=config["max_epochs"],
        input_sequence_length=config["input_sequence_length"],
        output_sequence_length=config["output_sequence_length"],
        model_size=config.get("model_size", "small")  # NEW: configurable model size
    )
    
    # Rest of the function remains the same...
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{logger.experiment.name}-{{epoch:02d}}-{{val_loss:.3f}}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15 if not config["debug_monthly_split"] else 10,
        mode='min',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator='gpu' if config["gpus"] > 0 else 'cpu',
        devices=config["gpus"] if config["gpus"] > 0 else None,
        precision=config["precision"],
        accumulate_grad_batches=1,
        val_check_interval=1.0,
        limit_train_batches=config["limit_train_batches"],
        limit_val_batches=config["limit_val_batches"],
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=logger,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
        benchmark=True,
    )
    
    try:
        trainer.fit(model, data_module)
        
        print("\n🧪 Running final test...")
        try:
            test_results = trainer.test(model, data_module, ckpt_path='best')
            print(f"✅ Test completed: {test_results}")
        except Exception as e:
            print(f"⚠️ Test failed (this is okay if no test data): {e}")
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Best model saved to: {checkpoint_callback.best_model_path}")
        print(f"🔗 View experiment at: {logger.experiment.url}")
        
        if checkpoint_callback.best_model_path:
            wandb.save(checkpoint_callback.best_model_path)
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print(f"📁 Last checkpoint saved to: {checkpoint_callback.last_model_path}")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        if 'logger' in locals():
            wandb.log({"error": str(e)})
        
        raise
    
    finally:
        if 'logger' in locals():
            wandb.finish()
    
    return trainer, model, data_module

"""    
    Hyperparameters:
        dataset_root: Path to preprocessed dataset with Cities_Tiles and DEM_2014_Tiles
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        learning_rate: Initial learning rate
        num_workers: Number of data loading workers
        gpus: Number of GPUs to use
        precision: Training precision ('32', '16', or 'mixed')
        limit_train_batches: Fraction of training data to use (for debugging)
        limit_val_batches: Fraction of validation data to use (for debugging)
        experiment_name: Name for logging
        checkpoint_dir: Directory to save checkpoints
        train_years: Years to use for training (if None, uses default 70/15/15 split)
        val_years: Years to use for validation
        test_years: Years to use for testing
        use_custom_years: Whether to use custom year splits in experiment name
        debug_monthly_split: If True, use monthly splits within debug_year for fast debugging
        debug_year: Year to use for debug monthly splits (default: 2014)
    """

if __name__ == "__main__":    
    wandb_project = "AAAI-Project"
    dataset_root = "./Data/ML"  
    minimal_hyperparameters={
        "dataset_root": dataset_root,
        "batch_size": 1,
        "max_epochs": 2,
        "learning_rate": 1e-3,
        "num_workers": 0,
        "gpus": 0,  # Use CPU for maximum compatibility
        "precision": "32",
        "limit_train_batches": 0.15,
        "limit_val_batches": 0.15,
        "wandb_project": "landsat-debug",
        "wandb_tags": ["minimal", "debug", "monthly", "cpu"],
        "debug_monthly_split": True,
        "debug_year": 2014,
        "input_sequence_length": 3,
        "output_sequence_length": 1
    }  
    hyperparameters = {
        "learning_rate": 1e-3,
        "gpus": 1,
        "precision": 16,
        "debug_monthly_split": True,
        "debug_year": 2014,
        "batch_size": 16,
        "max_epochs": 5,
        "num_workers": 8,
        "input_sequence_length": 3,
        "output_sequence_length": 1,
        "model_size": "small",  # NEW: "tiny", "small", "medium", "large"
        "train_years": [2013,2014,2015],
        "val_years": [2016],
        "test_years": [2017],
        "use_custom_years": True,
        "limit_train_batches": 0.3,
        "limit_val_batches": 0.3,
        "limit_test_batches": 0.3,
        "max_input_nodata_pct": 0.60
    }
    
    train_landsat_model(wandb_project, dataset_root, hyperparameters)