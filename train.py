import os
import torch
from model import LandsatLSTPredictor
from dataset import LandsatDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from typing import List, Optional
from pytorch_lightning.strategies import DDPStrategy

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
    
    # Create unique run directory - will be set after logger initialization
    log_dir = "./logs"
    
    print(f"✅ Found tiled dataset at {dataset_root}")
    
    # Initialize data module
    data_module = LandsatDataModule(
        dataset_root=dataset_root,
        cluster=config["cluster"],
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
    
    # BETTER: Create run-specific checkpoint directory after logger is initialized
    # This way each run gets its own checkpoint folder
    run_name = logger.experiment.name if hasattr(logger.experiment, 'name') else f"run_{logger.version}"
    checkpoint_dir = f"./checkpoints/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"🏷️ Run name: {run_name}")
    print(f"📁 Checkpoints will be saved to: {checkpoint_dir}")
    
    # Initialize model with configurable size
    model = LandsatLSTPredictor(
        learning_rate=config["learning_rate"],
        weight_decay=1e-5,
        warmup_steps=1000,
        max_epochs=config["max_epochs"],
        input_sequence_length=config["input_sequence_length"],
        output_sequence_length=config["output_sequence_length"],
        model_size=config.get("model_size", "small")
    )
    
    # SIMPLE: Use standard PyTorch Lightning checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch:02d}',  # Use 2 decimal places instead of 3
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
    
    # DDP strategy
    ddp_strategy = DDPStrategy(
        find_unused_parameters=False,
        static_graph=True,
        gradient_as_bucket_view=True
    )
    
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator='gpu' if config["gpus"] > 0 else 'cpu',
        strategy=ddp_strategy,
        devices=config["gpus"] if config["gpus"] > 0 else None,
        precision=config["precision"],
        accumulate_grad_batches=1,
        val_check_interval=0.5,
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
            # STANDARD: Use 'best' to automatically load the best checkpoint
            test_results = trainer.test(model, data_module, ckpt_path='best')
            print(f"✅ Test completed: {test_results}")
        except Exception as e:
            print(f"⚠️ Test failed: {e}")
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Best model saved to: {checkpoint_callback.best_model_path}")
        
        # Get the actual wandb URL
        if wandb.run is not None:
            wandb_url = wandb.run.url
            print(f"🔗 View experiment at: {wandb_url}")
        else:
            print(f"🔗 View experiment in your wandb project: {wandb_project}")
        
        # Save checkpoint to WandB
        if checkpoint_callback.best_model_path and wandb.run is not None:
            try:
                wandb.save(checkpoint_callback.best_model_path)
                print(f"✅ Model checkpoint saved to WandB")
            except Exception as e:
                print(f"⚠️ Failed to save checkpoint to WandB: {e}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print(f"📁 Last checkpoint saved to: {checkpoint_callback.last_model_path}")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Log error before finishing WandB
        if wandb.run is not None:
            try:
                wandb.log({"error": str(e)})
            except Exception as log_error:
                print(f"⚠️ Failed to log error to WandB: {log_error}")
        
        raise
    
    finally:
        # Always finish WandB properly
        try:
            if wandb.run is not None:
                wandb.finish()
        except Exception as finish_error:
            print(f"⚠️ Error finishing WandB: {finish_error}")
    
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
    wandb_project = "AAAI-Project-final-tests"
    dataset_root = "./Data/ML"  
    hyperparameters = {
        "learning_rate": 0.001,
        "gpus": 2,
        "precision": 16,
        "debug_monthly_split": True,
        "debug_year": 2014,
        "batch_size": 2, # Get batch size
        "max_epochs": 3,
        "num_workers": 8,
        "input_sequence_length": 3,
        "output_sequence_length": 1,
        "model_size": "medium",  # "tiny", "small", "medium", "large"
        "train_years": [2013,2014,2015,2016,2017,2018,2019,2020,2021],
        "val_years": [2022,2023],
        "test_years": [2024,2025],
        "use_custom_years": True,
        "limit_train_batches": 0.1,
        "limit_val_batches": 0.1,
        "limit_test_batches": 0.1,
        "max_input_nodata_pct": 0.95,
        "cluster": "all" #1,2,3,4, all
    }
    
    train_landsat_model(wandb_project, dataset_root, hyperparameters)