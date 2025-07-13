import os
import torch
from model import LandsatLSTPredictor
from dataset import LandsatDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
import argparse
import yaml
from typing import List, Optional

def load_sweep_config(config_file: str = "sweep_config.yml"):
    """Load sweep configuration from YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def initialize_sweep(config_file: str = "sweep_config.yml", project: str = "AAAI-Project-Sweep"):
    """Initialize a wandb sweep."""
    
    # Load sweep configuration
    sweep_config = load_sweep_config(config_file)
    
    print("🔧 Sweep Configuration:")
    print(yaml.dump(sweep_config, default_flow_style=False))
    
    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project
    )
    
    print(f"\n✅ Sweep initialized!")
    print(f"📊 Sweep ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/your-username/{project}/sweeps/{sweep_id}")
    print(f"\n🚀 To start sweep agents, run:")
    print(f"   python experiment.py --agent {sweep_id}")
    print(f"\n💡 To run multiple agents in parallel:")
    print(f"   # Terminal 1: python experiment.py --agent {sweep_id}")
    print(f"   # Terminal 2: python experiment.py --agent {sweep_id}")
    print(f"   # Terminal 3: python experiment.py --agent {sweep_id}")
    
    return sweep_id

def start_sweep_agent(sweep_id: str, count: int = None):
    """Start a sweep agent."""
    print(f"🤖 Starting sweep agent for sweep: {sweep_id}")
    
    if count:
        wandb.agent(sweep_id, function=train_landsat_model_sweep, count=count)
    else:
        wandb.agent(sweep_id, function=train_landsat_model_sweep)

def train_landsat_model_sweep():
    """
    Training function for wandb sweeps.
    Gets hyperparameters from wandb.config.
    """
    # Initialize wandb run (this gets the config from the sweep)
    wandb.init()
    config = wandb.config
    
    # Set project name
    dataset_root = "./Data/ML"
    
    # Determine tags based on configuration
    wandb_tags = [
        "landsat", "lst-prediction", "earthformer", "sweep",
        f"model-{config.model_size}",
        f"seq-{config.input_sequence_length}to{config.output_sequence_length}"
    ]
    
    if config.debug_monthly_split:
        wandb_tags.append("debug-monthly-split")
    else:
        wandb_tags.append("year-based-split")
    
    # Create directories
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    checkpoint_dir = "./checkpoints"
    log_dir = "./logs"
    
    print(f"✅ Found tiled dataset at {dataset_root}")
    print(f"🔧 Sweep config: {dict(config)}")
    
    # Initialize data module with sweep config
    data_module = LandsatDataModule(
        dataset_root=dataset_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        input_sequence_length=config.input_sequence_length,
        output_sequence_length=config.output_sequence_length,
        train_years=config.train_years,
        val_years=config.val_years,
        test_years=config.test_years,
        debug_monthly_split=config.debug_monthly_split,
        debug_year=config.debug_year,
        interpolated_scenes_file="./Data/ML/interpolated.txt",
        max_input_nodata_pct=config.max_input_nodata_pct,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches
    )
    
    # The logger is already initialized by wandb.init()
    logger = WandbLogger(
        save_dir=log_dir,
        log_model=False,  # Set to False for sweeps to save space
    )
    
    # Initialize model with sweep config
    model = LandsatLSTPredictor(
        learning_rate=config.learning_rate,
        weight_decay=1e-5,
        warmup_steps=1000,
        max_epochs=config.max_epochs,
        input_sequence_length=config.input_sequence_length,
        output_sequence_length=config.output_sequence_length,
        model_size=config.model_size
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{wandb.run.name}-{{epoch:02d}}-{{val_rmse_F:.3f}}',
        save_top_k=1,  # Only save best model for sweeps
        monitor='val_rmse_F',
        mode='min',
        save_last=False,  # Don't save last checkpoint for sweeps
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_rmse_F',
        patience=8 if config.debug_monthly_split else 12,
        mode='min',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator='gpu' if config.gpus > 0 else 'cpu',
        devices=config.gpus if config.gpus > 0 else None,
        precision=16,
        accumulate_grad_batches=1,
        val_check_interval=0.5,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
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
        
        # Log final metrics
        if trainer.callback_metrics:
            final_val_rmse = trainer.callback_metrics.get('val_rmse_F', float('inf'))
            final_val_mae = trainer.callback_metrics.get('val_mae_F', float('inf'))
            
            wandb.log({
                "final_val_rmse_F": final_val_rmse,
                "final_val_mae_F": final_val_mae,
                "training_completed": True
            })
        
        print(f"\n🎉 Sweep run completed successfully!")
        print(f"📁 Best model: {checkpoint_callback.best_model_path}")
        
    except Exception as e:
        print(f"\n❌ Sweep run failed: {e}")
        wandb.log({"training_completed": False, "error": str(e)})
        raise
    
    finally:
        wandb.finish()

def validate_and_cast_config(config):
    """Validate and cast wandb config to expected types."""
    casted_config = {}
    
    # Define expected types
    expected_types = {
        'learning_rate': float,
        'batch_size': int,
        'max_epochs': int,
        'num_workers': int,
        'input_sequence_length': int,
        'output_sequence_length': int,
        'gpus': int,
        'limit_train_batches': float,
        'limit_val_batches': float,
        'max_input_nodata_pct': float,
        'debug_year': int,
        'debug_monthly_split': bool,
        'precision': str,
        'model_size': str,
    }
    
    for key, value in config.items():
        if key in expected_types:
            try:
                casted_config[key] = expected_types[key](value)
            except (ValueError, TypeError) as e:
                print(f"❌ Error casting {key}={value} to {expected_types[key]}: {e}")
                raise
        elif key in ['train_years', 'val_years', 'test_years']:
            # Handle year lists - ensure they're lists of integers
            if isinstance(value, list):
                casted_config[key] = [int(year) for year in value]
            else:
                # If it's a single value, make it a list
                casted_config[key] = [int(value)]
        else:
            casted_config[key] = value
    
    return casted_config

# python experiment.py --agent bkdcaibr
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Landsat LST Prediction Model')
    parser.add_argument('--sweep-init', action='store_true', help='Initialize a new sweep')
    parser.add_argument('--agent', type=str, help='Start agent for existing sweep ID')
    parser.add_argument('--config', type=str, default='sweep_config.yml', help='Sweep config file')
    parser.add_argument('--project', type=str, default='AAAI-Project-Sweep', help='Wandb project name')
    parser.add_argument('--count', type=int, help='Number of runs for agent (default: unlimited)')
    
    args = parser.parse_args()
    
    if args.sweep_init:
        # Initialize a new sweep
        if not os.path.exists(args.config):
            print(f"❌ Config file '{args.config}' not found!")
            print("📝 Please create a sweep_config.yml file first.")
            exit(1)
        
        sweep_id = initialize_sweep(args.config, args.project)
        
        # Ask if user wants to start an agent immediately
        response = input("\n🤖 Start an agent now? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            start_sweep_agent(sweep_id, args.count)
    
    elif args.agent:
        # Start an agent for existing sweep
        start_sweep_agent(args.agent, args.count)
    