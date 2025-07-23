#!/usr/bin/env python3
"""
Data preprocessing script for Landsat LST prediction.

This script builds and caches tile sequences for train/val/test splits.
Run this once before training to avoid expensive setup during each training run.

Usage:
    python setup_data.py
    python setup_data.py --cluster 1 --debug
    python setup_data.py --splits train val
"""

import os
import sys
import argparse
import pickle
import time
from pathlib import Path
from typing import List, Optional

# Add current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_dataset import LandsatSequenceDataset

def get_cache_filename_for_dataset(dataset_instance) -> str:
    """Generate cache filename matching the dataset's configuration"""
    return get_cache_filename(
        dataset_root=str(dataset_instance.dataset_root),
        split=dataset_instance.split,
        cluster=dataset_instance.cluster,
        input_seq_len=dataset_instance.input_sequence_length,
        output_seq_len=dataset_instance.output_sequence_length,
        train_years=getattr(dataset_instance, 'train_years', None),
        val_years=getattr(dataset_instance, 'val_years', None),
        test_years=getattr(dataset_instance, 'test_years', None),
        debug_monthly_split=dataset_instance.debug_monthly_split,
        debug_year=dataset_instance.debug_year,
        max_input_nodata_pct=dataset_instance.max_input_nodata_pct
    )

def get_cache_filename(
    dataset_root: str,
    split: str, 
    cluster: str = "all",
    input_seq_len: int = 3,
    output_seq_len: int = 3,
    train_years: Optional[List[int]] = None,
    val_years: Optional[List[int]] = None,
    test_years: Optional[List[int]] = None,
    debug_monthly_split: bool = False,
    debug_year: int = 2014,
    max_input_nodata_pct: float = 0.60
) -> str:
    """Generate a unique cache filename based on dataset parameters"""
    
    # Create a hash of the configuration
    import hashlib
    
    config_str = f"{input_seq_len}_{output_seq_len}"
    config_str += f"_{test_years}"
    config_str += f"_{max_input_nodata_pct}"
    
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    cache_dir = Path(dataset_root) / "test_sequence_cache"
    cache_dir.mkdir(exist_ok=True)
    return str(cache_dir / f"test_sequences_{config_hash}.pkl")

def build_split_cache(
    dataset_root: str,
    split: str,
    cluster: str = "all",
    input_sequence_length: int = 3,
    output_sequence_length: int = 3,
    train_years: Optional[List[int]] = None,
    val_years: Optional[List[int]] = None,
    test_years: Optional[List[int]] = None,
    debug_monthly_split: bool = False,
    debug_year: int = 2014,
    max_input_nodata_pct: float = 0.60,
    force_rebuild: bool = False
) -> str:
    """
    Build and cache tile sequences for a single split.
    
    Returns:
        str: Path to the cache file
    """
    
    # Generate cache filename
    cache_file = get_cache_filename(
        dataset_root=dataset_root,
        split=split,
        cluster=cluster,
        input_seq_len=input_sequence_length,
        output_seq_len=output_sequence_length,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        debug_monthly_split=debug_monthly_split,
        debug_year=debug_year,
        max_input_nodata_pct=max_input_nodata_pct
    )
    
    # Check if cache already exists
    if os.path.exists(cache_file) and not force_rebuild:
        try:
            with open(cache_file, 'rb') as f:
                sequences = pickle.load(f)
            print(f"✅ {split.upper()} cache already exists: {len(sequences)} sequences")
            print(f"   Cache file: {cache_file}")
            return cache_file
        except Exception as e:
            print(f"⚠️ Failed to load existing cache ({e}), rebuilding...")
    
    # Build sequences using dataset (with parallelism)
    print(f"\n🔄 Building {split.upper()} sequences...")
    start_time = time.time()
    
    # Temporarily force parallelism regardless of environment
    original_local_rank = os.environ.get('LOCAL_RANK')
    if 'LOCAL_RANK' in os.environ:
        del os.environ['LOCAL_RANK']  # Remove DDP detection
    
    try:
        dataset = LandsatSequenceDataset(
            dataset_root=dataset_root,
            cluster=cluster,
            input_sequence_length=input_sequence_length,
            output_sequence_length=output_sequence_length,
            split=split,
            train_years=train_years,
            val_years=val_years,
            test_years=test_years,
            debug_monthly_split=debug_monthly_split,
            debug_year=debug_year,
            max_input_nodata_pct=max_input_nodata_pct
        )
        
        sequences = dataset.tile_sequences
        
    finally:
        # Restore original environment
        if original_local_rank is not None:
            os.environ['LOCAL_RANK'] = original_local_rank
    
    # Save to cache
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(sequences, f)
    
    elapsed = time.time() - start_time
    print(f"✅ {split.upper()} sequences built and cached: {len(sequences)} sequences")
    print(f"   Time: {elapsed:.1f} seconds")
    print(f"   Cache file: {cache_file}")
    
    return cache_file

def build_all_caches(
    dataset_root: str,
    splits: List[str] = ["train", "val", "test"],
    cluster: str = "all",
    input_sequence_length: int = 3,
    output_sequence_length: int = 3,
    train_years: Optional[List[int]] = None,
    val_years: Optional[List[int]] = None, 
    test_years: Optional[List[int]] = None,
    debug_monthly_split: bool = False,
    debug_year: int = 2014,
    max_input_nodata_pct: float = 0.60,
    force_rebuild: bool = False
) -> dict:
    """
    Build and cache tile sequences for all specified splits.
    
    Returns:
        dict: Mapping of split names to cache file paths
    """
    
    print("=" * 60)
    print("🚀 LANDSAT DATASET PREPROCESSING")
    print("=" * 60)
    print(f"Dataset root: {dataset_root}")
    print(f"Cluster: {cluster}")
    print(f"Splits to process: {splits}")
    print(f"Sequence lengths: {input_sequence_length} → {output_sequence_length}")
    if debug_monthly_split:
        print(f"Debug mode: Monthly splits within {debug_year}")
    else:
        print(f"Year splits: train={train_years}, val={val_years}, test={test_years}")
    print(f"Max input NODATA: {max_input_nodata_pct:.1%}")
    print(f"Force rebuild: {force_rebuild}")
    print()
    
    # Verify dataset exists
    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    
    for cluster in ["1", "2", "3", "4"]:
        cities_tiles = dataset_path /"Clustered" / cluster / "Cities_Tiles" 
        if not cities_tiles.exists():
            raise FileNotFoundError(f"Cities tiles not found: {cities_tiles}")

    dem_tiles = dataset_path / "DEM_2014_Tiles"
    if not dem_tiles.exists():
        raise FileNotFoundError(f"DEM tiles not found: {dem_tiles}")
    
    print(f"✅ Dataset directories verified")
    print()
    
    # Build caches for each split
    cache_files = {}
    total_start = time.time()
    
    for split in splits:
        try:
            cache_file = build_split_cache(
                dataset_root=dataset_root,
                split=split,
                cluster=cluster,
                input_sequence_length=input_sequence_length,
                output_sequence_length=output_sequence_length,
                train_years=train_years,
                val_years=val_years,
                test_years=test_years,
                debug_monthly_split=debug_monthly_split,
                debug_year=debug_year,
                max_input_nodata_pct=max_input_nodata_pct,
                force_rebuild=force_rebuild
            )
            cache_files[split] = cache_file
            
        except Exception as e:
            print(f"❌ Failed to build {split} cache: {e}")
            raise
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PREPROCESSING SUMMARY")
    print("=" * 60)
    
    total_sequences = 0
    for split in splits:
        if split in cache_files:
            try:
                with open(cache_files[split], 'rb') as f:
                    sequences = pickle.load(f)
                count = len(sequences)
                total_sequences += count
                print(f"{split.upper():>5}: {count:>8,} sequences")
            except Exception as e:
                print(f"{split.upper():>5}: Error loading cache ({e})")
    
    print(f"{'TOTAL':>5}: {total_sequences:>8,} sequences")
    print(f"Time: {total_elapsed:.1f} seconds")
    print()
    print("🎉 Preprocessing complete! You can now run training with fast dataset loading.")
    print()
    
    return cache_files

def main():
    parser = argparse.ArgumentParser(description="Preprocess Landsat dataset sequences")
    
    # Dataset parameters
    parser.add_argument("--dataset_root", type=str, default="./Data/ML/Clustered",
                        help="Root directory of the tiled dataset")
    parser.add_argument("--cluster", type=str, default="all", 
                        help="Cluster to use (1, 2, 3, 4, or 'all')")
    
    # Sequence parameters
    parser.add_argument("--input_length", type=int, default=3,
                        help="Input sequence length")
    parser.add_argument("--output_length", type=int, default=3,
                        help="Output sequence length")
    
    # Split selection
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test"],
                        help="Which splits to process")
    
    # Year configuration
    parser.add_argument("--debug", action="store_true",
                        help="Use debug monthly splits within a single year")
    parser.add_argument("--debug_year", type=int, default=2014,
                        help="Year to use for debug monthly splits")
    
    # Custom year splits (ignored if --debug is used)
    parser.add_argument("--train_years", type=int, nargs="*",
                        default=[2013,2014,2015,2016,2017,2018,2019,2020,2021],
                        help="Years for training split")
    parser.add_argument("--val_years", type=int, nargs="*", 
                        default=[2022,2023],
                        help="Years for validation split")
    parser.add_argument("--test_years", type=int, nargs="*",
                        default=[2024,2025], 
                        help="Years for test split")
    
    # Quality filtering
    parser.add_argument("--max_nodata", type=float, default=0.60,
                        help="Maximum input NODATA percentage (0.0-1.0)")
    
    # Control options
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild even if caches exist")
    parser.add_argument("--check_only", action="store_true",
                        help="Only check if caches exist, don't build")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0.0 <= args.max_nodata <= 1.0:
        parser.error("--max_nodata must be between 0.0 and 1.0")
    
    if args.check_only:
        # Just check cache status
        print("🔍 Checking cache status...")
        for split in args.splits:
            cache_file = get_cache_filename(
                dataset_root=args.dataset_root,
                split=split,
                cluster=args.cluster,
                input_seq_len=args.input_length,
                output_seq_len=args.output_length,
                train_years=args.train_years if not args.debug else None,
                val_years=args.val_years if not args.debug else None,
                test_years=args.test_years if not args.debug else None,
                debug_monthly_split=args.debug,
                debug_year=args.debug_year,
                max_input_nodata_pct=args.max_nodata
            )
            
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        sequences = pickle.load(f)
                    print(f"✅ {split.upper()}: {len(sequences):,} sequences cached")
                except:
                    print(f"❌ {split.upper()}: Cache file corrupted")
            else:
                print(f"⚪ {split.upper()}: No cache found")
        return
    
    # Build caches
    try:
        cache_files = build_all_caches(
            dataset_root=args.dataset_root,
            splits=args.splits,
            cluster=args.cluster,
            input_sequence_length=args.input_length,
            output_sequence_length=args.output_length,
            train_years=args.train_years if not args.debug else None,
            val_years=args.val_years if not args.debug else None,
            test_years=args.test_years if not args.debug else None,
            debug_monthly_split=args.debug,
            debug_year=args.debug_year,
            max_input_nodata_pct=args.max_nodata,
            force_rebuild=args.force
        )
        
        # Print next steps
        print("Next steps:")
        print("  python train.py  # Fast training with cached data")
        print("  python experiment.py --agent <sweep_id>  # Fast sweep runs")
        
    except KeyboardInterrupt:
        print("\n⚠️ Preprocessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()