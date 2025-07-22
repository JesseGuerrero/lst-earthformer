#original
import os
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path
import glob
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import multiprocessing as mp
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import pickle
import hashlib
import random

BAND_RANGES = {
    "red": {"min": 1.0, "max": 10000.0},
    "ndwi": {"min": -10000.0, "max": 10000.0},
    "ndvi": {"min": -10000.0, "max": 10000.0},
    "ndbi": {"min": -10000.0, "max": 10000.0},
    "LST": {"min": -189.0, "max": 211.0},
    "green": {"min": 1.0, "max": 10000.0},
    "blue": {"min": 1.0, "max": 10000.0},
    "DEM": {"min": 9899.0, "max": 13110.0},
    "albedo": {"min": 1.0, "max": 9980.0}
}

class LandsatSequenceDataset(Dataset):
    def __init__(
        self, 
        dataset_root: str,
        cluster: str = "all",
        input_sequence_length: int = 3,
        output_sequence_length: int = 3,
        split: str = 'train',
        train_years: Optional[List[int]] = None,
        val_years: Optional[List[int]] = None,
        test_years: Optional[List[int]] = None,
        debug_monthly_split: bool = False,
        debug_year: int = 2014,       
        limit_batches: Optional[float] = None,
        max_input_nodata_pct: float = 0.60,
    ):
        """
        Dataset for Landsat sequence prediction using tiled data with year-based splits
        """
        self.dataset_root = Path(dataset_root)
        self.cluster = cluster
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.total_sequence_length = input_sequence_length + output_sequence_length
        self.split = split
        self.debug_monthly_split = debug_monthly_split
        self.debug_year = debug_year
        self._scene_validation_cache = {}
        self._monthly_scenes_cache = {}
        self._dem_cache = {}
        self.max_input_nodata_pct = max_input_nodata_pct
        
        if debug_monthly_split:
            self._setup_debug_monthly_splits()
        else:
            self._setup_year_based_splits(train_years, val_years, test_years)
        
        self.cities = self._get_all_cities()
        self.band_names = ['DEM', 'LST', 'red', 'green', 'blue', 'ndvi', 'ndwi', 'ndbi', 'albedo']
        
        # Build tile sequences
        self.tile_sequences = self._build_tile_sequences()

        if limit_batches is not None and 0 < limit_batches <= 1.0:
            original_count = len(self.tile_sequences)
            limit_count = int(original_count * limit_batches)
            self.tile_sequences = self.tile_sequences[:limit_count]
            print(f"Limited {split} dataset: {original_count} -> {limit_count} sequences ({limit_batches:.1%})")

        if debug_monthly_split:
            print(f"DEBUG {split} split: {len(self.cities)} cities, year {debug_year}, "
                f"months {sorted(self.allowed_months)}, {len(self.tile_sequences)} tile sequences")
        else:
            print(f"{split} split: {len(self.cities)} cities, {len(self.years)} years "
                f"({min(self.years)}-{max(self.years)}), {len(self.tile_sequences)} tile sequences")
    
    def _get_cache_filename(self) -> str:
        """Generate a unique cache filename based on dataset parameters"""
        
        # Get the year sets (they're stored as self.years, but we need the original splits)
        if hasattr(self, 'train_years'):
            train_years = sorted(list(self.train_years)) if self.train_years else None
        else:
            train_years = None
            
        if hasattr(self, 'val_years'):
            val_years = sorted(list(self.val_years)) if self.val_years else None
        else:
            val_years = None
            
        if hasattr(self, 'test_years'):
            test_years = sorted(list(self.test_years)) if self.test_years else None
        else:
            test_years = None
        
        # Create a hash of the configuration - MUST match setup_data.py exactly
        config_str = f"{self.split}_{self.cluster}_{self.input_sequence_length}_{self.output_sequence_length}"
        config_str += f"_{train_years}_{val_years}_{test_years}"
        config_str += f"_{self.debug_monthly_split}_{self.debug_year}_{self.max_input_nodata_pct}"
        
        print(f"🔍 Cache config string: {config_str}")
        
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        cache_dir = self.dataset_root / "test_sequence_cache"
        cache_filename = str(cache_dir / f"test_sequences_{self.split}_{self.cluster}_{config_hash}.pkl")
        
        print(f"🔍 Looking for cache file: {cache_filename}")
        
        return cache_filename

    def _load_sequences_from_cache(self) -> Optional[List[Tuple[str, int, int, List[str], List[str]]]]:
        """Try to load sequences from cache"""
        cache_file = self._get_cache_filename()
        
        if not os.path.exists(cache_file):
            print(f"❌ Cache file not found: {os.path.basename(cache_file)}")
            print(f"   Full path: {cache_file}")
            print(f"💡 Run setup_data.py first to build cache:")
            print(f"   python setup_data.py --dataset_root {self.dataset_root} --cluster {self.cluster} --input_length {self.input_sequence_length} --output_length {self.output_sequence_length}")
            if not self.debug_monthly_split:
                print(f"   --train_years {' '.join(map(str, sorted(self.train_years)))} --val_years {' '.join(map(str, sorted(self.val_years)))} --test_years {' '.join(map(str, sorted(self.test_years)))}")
            else:
                print(f"   --debug --debug_year {self.debug_year}")
            print(f"   --max_nodata {self.max_input_nodata_pct}")
            
            # List what cache files DO exist
            cache_dir = os.path.dirname(cache_file)
            if os.path.exists(cache_dir):
                existing_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
                if existing_files:
                    print(f"   Existing cache files: {existing_files}")
            else:
                print(f"   Cache directory doesn't exist: {cache_dir}")
            
            return None
        
        try:
            print(f"📂 Loading cache: {os.path.basename(cache_file)}")
            with open(cache_file, 'rb') as f:
                sequences = pickle.load(f)
            
            print(f"✅ Loaded {len(sequences)} sequences from cache")
            return sequences
            
        except Exception as e:
            print(f"⚠️ Failed to load cache ({e})")
            print(f"💡 Try rebuilding cache with setup_data.py")
            return None
        
    
    def _normalize_scene(self, scene_data: np.ndarray) -> np.ndarray:
        """Normalize scene data to [0, 1] range using predefined ranges"""
        normalized_scene = scene_data.copy()
        
        for i, band_name in enumerate(self.band_names):
            band_data = scene_data[:, :, i]
            band_range = BAND_RANGES[band_name]
            
            # Normalize to [0, 1], keeping NODATA as 0
            valid_mask = band_data != 0
            normalized_band = np.zeros_like(band_data, dtype=np.float32)
            normalized_band[valid_mask] = (band_data[valid_mask] - band_range["min"]) / (band_range["max"] - band_range["min"])
            normalized_band = np.clip(normalized_band, 0, 1)
            
            normalized_scene[:, :, i] = normalized_band
        
        return normalized_scene
    
    def _generate_consecutive_sequences(self, city: str, tile_row: int, tile_col: int, 
                                    sorted_months: List[str], monthly_scenes: Dict[str, str]) -> List[Tuple]:
        """Generate sequences with strict consecutive month requirement (original logic for val/test)"""
        sequences = []
        
        for i in range(len(sorted_months) - self.total_sequence_length + 1):
            input_months = sorted_months[i:i + self.input_sequence_length]
            output_months = sorted_months[i + self.input_sequence_length:i + self.total_sequence_length]
            
            if self._are_consecutive_months(input_months + output_months):
                if self._verify_tile_sequence_exists(city, tile_row, tile_col, input_months + output_months):
                                        
                    # Check if any output scenes have NODATA in LST
                    has_nodata_output = False
                    for month in output_months:
                        if self._has_nodata_in_lst_tile(city, month, tile_row, tile_col, monthly_scenes):
                            has_nodata_output = True
                            break
                    
                    # Check if input sequence has excessive NODATA
                    has_excessive_input_nodata = False
                    if not has_nodata_output:
                        avg_nodata_pct = self._calculate_average_input_nodata(city, input_months, tile_row, tile_col, monthly_scenes)
                        if avg_nodata_pct > self.max_input_nodata_pct:
                            has_excessive_input_nodata = True
                    
                    # Only include sequence if passes all filters
                    if not has_nodata_output and not has_excessive_input_nodata:
                        sequences.append((city, tile_row, tile_col, input_months, output_months))
        
        return sequences

    def _generate_interpolated_sequences(self, city: str, tile_row: int, tile_col: int, 
                                    sorted_months: List[str], monthly_scenes: Dict[str, str]) -> List[Tuple]:
        """Generate sequences for training with interpolation for missing months"""
        sequences = []
        
        # Convert months to datetime for easier manipulation
        month_dates = []
        for month_str in sorted_months:
            year, month = map(int, month_str.split('-'))
            month_dates.append(datetime(year, month, 1))
        
        # Find all possible consecutive month spans that could work with interpolation
        min_date = month_dates[0]
        max_date = month_dates[-1]
        
        # Generate all possible starting points for sequences
        current_start = min_date
        while current_start <= max_date:
            # Calculate end date for this sequence
            sequence_end = current_start + relativedelta(months=self.total_sequence_length - 1)
            
            if sequence_end > max_date:
                break
            
            # Generate all months in this sequence
            sequence_months = []
            current_month = current_start
            for _ in range(self.total_sequence_length):
                month_str = f"{current_month.year}-{current_month.month:02d}"
                sequence_months.append(month_str)
                current_month += relativedelta(months=1)
            
            input_months = sequence_months[:self.input_sequence_length]
            output_months = sequence_months[self.input_sequence_length:]
            
            # Count how many of these months actually exist
            existing_input_months = [m for m in input_months if m in monthly_scenes]
            existing_output_months = [m for m in output_months if m in monthly_scenes]
            
            # Require at least 2 months in input sequence
            if len(existing_input_months) < 2:
                current_start += relativedelta(months=1)
                continue
            
            # Require at least 1 months in output sequence
            if len(existing_output_months) < 1:
                current_start += relativedelta(months=1)
                continue
            
            # Check tile existence for existing months only
            all_months_with_tiles = existing_input_months + existing_output_months
            if not self._verify_tile_sequence_exists(city, tile_row, tile_col, all_months_with_tiles):
                current_start += relativedelta(months=1)
                continue
            
            # Check NODATA in existing output months
            has_nodata_output = False
            for month in existing_output_months:
                if self._has_nodata_in_lst_tile(city, month, tile_row, tile_col, monthly_scenes):
                    has_nodata_output = True
                    break
            
            if has_nodata_output:
                current_start += relativedelta(months=1)
                continue
            
            # Check excessive NODATA in existing input months
            if existing_input_months:
                avg_nodata_pct = self._calculate_average_input_nodata(city, existing_input_months, tile_row, tile_col, monthly_scenes)
                if avg_nodata_pct > self.max_input_nodata_pct:
                    current_start += relativedelta(months=1)
                    continue
            
            # This sequence passes all checks - add it
            sequences.append((city, tile_row, tile_col, input_months, output_months))
            current_start += relativedelta(months=1)
        
        return sequences

    def _process_single_city(self, city: str) -> List[Tuple[str, int, int, List[str], List[str]]]:
        """Process a single city and return its sequences, with relaxed requirements for training split"""
        city_sequences = []
        
        available_tiles = self._get_available_tiles(city)
        monthly_scenes = self._get_monthly_scenes(city)
        
        min_required_scenes = 2
        
        if len(monthly_scenes) < min_required_scenes:
            return city_sequences
        
        sorted_months = sorted(monthly_scenes.keys())
        
        for (tile_row, tile_col) in available_tiles.keys():
            # Allow sequences with minimum 2 scenes and interpolate missing ones
            city_sequences.extend(self._generate_interpolated_sequences(
                city, tile_row, tile_col, sorted_months, monthly_scenes
            ))
        
        return city_sequences

    def _calculate_average_input_nodata(self, city: str, input_months: List[str], tile_row: int, tile_col: int, monthly_scenes: Dict[str, str]) -> float:
        """Calculate average NODATA percentage across all input sequence tiles (all bands)"""
        try:
            total_pixels = 0
            total_nodata_pixels = 0
            
            band_names = ['LST', 'red', 'green', 'blue', 'ndvi', 'ndwi', 'ndbi', 'albedo']  # Skip DEM as it's constant
            
            for month in input_months:
                scene_dir = Path(monthly_scenes[month])
                
                # Check each band for this timestep
                for band_name in band_names:
                    tile_path = scene_dir / f"{band_name}_row_{tile_row:03d}_col_{tile_col:03d}.tif"
                    
                    if not tile_path.exists():
                        # If file missing, count entire tile as NODATA
                        total_pixels += 128 * 128
                        total_nodata_pixels += 128 * 128
                        continue
                    
                    with rasterio.open(tile_path) as src:
                        band_data = src.read(1)
                        nodata_value = src.nodata if src.nodata is not None else 0
                        
                        # Count NODATA pixels (value 0 or actual nodata value)
                        nodata_mask = (band_data == 0) | (band_data == nodata_value)
                        
                        total_pixels += band_data.size
                        total_nodata_pixels += nodata_mask.sum()
            
            # Calculate average NODATA percentage across all input tiles
            if total_pixels == 0:
                return 1.0  # 100% NODATA if no valid data found
            
            avg_nodata_percentage = total_nodata_pixels / total_pixels
            return avg_nodata_percentage
            
        except Exception as e:
            print(f"Warning: Error calculating input NODATA for {city} {input_months} tile({tile_row},{tile_col}): {e}")
            return 1.0  # Treat errors as 100% NODATA to be safe

    def _has_nodata_in_lst_tile(self, city: str, month: str, tile_row: int, tile_col: int, monthly_scenes: Dict[str, str]) -> bool:
        """Check if LST tile contains any NODATA pixels (value 0)"""
        try:
            scene_dir = Path(monthly_scenes[month])
            lst_tile_path = scene_dir / f"LST_row_{tile_row:03d}_col_{tile_col:03d}.tif"
            
            if not lst_tile_path.exists():
                return True  # Missing file counts as having NODATA
            
            with rasterio.open(lst_tile_path) as src:
                lst_data = src.read(1)
                # Check for any NODATA pixels (value 0 or actual nodata value)
                nodata_value = src.nodata if src.nodata is not None else 0
                has_nodata = np.any((lst_data == 0) | (lst_data == nodata_value))
                return has_nodata
            
        except Exception as e:
            print(f"Warning: Error checking NODATA for {city} {month} tile({tile_row},{tile_col}): {e}")
            return True  # Treat errors as having NODATA to be safe
        
    def _setup_debug_monthly_splits(self):
        """Setup debug monthly splits within a single year"""
        train_months = [1, 2, 3, 4, 5, 6, 7, 8]
        val_months = [6, 7, 8, 9, 10]
        test_months = [8, 9, 10, 11, 12]
        
        if self.split == 'train':
            self.allowed_months = set(train_months)
        elif self.split == 'val':
            self.allowed_months = set(val_months)
        else:  # test
            self.allowed_months = set(test_months)
        
        self.years = {self.debug_year}
        
        self.train_months = set(train_months)
        self.val_months = set(val_months)
        self.test_months = set(test_months)
        
        print(f"Debug monthly splits for year {self.debug_year}:")
        print(f"  Train months: {sorted(train_months)} (Jan-Aug)")
        print(f"  Val months: {sorted(val_months)} (Jun-Oct)")
        print(f"  Test months: {sorted(test_months)} (Aug-Dec)")
        print(f"  Current split ({self.split}): {sorted(self.allowed_months)}")
    
    def _setup_year_based_splits(self, train_years, val_years, test_years):
        """Setup normal year-based splits"""
        if train_years is None or val_years is None or test_years is None:
            all_years = list(range(2013, 2026))  # 2013 to 2025 inclusive
            n_years = len(all_years)
            
            train_end = int(0.7 * n_years)  # 70% = ~9 years
            val_end = int(0.85 * n_years)   # 15% = ~2 years
            
            self.train_years = set(all_years[:train_end])           # 2013-2021 (9 years)
            self.val_years = set(all_years[train_end:val_end])      # 2022-2023 (2 years) 
            self.test_years = set(all_years[val_end:])              # 2024-2025 (2 years)
        else:
            self.train_years = set(train_years)
            self.val_years = set(val_years)
            self.test_years = set(test_years)
        
        if self.split == 'train':
            self.years = self.train_years
        elif self.split == 'val':
            self.years = self.val_years
        else:
            self.years = self.test_years
        
        self.allowed_months = None

    def get_excluded_cities(self):
        """Return list of cities to exclude from validation set"""
        return [
            "Athens-Clarke County unified government balance_GA",
            "Atlanta_GA",
            "Aurora_CO",
            "Birmingham_AL",
            "Cape Coral_FL",
            "Chesapeake_VA",
            "Columbia_SC",
            "Columbus_OH",
            "Denver_CO",
            "Durham_NC",
            "Hollywood_FL",
            "Jacksonville_FL",
            "Lehigh Acres_FL",
            "LouisvilleJefferson County metro government balance_KY",
            "Lubbock_TX",
            "Madison_WI",
            "Marana_AZ",
            "Miami_FL",
            "Mobile_AL",
            "Montgomery_AL",
            "New Orleans_LA",
            "North Port_FL",
            "Palm Springs_CA",
            "Peoria_AZ",
            "Phoenix_AZ",
            "Raleigh_NC",
            "Richmond_VA",
            "San Francisco_CA",
            "Savannah_GA",
            "Shreveport_LA",
            "Stockton_CA",
            "Tampa_FL",
            "Yuma_AZ"
        ]

    def _get_all_cities(self) -> List[str]:
        """Get all available cities from the tiled dataset, excluding validation-excluded cities for val split"""
        if self.cluster == "all":
            cities_dir = self.dataset_root / "Cities_Tiles"
        else:
            cities_dir = self.dataset_root / "Clustered" / self.cluster / "Cities_Tiles"
        
        all_cities = [d.name for d in cities_dir.iterdir() if d.is_dir()]
        
        # Exclude specific cities from validation set only
        if self.split == 'val':
            excluded_cities = set(self.get_excluded_cities())
            all_cities = [city for city in all_cities if city not in excluded_cities]
            print(f"Excluded {len(excluded_cities)} cities from validation set")
        
        return sorted(all_cities)
    
    def _get_available_tiles(self, city: str) -> Dict[Tuple[int, int], bool]:
        """Get all available tile positions for a city"""
        dem_dir = self.dataset_root / "DEM_2014_Tiles" / city
        available_tiles = {}
        
        if dem_dir.exists():
            for dem_file in dem_dir.glob("DEM_row_*_col_*.tif"):
                parts = dem_file.stem.split('_')
                if len(parts) >= 4:
                    try:
                        row = int(parts[2])
                        col = int(parts[4])
                        available_tiles[(row, col)] = True
                    except ValueError:
                        continue
        
        return available_tiles
    
    def _get_monthly_scenes(self, city: str) -> Dict[str, str]:
        """Get one scene per month for a city from tiled data, filtered by years and months"""
        if city in self._monthly_scenes_cache:
            return self._monthly_scenes_cache[city]
        monthly_scenes = {}
        scene_dirs = []
        for cluster in ["1", "2", "3", "4"]:
            city_dir = self.dataset_root / "Clustered" / cluster / "Cities_Tiles" / city # creates a Path
            if not city_dir.exists():
                return {}    
            for month_path in city_dir.iterdir(): #iterdir create path objects of the sub-contents of a folder.
                if month_path.is_dir():
                    scene_dirs.append(month_path)
        
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

        self._monthly_scenes_cache[city] = monthly_scenes
        return monthly_scenes
    
    def _validate_tiled_scene(self, scene_dir: Path) -> bool:
        """Check if scene has tile files"""
        tif_files = list(scene_dir.glob("*.tif"))
        return len(tif_files) > 0
    
    def _build_tile_sequences(self) -> List[Tuple[str, int, int, List[str], List[str]]]:
        """Build consecutive monthly sequences for each tile position, with caching support"""
        
        # Try to load from cache first
        cached_sequences = self._load_sequences_from_cache()
        if cached_sequences is not None:
            return cached_sequences
        
        print(f"\n🔄 Building tile sequences for {self.split} split (no cache found)...")
        print(f"💡 Tip: Run 'python setup_data.py' to pre-build caches for faster loading")
        
        if self.debug_monthly_split:
            month_stats = {month: 0 for month in self.allowed_months}
        else:
            year_stats = {year: 0 for year in self.years}
        
        # Check if we should use parallelism
        is_ddp = os.environ.get('LOCAL_RANK') is not None
        use_parallel = not is_ddp and len(self.cities) > 5
        
        if use_parallel:
            print(f"   Using {min(cpu_count(), len(self.cities))} cores for parallel processing...")
            # Process cities in parallel
            with Pool(processes=min(cpu_count(), len(self.cities))) as pool:
                city_results = list(tqdm(
                    pool.imap(self._process_single_city, self.cities),
                    total=len(self.cities),
                    desc=f"Processing cities ({self.split})",
                    unit="city"
                ))
        else:
            reason = "DDP detected" if is_ddp else f"only {len(self.cities)} cities"
            print(f"   Using sequential processing ({reason})...")
            city_results = []
            for city in tqdm(self.cities, desc=f"Processing cities ({self.split})", unit="city"):
                city_results.append(self._process_single_city(city))
        
        # Flatten results
        sequences = [seq for city_seqs in city_results for seq in city_seqs]
        
        # Calculate statistics (existing code)
        for city, tile_row, tile_col, input_months, output_months in sequences:
            if self.debug_monthly_split:
                first_month = int(input_months[0].split('-')[1])
                if first_month in month_stats:
                    month_stats[first_month] += 1
            else:
                first_year = int(input_months[0].split('-')[0])
                if first_year in year_stats:
                    year_stats[first_year] += 1
        
        # Print statistics (existing code)
        if self.debug_monthly_split:
            print(f"Sequences by month for {self.split} split (year {self.debug_year}):")
            for month in sorted(month_stats.keys()):
                month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1]
                print(f"  {month:02d} ({month_name}): {month_stats[month]} sequences")
        else:
            print(f"Sequences by year for {self.split} split:")
            for year in sorted(year_stats.keys()):
                print(f"  {year}: {year_stats[year]} sequences")
        
        return sequences
    
    def _verify_tile_sequence_exists(self, city: str, tile_row: int, tile_col: int, months: List[str]) -> bool:
        """Verify using cached scene validation"""
        cache_key = (city, tile_row, tile_col)
        
        if cache_key not in self._scene_validation_cache:
            monthly_scenes = self._get_monthly_scenes(city)
            valid_months = set()
            
            for month, scene_path in monthly_scenes.items():
                scene_dir = Path(scene_path)
                lst_tile = scene_dir / f"LST_row_{tile_row:03d}_col_{tile_col:03d}.tif"
                if lst_tile.exists():
                    valid_months.add(month)
            
            self._scene_validation_cache[cache_key] = valid_months
        
        valid_months = self._scene_validation_cache[cache_key]
        return all(month in valid_months for month in months)
    
    def _are_consecutive_months(self, months: List[str]) -> bool:
        """Check if months are consecutive"""
        dates = []
        for month_str in months:
            year, month = map(int, month_str.split('-'))
            dates.append((year, month))
        
        dates.sort()
        
        for i in range(1, len(dates)):
            prev_year, prev_month = dates[i-1]
            curr_year, curr_month = dates[i]
            
            if prev_month == 12:
                expected_year, expected_month = prev_year + 1, 1
            else:
                expected_year, expected_month = prev_year, prev_month + 1
            
            if (curr_year, curr_month) != (expected_year, expected_month):
                return False
        
        return True
    
    def _load_raster(self, file_path: str) -> np.ndarray:
        """Load raster file and return as numpy array"""
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    data[data == src.nodata] = 0
                return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return np.zeros((128, 128), dtype=np.float32)
    
    def _load_dem_tile(self, city: str, tile_row: int, tile_col: int) -> np.ndarray:
        """Load DEM tile for the city and position (with caching)"""
        cache_key = (city, tile_row, tile_col)
        
        # Check cache first
        if cache_key in self._dem_cache:
            return self._dem_cache[cache_key]
        
        dem_path = self.dataset_root / "DEM_2014_Tiles" / city / f"DEM_row_{tile_row:03d}_col_{tile_col:03d}.tif"
        dem_data = self._load_raster(str(dem_path))
        
        # Cache the result (DEM is static and reused frequently)
        self._dem_cache[cache_key] = dem_data
        return dem_data
    
    def _load_scene_tile(self, city: str, month: str, tile_row: int, tile_col: int) -> np.ndarray:
        """Load all bands for a scene tile"""
        monthly_scenes = self._get_monthly_scenes(city)
        scene_dir = Path(monthly_scenes[month])
        
        # Load DEM (constant for city/tile position)
        dem = self._load_dem_tile(city, tile_row, tile_col)
        
        # Load other bands
        bands = [dem]  # Start with DEM
        
        for band_name in self.band_names[1:]:  # Skip DEM since we loaded it separately
            tile_path = scene_dir / f"{band_name}_row_{tile_row:03d}_col_{tile_col:03d}.tif"
            band_data = self._load_raster(str(tile_path))
            bands.append(band_data)
        
        # Stack to (H, W, C)
        scene_data = np.stack(bands, axis=-1)
        return scene_data
    
    def __len__(self) -> int:
        return len(self.tile_sequences)
    

    def _interpolate_missing_scenes(self, months: List[str], existing_data: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """
        Optimized implementation with accurate extrapolation and fast integer operations
        """
        if len(existing_data) == len(months):
            return [existing_data[m] for m in months]
        
        # Fast integer conversion - no datetime objects
        def month_to_int(month_str: str) -> int:
            year, month = map(int, month_str.split('-'))
            return year * 12 + month  # Convert to months since year 0
        
        # Vectorized conversions
        target_nums = np.array([month_to_int(m) for m in months])
        existing_months = sorted(existing_data.keys())
        existing_nums = np.array([month_to_int(m) for m in existing_months])
        
        # Stack existing data once
        existing_stack = np.stack([existing_data[m] for m in existing_months])
        n_existing, height, width, channels = existing_stack.shape
        
        # Preallocate result
        result_stack = np.zeros((len(months), height, width, channels), dtype=np.float32)
        
        # Vectorized approach: find indices for all targets at once
        # Use searchsorted for fast binary search
        insert_indices = np.searchsorted(existing_nums, target_nums)
        
        for i, (target_num, insert_idx) in enumerate(zip(target_nums, insert_indices)):
            if insert_idx < len(existing_nums) and existing_nums[insert_idx] == target_num:
                # Exact match
                result_stack[i] = existing_stack[insert_idx]
            elif insert_idx == 0:
                # Before all existing data - extrapolate using first two points
                if len(existing_nums) >= 2:
                    x1, x2 = existing_nums[0], existing_nums[1]
                    y1, y2 = existing_stack[0], existing_stack[1]
                    weight = (target_num - x1) / (x2 - x1)
                    result_stack[i] = y1 + weight * (y2 - y1)
                else:
                    result_stack[i] = existing_stack[0]
            elif insert_idx == len(existing_nums):
                # After all existing data - extrapolate using last two points
                if len(existing_nums) >= 2:
                    x1, x2 = existing_nums[-2], existing_nums[-1]
                    y1, y2 = existing_stack[-2], existing_stack[-1]
                    weight = (target_num - x2) / (x2 - x1)
                    result_stack[i] = y2 + weight * (y2 - y1)
                else:
                    result_stack[i] = existing_stack[-1]
            else:
                # Interpolate between surrounding points
                before_idx = insert_idx - 1
                after_idx = insert_idx
                
                before_num = existing_nums[before_idx]
                after_num = existing_nums[after_idx]
                
                # Linear interpolation weight
                weight = (target_num - before_num) / (after_num - before_num)
                result_stack[i] = (1 - weight) * existing_stack[before_idx] + weight * existing_stack[after_idx]
        
        return [result_stack[i] for i in range(len(months))]

    def _load_sequence_with_interpolation(self, city: str, tile_row: int, tile_col: int, 
                                    months: List[str], monthly_scenes: Dict[str, str], 
                                    lst_only: bool = False) -> List[np.ndarray]:
        """Load a sequence of scenes with interpolation for missing months"""
        scenes = []
        monthly_scenes_for_city = monthly_scenes
        
        # First, load all existing scenes
        existing_data = {}
        for month in months:
            if month in monthly_scenes_for_city:
                try:
                    scene = self._load_scene_tile(city, month, tile_row, tile_col)
                    if scene is not None:
                        normalized_scene = self._normalize_scene(scene)
                        if lst_only:
                            normalized_scene = normalized_scene[:, :, 1:2]  # LST band only
                        existing_data[month] = normalized_scene
                except Exception as e:
                    # Skip months that fail to load (file corruption, etc.)
                    print(f"Warning: Failed to load {city} {month} tile({tile_row},{tile_col}): {e}")
                    continue
        
        # If we have all the data, return it directly
        if len(existing_data) == len(months):
            return [existing_data[month] for month in months]
        
        # If training split and missing some months, interpolate
        if len(existing_data) >= 2: 
            return self._interpolate_missing_scenes(months, existing_data)
        
        # This should never happen due to filtering - raise error to debug
        raise ValueError(
            f"Insufficient data for sequence interpolation: "
            f"city={city}, tile=({tile_row},{tile_col}), "
            f"months={months}, existing_months={list(existing_data.keys())}, "
            f"split={self.split}, lst_only={lst_only}. "
            f"Expected at least 2 existing months but found {len(existing_data)}."
        )

    def _apply_spatial_augmentation(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply spatial augmentations (rotation and flips) with 50% probability"""        
        if random.random() < 0.5:  # 50% chance to apply augmentation
            # Random rotation (0, 90, 180, 270 degrees)
            k = random.randint(0, 3)
            if k > 0:
                input_tensor = torch.rot90(input_tensor, k, dims=[-3, -2])  # Rotate H,W dimensions
                target_tensor = torch.rot90(target_tensor, k, dims=[-3, -2])
            
            # Random horizontal flip (50% chance)
            if random.random() < 0.5:
                input_tensor = torch.flip(input_tensor, dims=[-2])  # Flip width
                target_tensor = torch.flip(target_tensor, dims=[-2])
            
            # Random vertical flip (50% chance)
            if random.random() < 0.5:
                input_tensor = torch.flip(input_tensor, dims=[-3])  # Flip height
                target_tensor = torch.flip(target_tensor, dims=[-3])
    
        return input_tensor, target_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:       
        # Fallback to original disk loading
        city, tile_row, tile_col, input_months, output_months = self.tile_sequences[idx]
        monthly_scenes = self._get_monthly_scenes(city)

        # Load input sequence
        input_scenes = self._load_sequence_with_interpolation(
            city, tile_row, tile_col, input_months, monthly_scenes
        )
        
        #Create a load sequence with no interpolation if requiring more than 1 sequence
        output_scenes = self._load_sequence_with_interpolation(
            city, tile_row, tile_col, output_months, monthly_scenes, lst_only=True
        )
        
        # Convert to tensors
        input_tensor = torch.from_numpy(np.stack(input_scenes, axis=0))
        target_tensor = torch.from_numpy(np.stack(output_scenes, axis=0))
        
        # Apply augmentation only for training split
        if self.split == 'train':
            input_tensor, target_tensor = self._apply_spatial_augmentation(input_tensor, target_tensor)
        
        return input_tensor, target_tensor

# Updated LandsatDataModule class
class LandsatDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        input_sequence_length: int = 3,
        output_sequence_length: int = 3,  
        train_years: Optional[List[int]] = None,
        val_years: Optional[List[int]] = None,
        test_years: Optional[List[int]] = None,
        debug_monthly_split: bool = False,
        debug_year: int = 2014,
        max_input_nodata_pct: float = 0.60,
        cluster: str = "all",
        limit_train_batches: Optional[float] = None,  
        limit_val_batches: Optional[float] = None     
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        if input_sequence_length < 3:
            print("Warning: input_sequence_length should be at least 3, setting input sequence length to 3")
            input_sequence_length = 3
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years
        self.debug_monthly_split = debug_monthly_split
        self.debug_year = debug_year
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.max_input_nodata_pct = max_input_nodata_pct
        self.cluster = cluster

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage"""
        if self.limit_train_batches == 1:
            self.limit_train_batches = 1.0
        if self.limit_val_batches == 1:
            self.limit_val_batches = 1.0
        
        print(f"🔧 Setting up datasets for stage: {stage}")
        
        # Setup train and val datasets for training
        if stage == "fit" or stage is None:
            print("📚 Loading training dataset...")
            self.train_dataset = LandsatSequenceDataset(
                self.dataset_root,
                cluster=self.cluster,
                input_sequence_length=self.input_sequence_length,
                output_sequence_length=self.output_sequence_length,
                split='train',
                train_years=self.train_years,
                val_years=self.val_years,
                test_years=self.test_years,
                debug_monthly_split=self.debug_monthly_split,
                debug_year=self.debug_year,
                limit_batches=self.limit_train_batches,
                max_input_nodata_pct=self.max_input_nodata_pct
            )
            
            print("📊 Loading validation dataset...")
            self.val_dataset = LandsatSequenceDataset(
                self.dataset_root,
                cluster=self.cluster,
                input_sequence_length=self.input_sequence_length,
                output_sequence_length=self.output_sequence_length,
                split='val',
                train_years=self.train_years,
                val_years=self.val_years,
                test_years=self.test_years,
                debug_monthly_split=self.debug_monthly_split,
                debug_year=self.debug_year,
                limit_batches=self.limit_val_batches,
                max_input_nodata_pct=self.max_input_nodata_pct
            )
        
        # Setup test dataset
        if stage == "test" or stage is None:
            print("🧪 Loading test dataset...")
            self.test_dataset = LandsatSequenceDataset(
                self.dataset_root,
                cluster=self.cluster,
                input_sequence_length=self.input_sequence_length,
                output_sequence_length=self.output_sequence_length,
                split='test',
                train_years=self.train_years,
                val_years=self.val_years,
                test_years=self.test_years,
                debug_monthly_split=self.debug_monthly_split,
                debug_year=self.debug_year,
                max_input_nodata_pct=self.max_input_nodata_pct,             
                limit_batches=getattr(self, 'limit_test_batches', None)
            )
        
        print("✅ Dataset setup complete!")
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=8,
            multiprocessing_context='spawn'
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )