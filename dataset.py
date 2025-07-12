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

def load_interpolated_scenes(interpolated_file: str = "interpolated.txt") -> Set[str]:
    """
    Load list of interpolated scenes that should not be used as ground truth
    
    Args:
        interpolated_file: Path to file containing interpolated scene identifiers
        
    Returns:
        Set of scene identifiers in format "City_State/YYYY-MM-DDTHH:MM:SSZ"
    """
    interpolated_scenes = set()
    
    if os.path.exists(interpolated_file):
        with open(interpolated_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    interpolated_scenes.add(line)
        
        print(f"Loaded {len(interpolated_scenes)} interpolated scenes to exclude from ground truth")
        
        # Print some examples for verification
        if interpolated_scenes:
            print("Examples of interpolated scenes:")
            for i, scene in enumerate(sorted(list(interpolated_scenes))[:5]):
                print(f"  {scene}")
            if len(interpolated_scenes) > 5:
                print(f"  ... and {len(interpolated_scenes) - 5} more")
    else:
        print(f"Warning: Interpolated scenes file '{interpolated_file}' not found. All scenes will be used as ground truth.")
    
    return interpolated_scenes

class LandsatSequenceDataset(Dataset):
    def __init__(
        self, 
        dataset_root: str,
        input_sequence_length: int = 3,
        output_sequence_length: int = 3,
        split: str = 'train',
        train_years: Optional[List[int]] = None,
        val_years: Optional[List[int]] = None,
        test_years: Optional[List[int]] = None,
        debug_monthly_split: bool = False,
        debug_year: int = 2014,
        load_to_ram: bool = False,        
        interpolated_scenes_file: str = "interpolated.txt",
        limit_batches: Optional[float] = None,
        max_input_nodata_pct: float = 0.60,
        chunk_size: int = 100  # Increased default chunk size
    ):
        """
        Dataset for Landsat sequence prediction using tiled data with year-based splits
        """
        self.dataset_root = Path(dataset_root)
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.total_sequence_length = input_sequence_length + output_sequence_length
        self.split = split
        self.debug_monthly_split = debug_monthly_split
        self.debug_year = debug_year
        self._scene_validation_cache = {}
        self.cached_data = None
        self.cached_tiles = {}  # Cache for individual tiles
        self.max_input_nodata_pct = max_input_nodata_pct
        
        # Load interpolated scenes to exclude from ground truth
        self.interpolated_scenes = load_interpolated_scenes(interpolated_scenes_file)
        
        if debug_monthly_split:
            self._setup_debug_monthly_splits()
        else:
            self._setup_year_based_splits(train_years, val_years, test_years)
        
        self.cities = self._get_all_cities()
        self.band_names = ['DEM', 'LST', 'red', 'green', 'blue', 'ndvi', 'ndwi', 'ndbi', 'albedo']
        
        # Build tile sequences with interpolated scene filtering
        self.tile_sequences = self._build_tile_sequences()

        if limit_batches is not None and 0 < limit_batches <= 1.0:
            original_count = len(self.tile_sequences)
            limit_count = int(original_count * limit_batches)
            self.tile_sequences = self.tile_sequences[:limit_count]
            print(f"Limited {split} dataset: {original_count} -> {limit_count} sequences ({limit_batches:.1%})")
    
        
        # Print filtering statistics
        self._print_filtering_stats()    
                    
        if debug_monthly_split:
            print(f"DEBUG {split} split: {len(self.cities)} cities, year {debug_year}, "
                  f"months {sorted(self.allowed_months)}, {len(self.tile_sequences)} tile sequences")
        else:
            print(f"{split} split: {len(self.cities)} cities, {len(self.years)} years "
                  f"({min(self.years)}-{max(self.years)}), {len(self.tile_sequences)} tile sequences")
            
    @staticmethod
    def _load_sequence_for_cache(sequence_info, dataset_root, band_names):
        """Static method for parallel loading of a single sequence"""
        city, tile_row, tile_col, input_months, output_months = sequence_info
        dataset_root = Path(dataset_root)
        
        try:
            # Load input sequence
            input_scenes = []
            for month in input_months:
                scene = LandsatSequenceDataset._load_scene_tile_static(
                    dataset_root, city, month, tile_row, tile_col, band_names
                )
                if scene is None:
                    return None
                normalized_scene = LandsatSequenceDataset._normalize_scene_static(scene, band_names)
                input_scenes.append(normalized_scene)
            
            # Load output sequence
            output_scenes = []
            for month in output_months:
                scene = LandsatSequenceDataset._load_scene_tile_static(
                    dataset_root, city, month, tile_row, tile_col, band_names
                )
                if scene is None:
                    return None
                normalized_scene = LandsatSequenceDataset._normalize_scene_static(scene, band_names)
                lst_only = normalized_scene[:, :, 1:2]  # LST band only
                output_scenes.append(lst_only)
            
            return input_scenes, output_scenes
            
        except Exception as e:
            print(f"Error loading sequence {city} {tile_row},{tile_col}: {e}")
            return None  
              
    @staticmethod
    def _load_scene_tile_static(dataset_root, city, month, tile_row, tile_col, band_names):
        """Static version of _load_scene_tile for multiprocessing"""
        try:
            # Get scene path
            city_dir = dataset_root / "Cities_Tiles" / city
            monthly_scenes = {}
            
            for scene_dir in city_dir.iterdir():
                if scene_dir.is_dir():
                    date_obj = datetime.fromisoformat(scene_dir.name.replace('Z', '+00:00'))
                    month_key = f"{date_obj.year}-{date_obj.month:02d}"
                    if month_key == month:
                        monthly_scenes[month] = str(scene_dir)
                        break
            
            if month not in monthly_scenes:
                return None
                
            scene_dir = Path(monthly_scenes[month])
            
            # Load DEM
            dem_path = dataset_root / "DEM_2014_Tiles" / city / f"DEM_row_{tile_row:03d}_col_{tile_col:03d}.tif"
            with rasterio.open(dem_path) as src:
                dem = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    dem[dem == src.nodata] = 0
            
            # Load other bands
            bands = [dem]
            for band_name in band_names[1:]:  # Skip DEM
                tile_path = scene_dir / f"{band_name}_row_{tile_row:03d}_col_{tile_col:03d}.tif"
                with rasterio.open(tile_path) as src:
                    band_data = src.read(1).astype(np.float32)
                    if src.nodata is not None:
                        band_data[band_data == src.nodata] = 0
                    bands.append(band_data)
            
            return np.stack(bands, axis=-1)
            
        except Exception:
            return None

    @staticmethod
    def _normalize_scene_static(scene_data, band_names):
        """Static version of _normalize_scene for multiprocessing"""
        normalized_scene = scene_data.copy()
        
        for i, band_name in enumerate(band_names):
            band_data = scene_data[:, :, i]
            band_range = BAND_RANGES[band_name]
            
            valid_mask = band_data != 0
            normalized_band = np.zeros_like(band_data, dtype=np.float32)
            normalized_band[valid_mask] = (band_data[valid_mask] - band_range["min"]) / (band_range["max"] - band_range["min"])
            normalized_band = np.clip(normalized_band, 0, 1)
            
            normalized_scene[:, :, i] = normalized_band
        
        return normalized_scene

    def _print_filtering_stats(self):
        """Print statistics about interpolated scene filtering"""
        if not self.interpolated_scenes:
            return
            
        total_sequences_checked = 0
        sequences_with_interpolated_output = 0
        sequences_kept = len(self.tile_sequences)
        
        # We can't easily count filtered sequences without rebuilding, but we can show what was loaded
        print(f"\n=== INTERPOLATED SCENE FILTERING STATS ===")
        print(f"Interpolated scenes loaded: {len(self.interpolated_scenes)}")
        print(f"Valid sequences after filtering: {sequences_kept}")
        
        # Check if any interpolated scenes affect current split years
        affected_years = set()
        for scene_id in self.interpolated_scenes:
            try:
                # Extract year from scene_id format: "City_State/YYYY-MM-DDTHH:MM:SSZ"
                date_part = scene_id.split('/')[1]
                year = int(date_part.split('-')[0])
                if year in self.years:
                    affected_years.add(year)
            except:
                continue
        
        if affected_years:
            print(f"Interpolated scenes affect years in this split: {sorted(affected_years)}")
        else:
            print(f"No interpolated scenes affect years in this split: {sorted(self.years)}")
        print(f"{'='*50}")
    
    def _is_scene_interpolated(self, city: str, scene_date: str) -> bool:
        """
        Check if a scene is in the interpolated scenes list
        
        Args:
            city: City name (e.g., "Austin_TX")
            scene_date: Scene date string (e.g., "2013-12-15T12:00:00Z")
            
        Returns:
            True if scene is interpolated and should not be used as ground truth
        """
        # Create scene identifier in the format from interpolated.txt
        scene_id = f"{city}/{scene_date}"
        return scene_id in self.interpolated_scenes
    
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
    
    def _process_single_city(self, city: str) -> List[Tuple[str, int, int, List[str], List[str]]]:
        """Process a single city and return its sequences, filtering out those with interpolated ground truth, NODATA in outputs, or >60% average NODATA in inputs"""
        city_sequences = []
        
        available_tiles = self._get_available_tiles(city)
        monthly_scenes = self._get_monthly_scenes(city)
        
        if len(monthly_scenes) < self.total_sequence_length:
            return city_sequences
        
        sorted_months = sorted(monthly_scenes.keys())
        
        for (tile_row, tile_col) in available_tiles.keys():
            for i in range(len(sorted_months) - self.total_sequence_length + 1):
                input_months = sorted_months[i:i + self.input_sequence_length]
                output_months = sorted_months[i + self.input_sequence_length:i + self.total_sequence_length]
                
                if self._are_consecutive_months(input_months + output_months):
                    if self._verify_tile_sequence_exists(city, tile_row, tile_col, input_months + output_months):
                        
                        # Check if any output scenes are interpolated
                        has_interpolated_output = False
                        for month in output_months:
                            scene_path = monthly_scenes[month]
                            scene_dir = Path(scene_path)
                            scene_date = scene_dir.name  # e.g., "2013-12-15T12:00:00Z"
                            
                            if self._is_scene_interpolated(city, scene_date):
                                has_interpolated_output = True
                                break
                        
                        # Check if any output scenes have NODATA in LST
                        has_nodata_output = False
                        if not has_interpolated_output:  # Only check if not already filtered out
                            for month in output_months:
                                if self._has_nodata_in_lst_tile(city, month, tile_row, tile_col, monthly_scenes):
                                    has_nodata_output = True
                                    break
                        
                        # NEW: Check if input sequence has >60% average NODATA across all bands
                        has_excessive_input_nodata = False
                        if not has_interpolated_output and not has_nodata_output:
                            avg_nodata_pct = self._calculate_average_input_nodata(city, input_months, tile_row, tile_col, monthly_scenes)
                            if avg_nodata_pct > self.max_input_nodata_pct:  # Use configurable threshold
                                has_excessive_input_nodata = True
                        
                        # Only include sequence if passes all filters
                        if not has_interpolated_output and not has_nodata_output and not has_excessive_input_nodata:
                            city_sequences.append((city, tile_row, tile_col, input_months, output_months))
        
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

    def _get_all_cities(self) -> List[str]:
        """Get all available cities from the tiled dataset"""
        cities_dir = self.dataset_root / "Cities_Tiles"
        cities = [d.name for d in cities_dir.iterdir() if d.is_dir()]
        return sorted(cities)
    
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
        city_dir = self.dataset_root / "Cities_Tiles" / city
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
        
        return monthly_scenes
    
    def _validate_tiled_scene(self, scene_dir: Path) -> bool:
        """Check if scene has tile files"""
        tif_files = list(scene_dir.glob("*.tif"))
        return len(tif_files) > 0
    
    def _build_tile_sequences(self) -> List[Tuple[str, int, int, List[str], List[str]]]:
        """Build consecutive monthly sequences for each tile position, filtering interpolated ground truth"""
        if self.debug_monthly_split:
            month_stats = {month: 0 for month in self.allowed_months}
        else:
            year_stats = {year: 0 for year in self.years}
        
        print(f"\n🔄 Building tile sequences for {self.split} split using {min(cpu_count(), len(self.cities))} cores...")
        print(f"   Excluding {len(self.interpolated_scenes)} interpolated scenes from ground truth...")
        
        # Process cities in parallel
        with Pool(processes=min(cpu_count(), len(self.cities))) as pool:
            city_results = list(tqdm(
                pool.imap(self._process_single_city, self.cities),
                total=len(self.cities),
                desc=f"Processing cities ({self.split})",
                unit="city"
            ))
        
        # Flatten results
        sequences = [seq for city_seqs in city_results for seq in city_seqs]
        
        # Calculate statistics
        for city, tile_row, tile_col, input_months, output_months in sequences:
            if self.debug_monthly_split:
                first_month = int(input_months[0].split('-')[1])
                if first_month in month_stats:
                    month_stats[first_month] += 1
            else:
                first_year = int(input_months[0].split('-')[0])
                if first_year in year_stats:
                    year_stats[first_year] += 1
        
        # Print statistics
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
        """Load DEM tile for the city and position"""
        dem_path = self.dataset_root / "DEM_2014_Tiles" / city / f"DEM_row_{tile_row:03d}_col_{tile_col:03d}.tif"
        return self._load_raster(str(dem_path))
    
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
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns cached data if available, otherwise loads from disk"""
        if self.cached_data is not None:
            return self.cached_data[idx]
        
        # Fallback to original disk loading
        city, tile_row, tile_col, input_months, output_months = self.tile_sequences[idx]
        
        # Load input sequence
        input_scenes = []
        for month in input_months:
            scene = self._load_scene_tile(city, month, tile_row, tile_col)
            normalized_scene = self._normalize_scene(scene)
            input_scenes.append(normalized_scene)
        
        # Load output sequence
        output_scenes = []
        for month in output_months:
            scene = self._load_scene_tile(city, month, tile_row, tile_col)
            normalized_scene = self._normalize_scene(scene)
            lst_only = normalized_scene[:, :, 1:2]
            output_scenes.append(lst_only)
        
        # Convert to tensors
        input_tensor = torch.from_numpy(np.stack(input_scenes, axis=0))
        target_tensor = torch.from_numpy(np.stack(output_scenes, axis=0))
        
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
        interpolated_scenes_file: str = "interpolated.txt",
        max_input_nodata_pct: float = 0.60,
        limit_train_batches: Optional[float] = None,  # NEW PARAMETER
        limit_val_batches: Optional[float] = None     # NEW PARAMETER
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years
        self.debug_monthly_split = debug_monthly_split
        self.debug_year = debug_year
        self.interpolated_scenes_file = interpolated_scenes_file
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.max_input_nodata_pct = max_input_nodata_pct
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage with interpolated scene filtering"""
        
        # Setup train and val datasets for training
        if stage == "fit" or stage is None:
            self.train_dataset = LandsatSequenceDataset(
                self.dataset_root,
                input_sequence_length=self.input_sequence_length,
                output_sequence_length=self.output_sequence_length,
                split='train',
                train_years=self.train_years,
                val_years=self.val_years,
                test_years=self.test_years,
                debug_monthly_split=self.debug_monthly_split,
                debug_year=self.debug_year,
                interpolated_scenes_file=self.interpolated_scenes_file,                
                limit_batches=self.limit_train_batches,
                max_input_nodata_pct=self.max_input_nodata_pct
            )
            
            self.val_dataset = LandsatSequenceDataset(
                self.dataset_root,
                input_sequence_length=self.input_sequence_length,
                output_sequence_length=self.output_sequence_length,
                split='val',
                train_years=self.train_years,
                val_years=self.val_years,
                test_years=self.test_years,
                debug_monthly_split=self.debug_monthly_split,
                debug_year=self.debug_year,
                interpolated_scenes_file=self.interpolated_scenes_file,                
                limit_batches=self.limit_val_batches,
                max_input_nodata_pct=self.max_input_nodata_pct
            )
        
        # Setup test dataset
        if stage == "test" or stage is None:
            self.test_dataset = LandsatSequenceDataset(
                self.dataset_root,
                input_sequence_length=self.input_sequence_length,
                output_sequence_length=self.output_sequence_length,
                split='test',
                train_years=self.train_years,
                val_years=self.val_years,
                test_years=self.test_years,
                debug_monthly_split=self.debug_monthly_split,
                debug_year=self.debug_year,
                interpolated_scenes_file=self.interpolated_scenes_file,   
                max_input_nodata_pct=self.max_input_nodata_pct,             
                limit_batches=getattr(self, 'limit_test_batches', None)  # Use test-specific limit if available
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
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


# Usage example with interpolated scene filtering
def test_interpolated_filtering():
    """Test the interpolated scene filtering functionality"""
    
    print("=== TESTING INTERPOLATED SCENE FILTERING ===\n")
    
    # Create data module with interpolated scene filtering
    data_module = LandsatDataModule(
        dataset_root="./Data/ML",
        batch_size=2,
        input_sequence_length=3,
        output_sequence_length=1,
        debug_monthly_split=True,
        debug_year=2014,
        interpolated_scenes_file="./Data/ML/interpolated.txt"
    )
    
    # Setup and analyze the results
    data_module.setup("fit")
    
    print(f"Train sequences: {len(data_module.train_dataset.tile_sequences)}")
    print(f"Val sequences: {len(data_module.val_dataset.tile_sequences)}")
    
    # Check specific sequences for interpolated scenes
    if len(data_module.train_dataset.tile_sequences) > 0:
        sample_seq = data_module.train_dataset.tile_sequences[0]
        city, tile_row, tile_col, input_months, output_months = sample_seq
        
        print(f"\nSample sequence: {city} tile({tile_row},{tile_col})")
        print(f"  Input months: {input_months}")
        print(f"  Output months: {output_months}")
        
        # Verify no output months are interpolated
        interpolated_scenes = data_module.train_dataset.interpolated_scenes
        monthly_scenes = data_module.train_dataset._get_monthly_scenes(city)
        
        for month in output_months:
            if month in monthly_scenes:
                scene_path = monthly_scenes[month]
                scene_dir = Path(scene_path)
                scene_date = scene_dir.name
                scene_id = f"{city}/{scene_date}"
                
                is_interpolated = scene_id in interpolated_scenes
                print(f"    {month} ({scene_date}): {'INTERPOLATED' if is_interpolated else 'ORIGINAL'}")
    
    return data_module