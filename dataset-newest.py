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
from rasterio.windows import Window

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
    """Load list of interpolated scenes that should not be used as ground truth"""
    interpolated_scenes = set()
    
    if os.path.exists(interpolated_file):
        with open(interpolated_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    interpolated_scenes.add(line)
        print(f"Loaded {len(interpolated_scenes)} interpolated scenes to exclude from ground truth")
    else:
        print(f"Warning: Interpolated scenes file '{interpolated_file}' not found.")
    
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
        interpolated_scenes_file: str = "interpolated.txt",
        tile_size: int = 128,
        tile_overlap: float = 0.0,
        nodata_fill_value: float = 0.0,
        precision: str = "32"
    ):
        """
        Fast dataset using Repo 1's efficient tiling approach with Repo 2's sequence modeling
        """
        self.dataset_root = Path(dataset_root)
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.total_sequence_length = input_sequence_length + output_sequence_length
        self.split = split
        self.debug_monthly_split = debug_monthly_split
        self.debug_year = debug_year
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.nodata_fill_value = nodata_fill_value
        self.precision = precision
        self.cached_data = None
        
        # Load interpolated scenes to exclude from ground truth
        self.interpolated_scenes = load_interpolated_scenes(interpolated_scenes_file)
        
        if debug_monthly_split:
            self._setup_debug_monthly_splits()
        else:
            self._setup_year_based_splits(train_years, val_years, test_years)
        
        self.cities = self._get_all_cities()
        self.band_names = ['DEM', 'LST', 'red', 'green', 'blue', 'ndvi', 'ndwi', 'ndbi', 'albedo']
        
        # Build tile sequences using Repo 1's efficient approach
        self.tile_sequences = self._build_efficient_tile_sequences()
        
        # Print filtering statistics
        self._print_filtering_stats()
        
        if debug_monthly_split:
            print(f"DEBUG {split} split: {len(self.cities)} cities, year {debug_year}, "
                  f"months {sorted(self.allowed_months)}, {len(self.tile_sequences)} tile sequences")
        else:
            print(f"{split} split: {len(self.cities)} cities, {len(self.years)} years "
                  f"({min(self.years)}-{max(self.years)}), {len(self.tile_sequences)} tile sequences")

    def _get_tiles(self, img_size: Tuple[int, int], tile_size: int, 
                  tile_overlap: float) -> List[Tuple[int, int, int, int]]:
        """Generate tile coordinates using Repo 1's efficient approach"""
        tiles = []
        img_w, img_h = img_size
        
        stride_w = int((1 - tile_overlap) * tile_size)
        stride_h = int((1 - tile_overlap) * tile_size)
        
        for y in range(0, img_h - tile_size + 1, stride_h):
            for x in range(0, img_w - tile_size + 1, stride_w):
                x2 = x + tile_size
                y2 = y + tile_size
                
                if x2 <= img_w and y2 <= img_h:
                    tiles.append((x, y, x2, y2))
        
        return tiles

    def _get_scene_dimensions(self, city: str, scene_path: str) -> Tuple[int, int]:
        """Get scene dimensions for tile calculation"""
        # Try to get dimensions from any available band file
        scene_dir = Path(scene_path)
        for band_file in scene_dir.glob("*.tif"):
            try:
                with rasterio.open(band_file) as src:
                    return src.width, src.height
            except:
                continue
        return 128, 128  # fallback

    @staticmethod
    def _process_city_sequences_static(
        city: str,
        dataset_root: str,
        total_sequence_length: int,
        input_sequence_length: int,
        tile_size: int,
        tile_overlap: float,
        years: set,
        allowed_months: Optional[set],
        debug_monthly_split: bool,
        interpolated_scenes: set
    ) -> List[Tuple[str, int, int, int, int, List[str], List[str]]]:
        """Static method to process a single city's sequences"""
        try:
            dataset_root = Path(dataset_root)
            
            # Get monthly scenes for this city
            monthly_scenes = LandsatSequenceDataset._get_monthly_scenes_static(
                city, dataset_root, years, allowed_months, debug_monthly_split
            )
            
            if len(monthly_scenes) < total_sequence_length:
                return []
            
            sorted_months = sorted(monthly_scenes.keys())
            sequences = []
            
            # Get scene dimensions for tiling (use first available scene)
            if sorted_months:
                first_scene_path = monthly_scenes[sorted_months[0]]
                scene_width, scene_height = LandsatSequenceDataset._get_scene_dimensions_static(
                    city, first_scene_path
                )
                
                # Generate tiles
                tile_coords = LandsatSequenceDataset._get_tiles_static(
                    (scene_width, scene_height), tile_size, tile_overlap
                )
                
                # Build sequences for each tile
                for tile_idx, (x_min, y_min, x_max, y_max) in enumerate(tile_coords):
                    for i in range(len(sorted_months) - total_sequence_length + 1):
                        input_months = sorted_months[i:i + input_sequence_length]
                        output_months = sorted_months[i + input_sequence_length:i + total_sequence_length]
                        
                        if LandsatSequenceDataset._are_consecutive_months_static(input_months + output_months):
                            # Check if any output scenes are interpolated
                            has_interpolated_output = False
                            for month in output_months:
                                scene_path = monthly_scenes[month]
                                scene_dir = Path(scene_path)
                                scene_date = scene_dir.name
                                
                                if LandsatSequenceDataset._is_scene_interpolated_static(
                                    city, scene_date, interpolated_scenes
                                ):
                                    has_interpolated_output = True
                                    break
                            
                            # Only include sequence if NO output scenes are interpolated
                            if not has_interpolated_output:
                                # Verify all required files exist for this tile
                                if LandsatSequenceDataset._verify_tile_files_exist_static(
                                    city, x_min, y_min, x_max, y_max, input_months + output_months,
                                    monthly_scenes, dataset_root
                                ):
                                    sequences.append((
                                        city, x_min, y_min, x_max, y_max, input_months, output_months
                                    ))
            
            return sequences
        
        except Exception as e:
            print(f"Error processing city {city}: {e}")
            return []
        

    @staticmethod
    def _get_monthly_scenes_static(
        city: str, 
        dataset_root: Path, 
        years: set, 
        allowed_months: Optional[set], 
        debug_monthly_split: bool
    ) -> Dict[str, str]:
        """Static method to get monthly scenes for a city"""
        city_dir = dataset_root / "Dataset" / "Cities_Processed" / city
        if not city_dir.exists():
            return {}
        
        monthly_scenes = {}
        scene_dirs = [d for d in city_dir.iterdir() if d.is_dir()]
        
        for scene_dir in scene_dirs:
            try:
                date_str = scene_dir.name
                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                
                if date_obj.year not in years:
                    continue
                
                if debug_monthly_split and allowed_months is not None:
                    if date_obj.month not in allowed_months:
                        continue
                
                month_key = f"{date_obj.year}-{date_obj.month:02d}"
                
                if month_key not in monthly_scenes:
                    monthly_scenes[month_key] = str(scene_dir)
                        
            except Exception:
                continue
        
        return monthly_scenes

    @staticmethod
    def _get_scene_dimensions_static(city: str, scene_path: str) -> Tuple[int, int]:
        """Static method to get scene dimensions for tile calculation"""
        scene_dir = Path(scene_path)
        for band_file in scene_dir.glob("*.tif"):
            try:
                with rasterio.open(band_file) as src:
                    return src.width, src.height
            except:
                continue
        return 128, 128  # fallback

    @staticmethod
    def _get_tiles_static(img_size: Tuple[int, int], tile_size: int, tile_overlap: float) -> List[Tuple[int, int, int, int]]:
        """Static method to generate tile coordinates"""
        tiles = []
        img_w, img_h = img_size
        
        stride_w = int((1 - tile_overlap) * tile_size)
        stride_h = int((1 - tile_overlap) * tile_size)
        
        for y in range(0, img_h - tile_size + 1, stride_h):
            for x in range(0, img_w - tile_size + 1, stride_w):
                x2 = x + tile_size
                y2 = y + tile_size
                
                if x2 <= img_w and y2 <= img_h:
                    tiles.append((x, y, x2, y2))
        
        return tiles

    @staticmethod
    def _are_consecutive_months_static(months: List[str]) -> bool:
        """Static method to check if months are consecutive"""
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

    @staticmethod
    def _is_scene_interpolated_static(city: str, scene_date: str, interpolated_scenes: set) -> bool:
        """Static method to check if scene is interpolated"""
        scene_id = f"{city}/{scene_date}"
        return scene_id in interpolated_scenes

    @staticmethod
    def _verify_tile_files_exist_static(
        city: str, 
        x_min: int, 
        y_min: int, 
        x_max: int, 
        y_max: int, 
        months: List[str],
        monthly_scenes: Dict[str, str],
        dataset_root: Path
    ) -> bool:
        """Static method to verify that all required band files exist for the tile region"""
        for month in months:
            if month not in monthly_scenes:
                return False
            
            scene_path = Path(monthly_scenes[month])
            
            # Check that essential bands exist
            essential_bands = ['LST', 'red', 'green', 'blue']
            for band in essential_bands:
                band_file = scene_path / f"{band}.tif"
                if not band_file.exists():
                    return False
        
        return True

    def _build_efficient_tile_sequences(self) -> List[Tuple[str, int, int, int, int, List[str], List[str]]]:
        """Build tile sequences using efficient spatial tiling approach with parallel processing"""
        print(f"\n🔄 Building efficient tile sequences for {self.split} split...")
        
        if not self.cities:
            return []
        
        # Use multiprocessing to process cities in parallel
        num_cpu = min(cpu_count(), len(self.cities))
        print(f"Using {num_cpu} CPU cores for parallel processing")
        
        # Create partial function with all necessary parameters
        process_city_func = partial(
            self._process_city_sequences_static,
            dataset_root=str(self.dataset_root),
            total_sequence_length=self.total_sequence_length,
            input_sequence_length=self.input_sequence_length,
            tile_size=self.tile_size,
            tile_overlap=self.tile_overlap,
            years=self.years,
            allowed_months=getattr(self, 'allowed_months', None),
            debug_monthly_split=self.debug_monthly_split,
            interpolated_scenes=self.interpolated_scenes
        )
        
        # Process cities in parallel
        with Pool(processes=num_cpu) as pool:
            city_results = list(tqdm(
                pool.imap(process_city_func, self.cities, chunksize=1),
                total=len(self.cities),
                desc=f"Processing cities ({self.split})"
            ))
        
        # Flatten results from all cities
        sequences = []
        for city_sequences in city_results:
            if city_sequences:
                sequences.extend(city_sequences)
        
        print(f"✅ Built {len(sequences)} tile sequences from {len(self.cities)} cities")
        return sequences

    def _verify_tile_files_exist(self, city: str, x_min: int, y_min: int, x_max: int, y_max: int, months: List[str]) -> bool:
        """Verify that all required band files exist for the tile region"""
        monthly_scenes = self._get_monthly_scenes(city)
        
        for month in months:
            if month not in monthly_scenes:
                return False
            
            scene_path = Path(monthly_scenes[month])
            
            # Check that essential bands exist
            essential_bands = ['LST', 'red', 'green', 'blue']
            for band in essential_bands:
                band_file = scene_path / f"{band}.tif"
                if not band_file.exists():
                    return False
        
        return True

    def _load_tile_from_scene(self, scene_path: str, band_name: str, x_min: int, y_min: int, x_max: int, y_max: int) -> np.ndarray:
        """Load specific tile region using native int16 from TIF files"""
        scene_dir = Path(scene_path)
        
        if band_name == 'DEM':
            # Load DEM from separate directory structure
            dem_path = self.dataset_root / "DEM_2014" / scene_dir.parent.parent.name / "DEM.tif"
            file_path = dem_path
        else:
            file_path = scene_dir / f"{band_name}.tif"
        
        try:
            window = Window(col_off=x_min, row_off=y_min, width=x_max-x_min, height=y_max-y_min)
            
            with rasterio.open(file_path) as src:
                # Keep native int16 from TIF files
                tile_data = src.read(1, window=window)  # Keep original dtype (int16)
                
                # Handle nodata values (keep as int16)
                if src.nodata is not None:
                    tile_data[tile_data == src.nodata] = int(self.nodata_fill_value)
                
                # Ensure int16 dtype
                if tile_data.dtype != np.int16:
                    tile_data = tile_data.astype(np.int16)
                
                return tile_data
                
        except Exception as e:
            return np.full((y_max - y_min, x_max - x_min), int(self.nodata_fill_value), dtype=np.int16)

    def _normalize_scene(self, scene_data: np.ndarray) -> np.ndarray:
        """Normalize scene data - convert int16 to normalized float32 for training"""
        # Convert from int16 to normalized float32 for training
        normalized_scene = np.zeros(scene_data.shape, dtype=np.float32)
        
        for i, band_name in enumerate(self.band_names):
            band_data = scene_data[:, :, i].astype(np.float32)  # Convert to float for normalization
            band_range = BAND_RANGES[band_name]
            
            # Normalize to [0, 1], keeping NODATA as 0
            valid_mask = band_data != self.nodata_fill_value
            normalized_band = np.full_like(band_data, 0.0, dtype=np.float32)  # Use 0.0 for nodata in normalized space
            normalized_band[valid_mask] = (band_data[valid_mask] - band_range["min"]) / (band_range["max"] - band_range["min"])
            normalized_band = np.clip(normalized_band, 0, 1)
            
            normalized_scene[:, :, i] = normalized_band
        
        return normalized_scene

    @staticmethod
    def _load_sequence_static(sequence_info, dataset_root, band_names, input_length, output_length, nodata_value):
        """Static method for parallel loading using native int16"""
        city, x_min, y_min, x_max, y_max, input_months, output_months = sequence_info
        dataset_root = Path(dataset_root)
        
        try:
            # Load input sequence
            input_scenes = []
            for month in input_months:
                scene_data = LandsatSequenceDataset._load_scene_tile_static(
                    dataset_root, city, month, x_min, y_min, x_max, y_max, band_names, nodata_value
                )
                if scene_data is None:
                    return None
                
                normalized_scene = LandsatSequenceDataset._normalize_scene_static(scene_data, band_names, nodata_value)
                input_scenes.append(normalized_scene)
            
            # Load output sequence (LST only)
            output_scenes = []
            for month in output_months:
                scene_data = LandsatSequenceDataset._load_scene_tile_static(
                    dataset_root, city, month, x_min, y_min, x_max, y_max, band_names, nodata_value
                )
                if scene_data is None:
                    return None
                
                normalized_scene = LandsatSequenceDataset._normalize_scene_static(scene_data, band_names, nodata_value)
                lst_only = normalized_scene[:, :, 1:2]  # LST band only
                output_scenes.append(lst_only)
            
            return input_scenes, output_scenes
            
        except Exception as e:
            return None

    @staticmethod
    def _load_scene_tile_static(dataset_root, city, month, x_min, y_min, x_max, y_max, band_names, nodata_value):
        """Static method to load a single scene tile using native int16"""
        try:
            # Get scene path
            city_dir = dataset_root / "Dataset" / "Cities_Processed" / city
            monthly_scenes = {}
            
            for scene_dir in city_dir.iterdir():
                if scene_dir.is_dir():
                    try:
                        date_obj = datetime.fromisoformat(scene_dir.name.replace('Z', '+00:00'))
                        month_key = f"{date_obj.year}-{date_obj.month:02d}"
                        if month_key == month:
                            monthly_scenes[month] = str(scene_dir)
                            break
                    except:
                        continue
            
            if month not in monthly_scenes:
                return None
            
            scene_path = Path(monthly_scenes[month])
            
            # Load all bands for this tile
            bands = []
            for band_name in band_names:
                if band_name == 'DEM':
                    # Load DEM from separate location
                    dem_path = dataset_root / "Dataset" / "DEM_2014_Preprocessed" / city / "DEM.tif"
                    file_path = dem_path
                else:
                    file_path = scene_path / f"{band_name}.tif"
                
                # Load tile using window
                try:
                    window = Window(col_off=x_min, row_off=y_min, width=x_max-x_min, height=y_max-y_min)
                    with rasterio.open(file_path) as src:
                        # Keep native int16
                        band_data = src.read(1, window=window)  # Keep original dtype
                        if src.nodata is not None:
                            band_data[band_data == src.nodata] = int(nodata_value)
                        if band_data.dtype != np.int16:
                            band_data = band_data.astype(np.int16)
                        
                        bands.append(band_data)
                except:
                    # Create fallback data if file doesn't exist
                    fallback_data = np.full((y_max - y_min, x_max - x_min), int(nodata_value), dtype=np.int16)
                    bands.append(fallback_data)
            
            return np.stack(bands, axis=-1)
            
        except Exception:
            return None

    @staticmethod
    def _normalize_scene_static(scene_data, band_names, nodata_value):
        """Static normalization method - convert int16 to normalized float32"""
        # Convert from int16 to normalized float32
        normalized_scene = np.zeros(scene_data.shape, dtype=np.float32)
        
        for i, band_name in enumerate(band_names):
            band_data = scene_data[:, :, i].astype(np.float32)  # Convert to float for normalization
            band_range = BAND_RANGES[band_name]
            
            valid_mask = band_data != nodata_value
            normalized_band = np.full_like(band_data, 0.0, dtype=np.float32)  # Use 0.0 for nodata in normalized space
            normalized_band[valid_mask] = (band_data[valid_mask] - band_range["min"]) / (band_range["max"] - band_range["min"])
            normalized_band = np.clip(normalized_band, 0, 1)
            
            normalized_scene[:, :, i] = normalized_band
        
        return normalized_scene

    # Include all other necessary methods from original Repo 2
    def _setup_debug_monthly_splits(self):
        """Setup debug monthly splits within a single year"""
        train_months = [1, 2, 3, 4, 5, 6, 7, 8]
        val_months = [6, 7, 8, 9, 10]
        test_months = [8, 9, 10, 11, 12]
        
        if self.split == 'train':
            self.allowed_months = set(train_months)
        elif self.split == 'val':
            self.allowed_months = set(val_months)
        else:
            self.allowed_months = set(test_months)
        
        self.years = {self.debug_year}

    def _setup_year_based_splits(self, train_years, val_years, test_years):
        """Setup normal year-based splits"""
        if train_years is None or val_years is None or test_years is None:
            all_years = list(range(2013, 2026))
            n_years = len(all_years)
            
            train_end = int(0.7 * n_years)
            val_end = int(0.85 * n_years)
            
            self.train_years = set(all_years[:train_end])
            self.val_years = set(all_years[train_end:val_end])
            self.test_years = set(all_years[val_end:])
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
        """Get all available cities"""
        cities_dir = self.dataset_root / "Dataset" / "Cities_Processed"
        cities = [d.name for d in cities_dir.iterdir() if d.is_dir()]
        texas = []
        for city in cities:
            if "_TX" in city:
                texas.append(city)
        # return sorted(cities)
        print("Loading", texas)
        return sorted(texas)

    def _get_monthly_scenes(self, city: str) -> Dict[str, str]:
        """Get monthly scenes for a city"""
        city_dir = self.dataset_root / "Dataset" / "Cities_Processed" / city
        if not city_dir.exists():
            return {}
        
        monthly_scenes = {}
        scene_dirs = [d for d in city_dir.iterdir() if d.is_dir()]
        
        for scene_dir in scene_dirs:
            try:
                date_str = scene_dir.name
                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                
                if date_obj.year not in self.years:
                    continue
                
                if self.debug_monthly_split and self.allowed_months is not None:
                    if date_obj.month not in self.allowed_months:
                        continue
                
                month_key = f"{date_obj.year}-{date_obj.month:02d}"
                
                if month_key not in monthly_scenes:
                    monthly_scenes[month_key] = str(scene_dir)
                        
            except Exception:
                continue
        
        return monthly_scenes

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

    def _is_scene_interpolated(self, city: str, scene_date: str) -> bool:
        """Check if scene is interpolated"""
        scene_id = f"{city}/{scene_date}"
        return scene_id in self.interpolated_scenes

    def _print_filtering_stats(self):
        """Print filtering statistics"""
        if self.interpolated_scenes:
            print(f"Loaded {len(self.interpolated_scenes)} interpolated scenes for filtering")
            print(f"Valid sequences after filtering: {len(self.tile_sequences)}")

    def __len__(self) -> int:
        return len(self.tile_sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Fallback to disk loading
        if idx >= len(self.tile_sequences):
            raise IndexError(f"Index {idx} out of range for tile sequences length {len(self.tile_sequences)}")
        
        city, x_min, y_min, x_max, y_max, input_months, output_months = self.tile_sequences[idx]
        
        # Load input sequence
        input_scenes = []
        for month in input_months:
            scene_data = self._load_scene_tile_from_disk(city, month, x_min, y_min, x_max, y_max)
            normalized_scene = self._normalize_scene(scene_data)
            input_scenes.append(normalized_scene)
        
        # Load output sequence
        output_scenes = []
        for month in output_months:
            scene_data = self._load_scene_tile_from_disk(city, month, x_min, y_min, x_max, y_max)
            normalized_scene = self._normalize_scene(scene_data)
            lst_only = normalized_scene[:, :, 1:2]  # LST band only
            output_scenes.append(lst_only)
        
        target_dtype = torch.float32 if str(self.precision) == "32" else torch.float16
        input_tensor = torch.from_numpy(np.stack(input_scenes, axis=0)).to(target_dtype)
        target_tensor = torch.from_numpy(np.stack(output_scenes, axis=0)).to(target_dtype)
        
        return input_tensor, target_tensor

    def _load_scene_tile_from_disk(self, city: str, month: str, x_min: int, y_min: int, x_max: int, y_max: int) -> np.ndarray:
        """Load scene tile from disk with efficient windowed reading"""
        monthly_scenes = self._get_monthly_scenes(city)
        scene_path = Path(monthly_scenes[month])
        
        bands = []
        for band_name in self.band_names:
            tile_data = self._load_tile_from_scene(str(scene_path), band_name, x_min, y_min, x_max, y_max)
            bands.append(tile_data)
        
        return np.stack(bands, axis=-1)

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
        tile_size: int = 128,
        tile_overlap: float = 0.0,
        precision: str = "32"
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
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.precision = precision

    def setup(self, stage: Optional[str] = None):
        """Setup datasets with fast loading"""
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
                tile_size=self.tile_size,
                tile_overlap=self.tile_overlap,
                precision=self.precision
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
                tile_size=self.tile_size,
                tile_overlap=self.tile_overlap,
                precision=self.precision
            )
        
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
                tile_size=self.tile_size,
                tile_overlap=self.tile_overlap,
                precision=self.precision
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