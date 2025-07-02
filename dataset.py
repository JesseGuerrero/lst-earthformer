import os
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import glob
from collections import defaultdict

class LandsatSequenceDataset(Dataset):
    def __init__(
        self, 
        dataset_root: str,
        sequence_length: int = 3,
        split: str = 'train',
        train_years: Optional[List[int]] = None,
        val_years: Optional[List[int]] = None,
        test_years: Optional[List[int]] = None,
        # New debug option
        debug_monthly_split: bool = False,
        debug_year: int = 2014
    ):
        """
        Dataset for Landsat sequence prediction using tiled data with year-based splits
        
        Args:
            dataset_root: Path to Dataset folder containing Cities_Tiles and DEM_2014_Tiles
            sequence_length: Number of consecutive months (input=3, output=3)
            split: 'train', 'val', or 'test'
            train_years/val_years/test_years: Year lists for each split
            debug_monthly_split: If True, use monthly splits within debug_year
            debug_year: Year to use for debug monthly splits (default: 2014)
        """
        self.dataset_root = Path(dataset_root)
        self.sequence_length = sequence_length
        self.split = split
        self.debug_monthly_split = debug_monthly_split
        self.debug_year = debug_year
        
        if debug_monthly_split:
            # Debug mode: Use monthly splits within a single year
            self._setup_debug_monthly_splits()
        else:
            # Normal mode: Use year-based splits
            self._setup_year_based_splits(train_years, val_years, test_years)
        
        # Get all available cities (we still process all cities, just filter by years/months)
        self.cities = self._get_all_cities()
        
        # Band names (for tiled files)
        self.band_names = ['DEM', 'LST', 'red', 'green', 'blue', 'ndvi', 'ndwi', 'ndbi', 'albedo']
        
        # Build tile sequences filtered by years/months
        self.tile_sequences = self._build_tile_sequences()
        
        if debug_monthly_split:
            print(f"DEBUG {split} split: {len(self.cities)} cities, year {debug_year}, "
                  f"months {sorted(self.allowed_months)}, {len(self.tile_sequences)} tile sequences")
        else:
            print(f"{split} split: {len(self.cities)} cities, {len(self.years)} years "
                  f"({min(self.years)}-{max(self.years)}), {len(self.tile_sequences)} tile sequences")
    
    def _setup_debug_monthly_splits(self):
        """
        Setup debug monthly splits within a single year:
        Train: months [1,2,3,4,5,6,7,8] (Jan-Aug)
        Val: months [6,7,8,9,10] (Jun-Oct) 
        Test: months [8,9,10,11,12] (Aug-Dec)
        
        Note: Overlapping months are needed for sequence continuity
        """
        train_months = [1, 2, 3, 4, 5, 6, 7, 8]
        val_months = [6, 7, 8, 9, 10]
        test_months = [8, 9, 10, 11, 12]
        
        # Set allowed months for current split
        if self.split == 'train':
            self.allowed_months = set(train_months)
        elif self.split == 'val':
            self.allowed_months = set(val_months)
        else:  # test
            self.allowed_months = set(test_months)
        
        # Only use the debug year
        self.years = {self.debug_year}
        
        # Store all split info for reference
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
        # Define year splits based on temporal coverage (May 2013 - May 2025)
        if train_years is None or val_years is None or test_years is None:
            # Create chronological year splits: 70/15/15 temporal split
            # Available years: 2013-2025 (13 years total)
            all_years = list(range(2013, 2026))  # 2013 to 2025 inclusive
            n_years = len(all_years)
            
            # Calculate split points
            train_end = int(0.7 * n_years)  # 70% = ~9 years
            val_end = int(0.85 * n_years)   # 15% = ~2 years
            
            self.train_years = set(all_years[:train_end])           # 2013-2021 (9 years)
            self.val_years = set(all_years[train_end:val_end])      # 2022-2023 (2 years) 
            self.test_years = set(all_years[val_end:])              # 2024-2025 (2 years)
        else:
            self.train_years = set(train_years)
            self.val_years = set(val_years)
            self.test_years = set(test_years)
        
        # Get years for current split
        if self.split == 'train':
            self.years = self.train_years
        elif self.split == 'val':
            self.years = self.val_years
        else:
            self.years = self.test_years
        
        # No month filtering in normal mode
        self.allowed_months = None

    def _get_all_cities(self) -> List[str]:
        """Get all available cities from the tiled dataset"""
        cities_dir = self.dataset_root / "Cities_Tiles"
        cities = [d.name for d in cities_dir.iterdir() if d.is_dir()]
        return sorted(cities)
    
    def _get_available_tiles(self, city: str) -> Dict[Tuple[int, int], bool]:
        """
        Get all available tile positions for a city
        Returns: {(row, col): True} for available tiles
        """
        # Check DEM tiles first to get available tile positions
        dem_dir = self.dataset_root / "DEM_2014_Tiles" / city
        available_tiles = {}
        
        if dem_dir.exists():
            for dem_file in dem_dir.glob("DEM_row_*_col_*.tif"):
                # Parse row and col from filename: DEM_row_000_col_001.tif
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
        """
        Get one scene per month for a city from tiled data, filtered by years and months
        Returns: {YYYY-MM: scene_path}
        """
        city_dir = self.dataset_root / "Cities_Tiles" / city
        if not city_dir.exists():
            return {}
        
        monthly_scenes = {}
        
        # Get all scene directories
        scene_dirs = [d for d in city_dir.iterdir() if d.is_dir()]
        
        for scene_dir in scene_dirs:
            try:
                # Parse datetime from folder name
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
                    # Verify scene has tiles
                    if self._validate_tiled_scene(scene_dir):
                        monthly_scenes[month_key] = str(scene_dir)
                        
            except Exception as e:
                print(f"Warning: Could not parse date from {scene_dir.name}: {e}")
                continue
        
        return monthly_scenes
    
    def _validate_tiled_scene(self, scene_dir: Path) -> bool:
        """Check if scene has tile files"""
        # Check if there are any .tif files in the scene directory
        tif_files = list(scene_dir.glob("*.tif"))
        return len(tif_files) > 0
    
    def _build_tile_sequences(self) -> List[Tuple[str, int, int, List[str], List[str]]]:
        """
        Build consecutive monthly sequences for each tile position, filtered by years/months
        Returns: List of (city, tile_row, tile_col, input_months, output_months)
        """
        sequences = []
        
        if self.debug_monthly_split:
            month_stats = {month: 0 for month in self.allowed_months}  # Track sequences per month
        else:
            year_stats = {year: 0 for year in self.years}  # Track sequences per year
        
        for city in self.cities:
            # Get available tile positions for this city
            available_tiles = self._get_available_tiles(city)
            
            # Get monthly scenes (already filtered by years/months)
            monthly_scenes = self._get_monthly_scenes(city)
            
            if len(monthly_scenes) < 2 * self.sequence_length:
                continue
            
            # Sort months chronologically
            sorted_months = sorted(monthly_scenes.keys())
            
            # For each tile position
            for (tile_row, tile_col) in available_tiles.keys():
                # Find consecutive sequences for this tile
                for i in range(len(sorted_months) - 2 * self.sequence_length + 1):
                    input_months = sorted_months[i:i + self.sequence_length]
                    output_months = sorted_months[i + self.sequence_length:i + 2 * self.sequence_length]
                    
                    # Verify months are consecutive
                    if self._are_consecutive_months(input_months + output_months):
                        # Verify all required tiles exist for this sequence
                        if self._verify_tile_sequence_exists(city, tile_row, tile_col, input_months + output_months):
                            sequences.append((city, tile_row, tile_col, input_months, output_months))
                            
                            # Track statistics
                            if self.debug_monthly_split:
                                # Track by first input month
                                first_month = int(input_months[0].split('-')[1])
                                if first_month in month_stats:
                                    month_stats[first_month] += 1
                            else:
                                # Track by first input year
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
        """Verify that all required tiles exist for a sequence"""
        monthly_scenes = self._get_monthly_scenes(city)
        
        for month in months:
            if month not in monthly_scenes:
                return False
            
            scene_dir = Path(monthly_scenes[month])
            
            # Check if LST tile exists (minimum requirement)
            lst_tile = scene_dir / f"LST_row_{tile_row:03d}_col_{tile_col:03d}.tif"
            if not lst_tile.exists():
                return False
        
        return True
    
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
            
            # Calculate expected next month
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
                # Replace nodata with 0
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
        """
        Load all bands for a scene tile
        Returns: (H, W, 9) array with bands in order: DEM, LST, red, green, blue, ndvi, ndwi, ndbi, albedo
        """
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
        """
        Returns:
            input_sequence: (T, H, W, C) = (3, 128, 128, 9)
            target_sequence: (T, H, W, 1) = (3, 128, 128, 1) - LST only
        """
        city, tile_row, tile_col, input_months, output_months = self.tile_sequences[idx]
        
        # Load input sequence
        input_scenes = []
        for month in input_months:
            scene = self._load_scene_tile(city, month, tile_row, tile_col)
            input_scenes.append(scene)
        
        # Load output sequence (LST only)
        output_scenes = []
        for month in output_months:
            scene = self._load_scene_tile(city, month, tile_row, tile_col)
            lst_only = scene[:, :, 1:2]  # LST is index 1, keep dims
            output_scenes.append(lst_only)
        
        # Convert to tensors
        input_tensor = torch.from_numpy(np.stack(input_scenes, axis=0))  # (T, H, W, C)
        target_tensor = torch.from_numpy(np.stack(output_scenes, axis=0))  # (T, H, W, 1)
        
        return input_tensor, target_tensor


# Updated LandsatDataModule class
class LandsatDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        sequence_length: int = 3,
        train_years: Optional[List[int]] = None,
        val_years: Optional[List[int]] = None,
        test_years: Optional[List[int]] = None,
        # New debug option
        debug_monthly_split: bool = False,
        debug_year: int = 2014
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sequence_length = sequence_length
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years
        self.debug_monthly_split = debug_monthly_split
        self.debug_year = debug_year
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage with year-based or debug monthly splits"""
        if stage == "fit" or stage is None:
            self.train_dataset = LandsatSequenceDataset(
                self.dataset_root,
                sequence_length=self.sequence_length,
                split='train',
                train_years=self.train_years,
                val_years=self.val_years,
                test_years=self.test_years,
                debug_monthly_split=self.debug_monthly_split,
                debug_year=self.debug_year
            )
            
            self.val_dataset = LandsatSequenceDataset(
                self.dataset_root,
                sequence_length=self.sequence_length,
                split='val',
                train_years=self.train_years,
                val_years=self.val_years,
                test_years=self.test_years,
                debug_monthly_split=self.debug_monthly_split,
                debug_year=self.debug_year
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = LandsatSequenceDataset(
                self.dataset_root,
                sequence_length=self.sequence_length,
                split='test',
                train_years=self.train_years,
                val_years=self.val_years,
                test_years=self.test_years,
                debug_monthly_split=self.debug_monthly_split,
                debug_year=self.debug_year
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


# Debug usage examples
def create_debug_monthly_split_example():
    """Example of how to use debug monthly splits"""
    
    print("Creating debug monthly split data module...")
    
    # Debug monthly split within 2014
    debug_data_module = LandsatDataModule(
        dataset_root="./Data/Dataset",
        batch_size=2,
        sequence_length=3,
        debug_monthly_split=True,
        debug_year=2014
    )
    
    # Test the setup
    debug_data_module.setup("fit")
    
    print(f"Debug train sequences: {len(debug_data_module.train_dataset.tile_sequences)}")
    print(f"Debug val sequences: {len(debug_data_module.val_dataset.tile_sequences)}")
    
    # Check a sample
    train_loader = debug_data_module.train_dataloader()
    if len(train_loader) > 0:
        sample_batch = next(iter(train_loader))
        inputs, targets = sample_batch
        print(f"Sample batch - Inputs: {inputs.shape}, Targets: {targets.shape}")
        
        # Show which months are being used in the sample
        sample_seq = debug_data_module.train_dataset.tile_sequences[0]
        city, tile_row, tile_col, input_months, output_months = sample_seq
        print(f"Sample sequence: {city} tile({tile_row},{tile_col})")
        print(f"  Input months: {input_months}")
        print(f"  Output months: {output_months}")
    
    return debug_data_module


def compare_normal_vs_debug_splits():
    """Compare normal year-based splits vs debug monthly splits"""
    
    print("=== COMPARISON: Normal vs Debug Splits ===\n")
    
    # Normal year-based split
    print("1. Normal year-based split:")
    normal_dm = LandsatDataModule(
        dataset_root="./Data/Dataset",
        batch_size=2,
        sequence_length=3,
        train_years=[2014],  # Use only 2014 for fair comparison
        val_years=[2015],
        test_years=[2016]
    )
    normal_dm.setup("fit")
    print(f"   Train sequences: {len(normal_dm.train_dataset.tile_sequences)}")
    print(f"   Val sequences: {len(normal_dm.val_dataset.tile_sequences)}")
    
    # Debug monthly split
    print("\n2. Debug monthly split (2014 only):")
    debug_dm = LandsatDataModule(
        dataset_root="./Data/Dataset",
        batch_size=2,
        sequence_length=3,
        debug_monthly_split=True,
        debug_year=2014
    )
    debug_dm.setup("fit")
    print(f"   Train sequences: {len(debug_dm.train_dataset.tile_sequences)}")
    print(f"   Val sequences: {len(debug_dm.val_dataset.tile_sequences)}")
    
    print("\nDebug splits allow for rapid prototyping with smaller datasets!")
    return normal_dm, debug_dm
