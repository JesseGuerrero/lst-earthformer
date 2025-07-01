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
        train_cities: Optional[List[str]] = None,
        val_cities: Optional[List[str]] = None,
        test_cities: Optional[List[str]] = None
    ):
        """
        Dataset for Landsat sequence prediction
        
        Args:
            dataset_root: Path to Dataset folder
            sequence_length: Number of consecutive months (input=3, output=3)
            split: 'train', 'val', or 'test'
            train_cities/val_cities/test_cities: City lists for each split
        """
        self.dataset_root = Path(dataset_root)
        self.sequence_length = sequence_length
        self.split = split
        
        # Define city splits
        if train_cities is None or val_cities is None or test_cities is None:
            all_cities = self._get_all_cities()
            # Default 70/15/15 split
            n_cities = len(all_cities)
            train_end = int(0.7 * n_cities)
            val_end = int(0.85 * n_cities)
            
            self.train_cities = set(all_cities[:train_end])
            self.val_cities = set(all_cities[train_end:val_end])
            self.test_cities = set(all_cities[val_end:])
        else:
            self.train_cities = set(train_cities)
            self.val_cities = set(val_cities)
            self.test_cities = set(test_cities)
        
        # Get cities for current split
        if split == 'train':
            self.cities = self.train_cities
        elif split == 'val':
            self.cities = self.val_cities
        else:
            self.cities = self.test_cities
        
        # Band order: DEM, LST, red, green, blue, ndvi, ndwi, ndbi, albedo
        self.band_files = ['DEM.tif', 'LST.tif', 'red.tif', 'green.tif', 'blue.tif', 
                          'ndvi.tif', 'ndwi.tif', 'ndbi.tif', 'albedo.tif']
        
        # Build sequences
        self.sequences = self._build_sequences()
        
        print(f"{split} split: {len(self.cities)} cities, {len(self.sequences)} sequences")
    
    def _get_all_cities(self) -> List[str]:
        """Get all available cities from the dataset"""
        cities_dir = self.dataset_root / "Cities_Preprocessed"
        cities = [d.name for d in cities_dir.iterdir() if d.is_dir()]
        return sorted(cities)
    
    def _get_monthly_scenes(self, city: str) -> Dict[str, str]:
        """
        Get one scene per month for a city
        Returns: {YYYY-MM: scene_path}
        """
        city_dir = self.dataset_root / "Cities_Preprocessed" / city
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
                month_key = f"{date_obj.year}-{date_obj.month:02d}"
                
                # Only keep first scene per month
                if month_key not in monthly_scenes:
                    # Verify all required files exist
                    if self._validate_scene(scene_dir):
                        monthly_scenes[month_key] = str(scene_dir)
                        
            except Exception as e:
                print(f"Warning: Could not parse date from {scene_dir.name}: {e}")
                continue
        
        return monthly_scenes
    
    def _validate_scene(self, scene_dir: Path) -> bool:
        """Check if scene has all required files (except DEM)"""
        required_files = ['LST.tif', 'red.tif', 'green.tif', 'blue.tif', 
                         'ndvi.tif', 'ndwi.tif', 'ndbi.tif', 'albedo.tif']
        
        for file_name in required_files:
            if not (scene_dir / file_name).exists():
                return False
        return True
    
    def _build_sequences(self) -> List[Tuple[str, List[str], List[str]]]:
        """
        Build consecutive monthly sequences
        Returns: List of (city, input_months, output_months)
        """
        sequences = []
        
        for city in self.cities:
            monthly_scenes = self._get_monthly_scenes(city)
            
            if len(monthly_scenes) < 2 * self.sequence_length:
                continue
            
            # Sort months chronologically
            sorted_months = sorted(monthly_scenes.keys())
            
            # Find consecutive sequences
            for i in range(len(sorted_months) - 2 * self.sequence_length + 1):
                input_months = sorted_months[i:i + self.sequence_length]
                output_months = sorted_months[i + self.sequence_length:i + 2 * self.sequence_length]
                
                # Verify months are consecutive
                if self._are_consecutive_months(input_months + output_months):
                    sequences.append((city, input_months, output_months))
        
        return sequences
    
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
    
    def _load_dem(self, city: str) -> np.ndarray:
        """Load DEM for the city"""
        dem_path = self.dataset_root / "DEM_2014_Preprocessed" / city / "DEM.tif"
        return self._load_raster(str(dem_path))
    
    def _load_scene(self, city: str, month: str) -> np.ndarray:
        """
        Load all bands for a scene
        Returns: (H, W, 9) array with bands in order: DEM, LST, red, green, blue, ndvi, ndwi, ndbi, albedo
        """
        monthly_scenes = self._get_monthly_scenes(city)
        scene_dir = Path(monthly_scenes[month])
        
        # Load DEM (constant for city)
        dem = self._load_dem(city)
        
        # Load other bands
        bands = [dem]  # Start with DEM
        
        for band_file in self.band_files[1:]:  # Skip DEM.tif since we loaded it separately
            band_path = scene_dir / band_file
            band_data = self._load_raster(str(band_path))
            bands.append(band_data)
        
        # Stack to (H, W, C)
        scene_data = np.stack(bands, axis=-1)
        return scene_data
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_sequence: (T, H, W, C) = (3, 128, 128, 9)
            target_sequence: (T, H, W, 1) = (3, 128, 128, 1) - LST only
        """
        city, input_months, output_months = self.sequences[idx]
        
        # Load input sequence
        input_scenes = []
        for month in input_months:
            scene = self._load_scene(city, month)
            input_scenes.append(scene)
        
        # Load output sequence (LST only)
        output_scenes = []
        for month in output_months:
            scene = self._load_scene(city, month)
            lst_only = scene[:, :, 1:2]  # LST is index 1, keep dims
            output_scenes.append(lst_only)
        
        # Convert to tensors
        input_tensor = torch.from_numpy(np.stack(input_scenes, axis=0))  # (T, H, W, C)
        target_tensor = torch.from_numpy(np.stack(output_scenes, axis=0))  # (T, H, W, 1)
        
        return input_tensor, target_tensor


class LandsatDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        sequence_length: int = 3,
        train_cities: Optional[List[str]] = None,
        val_cities: Optional[List[str]] = None,
        test_cities: Optional[List[str]] = None
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sequence_length = sequence_length
        self.train_cities = train_cities
        self.val_cities = val_cities
        self.test_cities = test_cities
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage"""
        if stage == "fit" or stage is None:
            self.train_dataset = LandsatSequenceDataset(
                self.dataset_root,
                sequence_length=self.sequence_length,
                split='train',
                train_cities=self.train_cities,
                val_cities=self.val_cities,
                test_cities=self.test_cities
            )
            
            self.val_dataset = LandsatSequenceDataset(
                self.dataset_root,
                sequence_length=self.sequence_length,
                split='val',
                train_cities=self.train_cities,
                val_cities=self.val_cities,
                test_cities=self.test_cities
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = LandsatSequenceDataset(
                self.dataset_root,
                sequence_length=self.sequence_length,
                split='test',
                train_cities=self.train_cities,
                val_cities=self.val_cities,
                test_cities=self.test_cities
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


# Example usage
if __name__ == "__main__":
    # Test the dataloader
    data_module = LandsatDataModule(
        dataset_root="./Data/Dataset",
        batch_size=2,
        num_workers=2,
        sequence_length=3
    )
    
    data_module.setup("fit")
    
    # Test a batch
    train_loader = data_module.train_dataloader()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Input shape: {inputs.shape}")   # Should be (B, T, H, W, C) = (2, 3, 128, 128, 9)
        print(f"  Target shape: {targets.shape}") # Should be (B, T, H, W, 1) = (2, 3, 128, 128, 1)
        
        if batch_idx == 2:  # Test a few batches
            break