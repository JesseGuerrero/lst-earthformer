# ML Project Directory Structure

Based on your files and `.gitignore`, here's the current and expected directory structure:

```
ML/
├── .gitignore                    # Git ignore rules
├── main.ipynb                   # Main training notebook
├── dataset.py                   # Dataset and DataModule classes
├── model.py                     # Lightning model implementation
├── stac_scrapper.ipynb         # Data collection from STAC catalogs
│
├── Data/                        # Main data directory (gitignored)
│   ├── Dataset/                 # Raw dataset structure
│   │   ├── Cities/             # Monthly Landsat scenes by city
│   │   │   ├── {city_name}/
│   │   │   │   └── {YYYY-MM-DDTHH:MM:SSZ}/
│   │   │   │       ├── LST.tif
│   │   │   │       ├── red.tif
│   │   │   │       ├── green.tif
│   │   │   │       ├── blue.tif
│   │   │   │       ├── ndvi.tif
│   │   │   │       ├── ndwi.tif
│   │   │   │       ├── ndbi.tif
│   │   │   │       └── albedo.tif
│   │   │   └── ...
│   │   ├── DEM_2014/           # Digital Elevation Models
│   │   │   ├── {city_name}/
│   │   │   │   └── DEM.tif
│   │   │   └── ...
│   │   ├── Cities_Tiles/       # Tiled version for training
│   │   │   ├── {city_name}/
│   │   │   │   └── {YYYY-MM-DDTHH:MM:SSZ}/
│   │   │   │       ├── LST_row_000_col_001.tif
│   │   │   │       ├── red_row_000_col_001.tif
│   │   │   │       └── ... (all bands as tiles)
│   │   │   └── ...
│   │   ├── DEM_2014_Tiles/     # Tiled DEM data
│   │   │   ├── {city_name}/
│   │   │   │   ├── DEM_row_000_col_001.tif
│   │   │   │   └── ...
│   │   │   └── ...
│   │   ├── City_Shapes/        # Shapefiles for city boundaries
│   │   │   ├── {city_name}.shp
│   │   │   ├── {city_name}.shx
│   │   │   ├── {city_name}.dbf
│   │   │   └── ...
│   │   └── checkpoint.json     # Processing checkpoint file
│   │
│   └── ML/                     # Processed ML-ready data (gitignored)
│       ├── Cities_Tiles/       # Same structure as Dataset/Cities_Tiles
│       ├── DEM_2014_Tiles/     # Same structure as Dataset/DEM_2014_Tiles
│       └── ...
│
├── logs/                       # Training logs (gitignored)
│   ├── tensorboard/
│   ├── wandb/
│   └── ...
│
├── checkpoints/                # Model checkpoints (gitignored)
│   ├── {experiment_name}-epoch={epoch}-val_loss={loss}.ckpt
│   ├── last.ckpt
│   └── ...
│
├── wandb/                      # Weights & Biases logs (gitignored)
│   └── ...
│
└── __pycache__/               # Python cache (gitignored)
    └── ...
```

## Key Components

### Data Pipeline
- **Raw Data**: `Data/Dataset/` contains original Landsat scenes and DEM data
- **Tiled Data**: `Data/Dataset/*_Tiles/` contains 128x128 tiles for efficient training
- **ML Data**: `Data/ML/` is the processed dataset used for training

### Model Architecture
- **Earthformer**: Cuboid Transformer for spatiotemporal prediction
- **Input**: 3 timesteps × 128×128 × 9 bands (DEM, LST, RGB, NDVI, NDWI, NDBI, Albedo)
- **Output**: 3 timesteps × 128×128 × 1 band (LST prediction)

### Training Configuration
- **Temporal Splits**: Year-based (2013-2025) or debug monthly splits
- **Sequence Length**: 3 input + 3 output timesteps (configurable)
- **Batch Processing**: Parallel tile processing with multiprocessing

### Data Splits
- **Year-based**: Train (2013-2021), Val (2022-2023), Test (2024-2025)
- **Debug Monthly**: Within single year for rapid prototyping
  - Train: Jan-Aug
  - Val: Jun-Oct  
  - Test: Aug-Dec

### Bands & Normalization
All bands are normalized to [0,1] using predefined ranges:
- **DEM**: 9899-13110 (with +10k offset)
- **LST**: -189 to 211°F
- **RGB**: 1-10000 (reflectance × 10000)
- **Indices**: -10000 to 10000 (NDVI/NDWI/NDBI × 10000)
- **Albedo**: 1-9980 (albedo × 10000)

### Dataset Size
124 Cities, minimum 90 square miles each city, 30m spatial resolution, monthly temporal resolution, Landsat data. Each tif is 128x128 pixels from the city satellite scenes at int16.