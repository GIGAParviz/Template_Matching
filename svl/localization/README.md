# Visual Localization Project

This project performs visual localization by matching drone images (query) with satellite images (reference).

## Setup

1. Install the required dependencies (assumes you have the main project dependencies installed):

```bash
pip install pandas numpy opencv-python torch
```

2. Organize your images in two separate folders:
   - Reference folder: contains satellite/map images
   - Query folder: contains drone/aerial images to be localized

## Usage with run_with_dummy_csv.py

If you don't have CSV metadata files for your images, you can use the `run_with_dummy_csv.py` script which will automatically create dummy CSV files with random metadata:

```bash
python run_with_dummy_csv.py --ref_folder /path/to/reference/images --query_folder /path/to/query/images --output_path /path/to/output
```

### Arguments

- `--ref_folder`: Path to the folder containing reference (satellite/map) images
- `--query_folder`: Path to the folder containing query (drone/aerial) images
- `--output_path`: (Optional) Path where results will be saved (defaults to "output")

## Notes

- The dummy CSV generation creates random GPS coordinates and orientations.
- Since the metadata is random, the computed metrics may not be meaningful.
- The script automatically detects if GPU is available and uses it; otherwise, it falls back to CPU.
- Image sizes are automatically detected from the query images to set appropriate camera parameters.

## Standard Usage (with real CSV files)

If you have proper CSV metadata files, place them in the respective folders:

- For reference images: A CSV with columns "Filename", "Top_left_lat", "Top_left_lon", "Bottom_right_lat", "Bottom_right_long"
- For query images: A CSV with columns "Filename", "Latitude", "Longitude", "Altitude", "Gimball_Roll", "Gimball_Yaw", "Gimball_Pitch", "Flight_Roll", "Flight_Yaw", "Flight_Pitch"

In this case, you can still use the same script, and it will use your real CSV files instead of creating dummy ones. 