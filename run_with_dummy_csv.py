import logging
import os
import gc
import shutil
from pathlib import Path
from pprint import pprint
import pandas as pd
import random
import cv2
import numpy as np
import torch
from svl.keypoint_pipeline.detection_and_description import SuperPointAlgorithm
from svl.keypoint_pipeline.matcher import SuperGlueMatcher
from svl.keypoint_pipeline.typing import SuperGlueConfig, SuperPointConfig
from svl.localization.drone_streamer import DroneImageStreamer
from svl.localization.map_reader import SatelliteMapReader
from svl.localization.pipeline import Pipeline, PipelineConfig
from svl.localization.preprocessing import QueryProcessor
from svl.tms.data_structures import CameraModel
from svl.tms.schemas import GpsCoordinate


torch.cuda.empty_cache()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def split_and_save_image(image_path, output_folder, grid_size=(2, 2), overlap=0):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to read image: {image_path}")
        return []

    height, width = img.shape[:2]

    tile_height = height // grid_size[0]
    tile_width = width // grid_size[1]

    overlap_h = min(overlap, tile_height // 2)
    overlap_w = min(overlap, tile_width // 2)

    output_paths = []

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):

            y_start = max(0, i * tile_height - overlap_h)
            y_end = min(height, (i + 1) * tile_height + overlap_h)
            x_start = max(0, j * tile_width - overlap_w)
            x_end = min(width, (j + 1) * tile_width + overlap_w)

            tile = img[y_start:y_end, x_start:x_end]

            base_name = Path(image_path).stem
            output_filename = f"{base_name}_tile_{i}_{j}.jpg"
            output_path = output_folder / output_filename

            cv2.imwrite(str(output_path), tile)
            output_paths.append(output_path)
            print(f"Saved tile to {output_path}")

    return output_paths

def create_dummy_ref_csv(ref_folder):
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        image_files.extend(list(Path(ref_folder).glob(f'*{ext}')))

    data = []
    for img_path in image_files:
        top_left_lat = random.uniform(35.0, 45.0)
        top_left_lon = random.uniform(-120.0, -110.0)
        bottom_right_lat = top_left_lat - random.uniform(0.01, 0.1)
        bottom_right_lon = top_left_lon + random.uniform(0.01, 0.1)

        data.append({
            "Filename": img_path.name,
            "Top_left_lat": top_left_lat,
            "Top_left_lon": top_left_lon,
            "Bottom_right_lat": bottom_right_lat,
            "Bottom_right_long": bottom_right_lon
        })

    df = pd.DataFrame(data)
    csv_path = os.path.join(ref_folder, "reference_metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"Created dummy reference CSV at {csv_path}")
    return df

def create_dummy_query_csv(query_folder):
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        image_files.extend(list(Path(query_folder).glob(f'*{ext}')))

    if not image_files:
        print(f"WARNING: No image files found in {query_folder}")
        return None

    data = []
    for img_path in image_files:
        data.append({
            "Filename": img_path.name,
            "Latitude": random.uniform(35.0, 45.0),
            "Longitude": random.uniform(-120.0, -110.0),
            "Altitude": random.uniform(100.0, 500.0),
            "Gimball_Roll": random.uniform(-10.0, 10.0),
            "Gimball_Yaw": random.uniform(-180.0, 180.0),
            "Gimball_Pitch": random.uniform(-90.0, 0.0),
            "Flight_Roll": random.uniform(-10.0, 10.0),
            "Flight_Yaw": random.uniform(-180.0, 180.0),
            "Flight_Pitch": random.uniform(-10.0, 10.0)
        })

    df = pd.DataFrame(data)
    csv_path = os.path.join(query_folder, "query_metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"Created dummy query CSV at {csv_path} with {len(data)} entries")
    return csv_path

def get_image_size(image_path):
    """Get the size of an image."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (1024, 1024)
    return img.shape[:2]

def manually_assign_metadata(map_reader, ref_folder):

    csv_files = list(Path(ref_folder).glob("*.csv"))

    if not csv_files:
        print("No metadata CSV found. Creating dummy metadata...")
        metadata_df = create_dummy_ref_csv(ref_folder)

    else:
        csv_path = csv_files[0]
        try:
            metadata_df = pd.read_csv(csv_path)
            print(f"Loaded metadata from {csv_path}")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            metadata_df = create_dummy_ref_csv(ref_folder)

    required_columns = ["Filename", "Top_left_lat", "Top_left_lon", "Bottom_right_lat", "Bottom_right_long"]
    if not all(col in metadata_df.columns for col in required_columns):
        print(f"CSV missing required columns. Creating new metadata...")
        metadata_df = create_dummy_ref_csv(ref_folder)

    for _, row in metadata_df.iterrows():
        filename = row["Filename"]
        if filename in map_reader._image_db:
            satellite_image = map_reader._image_db[filename]

            top_left = GpsCoordinate(
                lat=float(row["Top_left_lat"]),
                long=float(row["Top_left_lon"])
            )
            bottom_right = GpsCoordinate(
                lat=float(row["Bottom_right_lat"]),
                long=float(row["Bottom_right_long"])
            )

            satellite_image.top_left = top_left
            satellite_image.bottom_right = bottom_right

            print(f"Set metadata for {filename}: TL={top_left.lat},{top_left.long} BR={bottom_right.lat},{bottom_right.long}")

def checker(path):
    if len(list(path.glob("*.jpg"))) == 1:
        return True
    return False

def ensure_query_images(query_folder, ref_folder):
    """Check if query folder has images, if not, copy from ref_folder"""
    query_folder = Path(query_folder)
    ref_folder = Path(ref_folder)

    query_images = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        query_images.extend(list(query_folder.glob(f'*{ext}')))

    if not query_images:
        print(f"No images found in query folder {query_folder}. Copying from reference folder...")

        ref_images = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            ref_images.extend(list(ref_folder.glob(f'*{ext}')))

        if not ref_images:
            raise ValueError(f"No images found in reference folder {ref_folder} either!")

        query_folder.mkdir(parents=True, exist_ok=True)

        for img_path in ref_images[:2]:
            dest_path = query_folder / img_path.name
            shutil.copy2(img_path, dest_path)
            print(f"Copied {img_path.name} to query folder")

        query_images = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            query_images.extend(list(query_folder.glob(f'*{ext}')))

        if not query_images:
            raise ValueError("Failed to copy reference images to query folder!")

    return len(query_images)

def main(ref_folder, target_path, query_folder, output_path=None):
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    ref_folder = Path(ref_folder)
    target_path = Path(target_path)
    query_folder = Path(query_folder)

    ref_folder.mkdir(parents=True, exist_ok=True)
    target_path.mkdir(parents=True, exist_ok=True)
    query_folder.mkdir(parents=True, exist_ok=True)

    ref_images = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        ref_images.extend(list(ref_folder.glob(f'*{ext}')))

    if not ref_images:
        raise ValueError(f"No images found in reference folder {ref_folder}!")

    if checker(ref_folder):
        split_and_save_image(next(ref_folder.glob("*.jpg")), target_path, grid_size=(3, 3), overlap=20)
        ref_folder = Path(target_path)

    num_query_images = ensure_query_images(query_folder, ref_folder)
    print(f"Found {num_query_images} images in query folder")

    ref_csv_files = list(ref_folder.glob("*.csv"))
    if len(ref_csv_files) == 0:
        create_dummy_ref_csv(ref_folder)

    query_csv_files = list(query_folder.glob("*.csv"))
    if len(query_csv_files) == 0:
        create_dummy_query_csv(query_folder)

    superpoint_config = SuperPointConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        nms_radius=4,
        keypoint_threshold=0.01,
        max_keypoints=5000,
    )
    superpoint_algorithm = SuperPointAlgorithm(superpoint_config)

    superglue_config = SuperGlueConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        weights="outdoor",
        sinkhorn_iterations=20,
        match_threshold=0.5,
    )
    superglue_matcher = SuperGlueMatcher(superglue_config)

    map_reader = SatelliteMapReader(
        db_path=ref_folder,
        resize_size=None , #(384, 384)
        logger=logging.getLogger("SatelliteMapReader"),
        metadata_method="CSV",
    )
    map_reader.initialize_db()
    map_reader.setup_db()

    manually_assign_metadata(map_reader, ref_folder)

    map_reader.resize_db_images()

    with torch.cuda.amp.autocast():
        map_reader.describe_db_images(superpoint_algorithm)

    streamer = DroneImageStreamer(
        image_folder=query_folder,
        has_gt=True,
        logger=logging.getLogger("DroneImageStreamer"),
    )
    print(f"Found {len(streamer)} query images")

    if len(streamer) == 0:
        print("ERROR: No query images available. Please check your query folder.")
        return []

    sample_images = list(query_folder.glob("*.jpg")) or list(query_folder.glob("*.png"))
    if sample_images:
        sample_img_path = sample_images[0]
        img_height, img_width = get_image_size(sample_img_path)
    else:
        img_height, img_width = 1024, 1024

    camera_model = CameraModel(
        focal_length=4.5 / 1000,
        resolution_height=img_height,
        resolution_width=img_width,
        hfov_deg=82.9,
    )

    query_processor = QueryProcessor(
        processings=[],  
        camera_model=camera_model,
        satellite_resolution=None,
        size=None, 
    )

    logger = logging.getLogger("Pipeline")
    logger.setLevel(logging.DEBUG)

    original_compute_geo_pose = Pipeline.compute_geo_pose

    def patched_compute_geo_pose(self, satellite_image, matching_center):
        try:
            if satellite_image.top_left is None or satellite_image.bottom_right is None:
                print(f"WARNING: Missing metadata for image {satellite_image.name}, using dummy values")
                return GpsCoordinate(lat=37.0, long=-122.0)
            return original_compute_geo_pose(self, satellite_image, matching_center)
        except Exception as e:
            print(f"ERROR in compute_geo_pose: {e}")
            return GpsCoordinate(lat=37.0, long=-122.0)

    Pipeline.compute_geo_pose = patched_compute_geo_pose

    pipeline = Pipeline(
        map_reader=map_reader,
        drone_streamer=streamer,
        detector=superpoint_algorithm,
        matcher=superglue_matcher,
        query_processor=query_processor,
        config=PipelineConfig(),
        logger=logger,
    )

    if output_path is None:
        output_path = "output"
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output path: {output_path}")

    all_preds = []
    streamer_iter = iter(streamer)

    try:
        for i in range(len(streamer)):
            torch.cuda.empty_cache()
            print(f"Processing query image {i+1}/{len(streamer)}")

            try:
                drone_image = next(streamer_iter)
                query = query_processor(drone_image)
                if hasattr(query, 'image') and query.image is not None:
                    print(f"Query image size: {query.image.shape}")

                with torch.cuda.amp.autocast():
                    try:
                        pred = pipeline.run_on_image(query, output_path)
                        all_preds.append(pred)
                        print(f"Processed image {i+1} - Match: {pred['is_match']}")

                    except Exception as e:
                        print(f"Error processing image {i+1}: {e}")

            except Exception as e:
                print(f"Error with drone image {i+1}: {e}")
                continue

            gc.collect()
    except Exception as e:
        print(f"Error during processing: {e}")

    try:
        if all_preds:
            metrics = pipeline.compute_metrics(all_preds)
            pprint(metrics)
        else:
            print("No predictions to compute metrics for.")
    except Exception as e:
        print(f"Could not compute metrics: {e}")
        print("This is expected when using dummy CSV data.")

    return all_preds

if __name__ == "__main__":    
    main(r"D:\\deep_learning\\QATM\\superglue\\visual_localization\\src\\ref\\",
        r"D:\\deep_learning\\QATM\\superglue\\visual_localization\\src\\refs\\", 
        r"D:\deep_learning\QATM\superglue\visual_localization\src\query", 
        r"D:\\deep_learning\QATM\\superglue\\visual_localization\\src\\output")
