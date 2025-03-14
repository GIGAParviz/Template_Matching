import logging
import os
import gc
import shutil
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import pandas as pd
import random
import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import your project modules
from svl.keypoint_pipeline.detection_and_description import SuperPointAlgorithm
from svl.keypoint_pipeline.matcher import SuperGlueMatcher
from svl.keypoint_pipeline.typing import SuperGlueConfig, SuperPointConfig
from svl.localization.drone_streamer import DroneImageStreamer
from svl.localization.map_reader import SatelliteMapReader
from svl.localization.pipeline import Pipeline, PipelineConfig
from svl.localization.preprocessing import QueryProcessor
from svl.tms.data_structures import CameraModel
from svl.tms.schemas import GpsCoordinate

# Set up logging to capture output in the GUI
class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        logging.Handler.__init__(self)
        self.text_widget = text_widget
        
    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.see(tk.END)
            self.text_widget.configure(state='disabled')
        self.text_widget.after(0, append)

class VisualLocalizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SuperGlue Visual Localization")
        self.root.geometry("900x700")
        
        # Create main tabs
        self.tab_control = ttk.Notebook(root)
        
        self.settings_tab = ttk.Frame(self.tab_control)
        self.processing_tab = ttk.Frame(self.tab_control)
        self.results_tab = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.settings_tab, text="Settings")
        self.tab_control.add(self.processing_tab, text="Processing")
        self.tab_control.add(self.results_tab, text="Results")
        
        self.tab_control.pack(expand=1, fill="both")
        
        # Initialize variables
        self.ref_folder_var = tk.StringVar()
        self.target_folder_var = tk.StringVar()
        self.query_folder_var = tk.StringVar()
        self.output_folder_var = tk.StringVar()
        self.keypoints_var = tk.IntVar(value=256)
        self.grid_size_var = tk.IntVar(value=3)
        self.ref_image_size_var = tk.IntVar(value=384)
        self.query_image_size_var = tk.IntVar(value=192)
        self.sinkhorn_iterations_var = tk.IntVar(value=10)
        self.match_threshold_var = tk.DoubleVar(value=0.5)
        self.use_gpu_var = tk.BooleanVar(value=True)
        self.use_fp16_var = tk.BooleanVar(value=True)
        
        # Initialize results storage
        self.results = None
        self.map_reader = None
        self.streamer = None
        
        # Setup the tabs
        self.setup_settings_tab()
        self.setup_processing_tab()
        self.setup_results_tab()
        
        # Check GPU availability
        if torch.cuda.is_available():
            self.log_box.configure(state='normal')
            self.log_box.insert(tk.END, f"GPU detected: {torch.cuda.get_device_name(0)}\n")
            self.log_box.configure(state='disabled')
        else:
            self.use_gpu_var.set(False)
            self.log_box.configure(state='normal')
            self.log_box.insert(tk.END, "No GPU detected. Running on CPU.\n")
            self.log_box.configure(state='disabled')
    
    def setup_settings_tab(self):
        # Create settings frame with padding
        settings_frame = ttk.LabelFrame(self.settings_tab, text="Folder Settings")
        settings_frame.pack(padx=10, pady=10, fill="x")
        
        # Reference folder
        ttk.Label(settings_frame, text="Reference Folder:").grid(column=0, row=0, sticky="w", padx=5, pady=5)
        ttk.Entry(settings_frame, textvariable=self.ref_folder_var, width=50).grid(column=1, row=0, padx=5, pady=5)
        ttk.Button(settings_frame, text="Browse...", command=self.browse_ref_folder).grid(column=2, row=0, padx=5, pady=5)
        
        # Target folder for split images
        ttk.Label(settings_frame, text="Target Folder:").grid(column=0, row=1, sticky="w", padx=5, pady=5)
        ttk.Entry(settings_frame, textvariable=self.target_folder_var, width=50).grid(column=1, row=1, padx=5, pady=5)
        ttk.Button(settings_frame, text="Browse...", command=self.browse_target_folder).grid(column=2, row=1, padx=5, pady=5)
        
        # Query folder
        ttk.Label(settings_frame, text="Query Folder:").grid(column=0, row=2, sticky="w", padx=5, pady=5)
        ttk.Entry(settings_frame, textvariable=self.query_folder_var, width=50).grid(column=1, row=2, padx=5, pady=5)
        ttk.Button(settings_frame, text="Browse...", command=self.browse_query_folder).grid(column=2, row=2, padx=5, pady=5)
        
        # Output folder
        ttk.Label(settings_frame, text="Output Folder:").grid(column=0, row=3, sticky="w", padx=5, pady=5)
        ttk.Entry(settings_frame, textvariable=self.output_folder_var, width=50).grid(column=1, row=3, padx=5, pady=5)
        ttk.Button(settings_frame, text="Browse...", command=self.browse_output_folder).grid(column=2, row=3, padx=5, pady=5)
        
        # Create algorithm settings frame
        algo_frame = ttk.LabelFrame(self.settings_tab, text="Algorithm Settings")
        algo_frame.pack(padx=10, pady=10, fill="x")
        
        # SuperPoint keypoints
        ttk.Label(algo_frame, text="Max Keypoints:").grid(column=0, row=0, sticky="w", padx=5, pady=5)
        keypoints_scale = ttk.Scale(algo_frame, from_=64, to=512, orient=tk.HORIZONTAL, 
                                    variable=self.keypoints_var, length=200)
        keypoints_scale.grid(column=1, row=0, padx=5, pady=5)
        ttk.Label(algo_frame, textvariable=self.keypoints_var).grid(column=2, row=0, padx=5, pady=5)
        
        # Grid size for splitting
        ttk.Label(algo_frame, text="Grid Size:").grid(column=0, row=1, sticky="w", padx=5, pady=5)
        grid_scale = ttk.Scale(algo_frame, from_=2, to=5, orient=tk.HORIZONTAL, 
                               variable=self.grid_size_var, length=200)
        grid_scale.grid(column=1, row=1, padx=5, pady=5)
        ttk.Label(algo_frame, textvariable=self.grid_size_var).grid(column=2, row=1, padx=5, pady=5)
        
        # Reference image size
        ttk.Label(algo_frame, text="Ref Image Size:").grid(column=0, row=2, sticky="w", padx=5, pady=5)
        ref_size_scale = ttk.Scale(algo_frame, from_=128, to=512, orient=tk.HORIZONTAL, 
                                   variable=self.ref_image_size_var, length=200)
        ref_size_scale.grid(column=1, row=2, padx=5, pady=5)
        ttk.Label(algo_frame, textvariable=self.ref_image_size_var).grid(column=2, row=2, padx=5, pady=5)
        
        # Query image size
        ttk.Label(algo_frame, text="Query Image Size:").grid(column=0, row=3, sticky="w", padx=5, pady=5)
        query_size_scale = ttk.Scale(algo_frame, from_=64, to=256, orient=tk.HORIZONTAL, 
                                     variable=self.query_image_size_var, length=200)
        query_size_scale.grid(column=1, row=3, padx=5, pady=5)
        ttk.Label(algo_frame, textvariable=self.query_image_size_var).grid(column=2, row=3, padx=5, pady=5)
        
        # Sinkhorn iterations
        ttk.Label(algo_frame, text="Sinkhorn Iterations:").grid(column=0, row=4, sticky="w", padx=5, pady=5)
        sinkhorn_scale = ttk.Scale(algo_frame, from_=1, to=20, orient=tk.HORIZONTAL, 
                                   variable=self.sinkhorn_iterations_var, length=200)
        sinkhorn_scale.grid(column=1, row=4, padx=5, pady=5)
        ttk.Label(algo_frame, textvariable=self.sinkhorn_iterations_var).grid(column=2, row=4, padx=5, pady=5)
        
        # Match threshold
        ttk.Label(algo_frame, text="Match Threshold:").grid(column=0, row=5, sticky="w", padx=5, pady=5)
        match_scale = ttk.Scale(algo_frame, from_=0.1, to=0.9, orient=tk.HORIZONTAL, 
                                variable=self.match_threshold_var, length=200)
        match_scale.grid(column=1, row=5, padx=5, pady=5)
        # Display with 2 decimal places
        threshold_label = ttk.Label(algo_frame, text="0.50")
        threshold_label.grid(column=2, row=5, padx=5, pady=5)
        
        # Update threshold label when value changes
        def update_threshold_label(*args):
            threshold_label.config(text=f"{self.match_threshold_var.get():.2f}")
        
        self.match_threshold_var.trace_add("write", update_threshold_label)
        
        # Hardware settings
        hw_frame = ttk.LabelFrame(self.settings_tab, text="Hardware Settings")
        hw_frame.pack(padx=10, pady=10, fill="x")
        
        # Use GPU
        ttk.Checkbutton(hw_frame, text="Use GPU", variable=self.use_gpu_var).grid(column=0, row=0, sticky="w", padx=5, pady=5)
        
        # Use FP16
        ttk.Checkbutton(hw_frame, text="Use FP16 (Mixed Precision)", variable=self.use_fp16_var).grid(column=1, row=0, sticky="w", padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self.settings_tab)
        button_frame.pack(padx=10, pady=10, fill="x")
        
        ttk.Button(button_frame, text="Load Default Settings", command=self.load_default_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Start Processing", command=self.start_processing).pack(side=tk.RIGHT, padx=5)
    
    def setup_processing_tab(self):
        # Create log output frame
        log_frame = ttk.LabelFrame(self.processing_tab, text="Processing Log")
        log_frame.pack(padx=10, pady=10, expand=True, fill="both")
        
        # Add scrolled text for logging
        self.log_box = scrolledtext.ScrolledText(log_frame, state='disabled', height=20)
        self.log_box.pack(padx=5, pady=5, expand=True, fill="both")
        
        # Configure logging to use the text widget
        text_handler = TextHandler(self.log_box)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        text_handler.setFormatter(formatter)
        
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(text_handler)
        
        # Progress bar
        progress_frame = ttk.Frame(self.processing_tab)
        progress_frame.pack(padx=10, pady=10, fill="x")
        
        ttk.Label(progress_frame, text="Progress:").pack(side=tk.LEFT, padx=5)
        self.progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(side=tk.LEFT, padx=5, fill="x", expand=True)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)
        
        # Control buttons
        control_frame = ttk.Frame(self.processing_tab)
        control_frame.pack(padx=10, pady=10, fill="x")
        
        self.start_button = ttk.Button(control_frame, text="Start Processing", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Processing", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Clear Log", command=self.clear_log).pack(side=tk.RIGHT, padx=5)
    
    def setup_results_tab(self):
        # Results will contain:
        # 1. A list of matches
        # 2. Visualization of the best match
        # 3. Metrics summary
        
        # Create frames
        results_pane = ttk.PanedWindow(self.results_tab, orient=tk.HORIZONTAL)
        results_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_frame = ttk.Frame(results_pane)
        right_frame = ttk.Frame(results_pane)
        
        results_pane.add(left_frame, weight=1)
        results_pane.add(right_frame, weight=2)
        
        # Left side: Results list and metrics
        self.results_tree = ttk.Treeview(left_frame, columns=('Query', 'Reference', 'Match', 'Distance'))
        self.results_tree.heading('#0', text='#')
        self.results_tree.heading('Query', text='Query Image')
        self.results_tree.heading('Reference', text='Matched Reference')
        self.results_tree.heading('Match', text='Is Match')
        self.results_tree.heading('Distance', text='Distance')
        
        self.results_tree.column('#0', width=40)
        self.results_tree.column('Query', width=120)
        self.results_tree.column('Reference', width=120)
        self.results_tree.column('Match', width=70)
        self.results_tree.column('Distance', width=70)
        
        self.results_tree.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.results_tree.bind('<<TreeviewSelect>>', self.on_result_select)
        
        # Metrics frame
        metrics_frame = ttk.LabelFrame(left_frame, text="Metrics Summary")
        metrics_frame.pack(padx=5, pady=5, fill=tk.X)
        
        self.metrics_text = scrolledtext.ScrolledText(metrics_frame, height=10, state='disabled')
        self.metrics_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        # Right side: Visualization
        viz_frame = ttk.LabelFrame(right_frame, text="Match Visualization")
        viz_frame.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        # Canvas for matplotlib figure
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Export button
        ttk.Button(right_frame, text="Export Results", command=self.export_results).pack(padx=5, pady=5, side=tk.RIGHT)
    
    # Folder browsing functions
    def browse_ref_folder(self):
        folder = filedialog.askdirectory(title="Select Reference Folder")
        if folder:
            self.ref_folder_var.set(folder)
    
    def browse_target_folder(self):
        folder = filedialog.askdirectory(title="Select Target Folder")
        if folder:
            self.target_folder_var.set(folder)
    
    def browse_query_folder(self):
        folder = filedialog.askdirectory(title="Select Query Folder")
        if folder:
            self.query_folder_var.set(folder)
    
    def browse_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder_var.set(folder)
    
    def load_default_settings(self):
        self.keypoints_var.set(256)
        self.grid_size_var.set(3)
        self.ref_image_size_var.set(384)
        self.query_image_size_var.set(192)
        self.sinkhorn_iterations_var.set(10)
        self.match_threshold_var.set(0.5)
        self.use_gpu_var.set(torch.cuda.is_available())
        self.use_fp16_var.set(True)
    
    def start_processing(self):
        if not self.ref_folder_var.get() or not Path(self.ref_folder_var.get()).exists():
            messagebox.showerror("Error", "Please select a valid reference folder")
            return
        
        if not self.target_folder_var.get():
            self.target_folder_var.set(os.path.join(self.ref_folder_var.get(), "tiles"))
        
        if not self.query_folder_var.get() or not Path(self.query_folder_var.get()).exists():
            messagebox.showerror("Error", "Please select a valid query folder")
            return
        
        if not self.output_folder_var.get():
            self.output_folder_var.set(os.path.join(os.path.dirname(self.ref_folder_var.get()), "output"))
        
        self.tab_control.select(1)
        self.clear_log()
        
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Processing...")
        self.progress_bar["value"] = 0
        
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(target=self.run_processing)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def run_processing(self):
        try:
            # Log settings
            logging.info(f"Starting processing with settings:")
            logging.info(f"Reference folder: {self.ref_folder_var.get()}")
            logging.info(f"Target folder: {self.target_folder_var.get()}")
            logging.info(f"Query folder: {self.query_folder_var.get()}")
            logging.info(f"Output folder: {self.output_folder_var.get()}")
            logging.info(f"Max keypoints: {self.keypoints_var.get()}")
            logging.info(f"Grid size: {self.grid_size_var.get()}")
            logging.info(f"Reference image size: {self.ref_image_size_var.get()}")
            logging.info(f"Query image size: {self.query_image_size_var.get()}")
            logging.info(f"Using GPU: {self.use_gpu_var.get()}")
            logging.info(f"Using FP16: {self.use_fp16_var.get()}")
            
            # Clear CUDA cache
            if self.use_gpu_var.get() and torch.cuda.is_available():
                torch.cuda.empty_cache()
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            ref_folder = Path(self.ref_folder_var.get())
            target_path = Path(self.target_folder_var.get())
            query_folder = Path(self.query_folder_var.get())
            output_path = Path(self.output_folder_var.get())
            
            self.progress_bar["value"] = 10
            self.root.update_idletasks()
            
            if checker(ref_folder):
                grid_size = (self.grid_size_var.get(), self.grid_size_var.get())
                split_and_save_image(next(ref_folder.glob("*.jpg")), target_path, grid_size=grid_size, overlap=50)
                ref_folder = Path(target_path)
            
            self.progress_bar["value"] = 20
            self.root.update_idletasks()
            
            ref_csv_files = list(ref_folder.glob("*.csv"))
            if len(ref_csv_files) == 0:
                create_dummy_ref_csv(ref_folder)
            
            query_csv_files = list(query_folder.glob("*.csv"))
            if len(query_csv_files) == 0:
                create_dummy_query_csv(query_folder)
            
            self.progress_bar["value"] = 30
            self.root.update_idletasks()
            
            device = "cuda" if (self.use_gpu_var.get() and torch.cuda.is_available()) else "cpu"
            
            superpoint_config = SuperPointConfig(
                device=device,
                nms_radius=4,
                keypoint_threshold=0.005,
                max_keypoints=self.keypoints_var.get(),
            )
            superpoint_algorithm = SuperPointAlgorithm(superpoint_config)

            superglue_config = SuperGlueConfig(
                device=device,
                weights="outdoor",
                sinkhorn_iterations=self.sinkhorn_iterations_var.get(),
                match_threshold=self.match_threshold_var.get(),
            )
            superglue_matcher = SuperGlueMatcher(superglue_config)
            
            # Initialize map reader
            self.progress_bar["value"] = 40
            self.root.update_idletasks()
            
            self.map_reader = SatelliteMapReader(
                db_path=ref_folder,
                resize_size=(self.ref_image_size_var.get(), self.ref_image_size_var.get()),
                logger=logging.getLogger("SatelliteMapReader"),
                metadata_method="CSV",
            )
            self.map_reader.initialize_db()
            self.map_reader.setup_db()
            
            # Assign metadata
            manually_assign_metadata(self.map_reader, ref_folder)
            self.map_reader.resize_db_images()
            
            # Extract features
            self.progress_bar["value"] = 50
            self.root.update_idletasks()
            
            if self.use_fp16_var.get() and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    self.map_reader.describe_db_images(superpoint_algorithm)
            else:
                self.map_reader.describe_db_images(superpoint_algorithm)
            
            # Initialize streamer
            self.progress_bar["value"] = 60
            self.root.update_idletasks()
            
            self.streamer = DroneImageStreamer(
                image_folder=query_folder,
                has_gt=True,
                logger=logging.getLogger("DroneImageStreamer"),
            )
            
            if len(self.streamer) == 0:
                raise ValueError("No query images found")
            
            # Configure query processor
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
                processings=["resize"],
                camera_model=camera_model,
                satellite_resolution=None,
                size=(self.query_image_size_var.get(),),
            )
            
            # Set up pipeline
            self.progress_bar["value"] = 70
            self.root.update_idletasks()
            
            # Patch compute_geo_pose to handle None values
            original_compute_geo_pose = Pipeline.compute_geo_pose
            
            def patched_compute_geo_pose(self, satellite_image, matching_center):
                try:
                    # Check if metadata is available
                    if satellite_image.top_left is None or satellite_image.bottom_right is None:
                        logging.warning(f"Missing metadata for image {satellite_image.name}, using dummy values")
                        # Return dummy coordinates
                        return GpsCoordinate(lat=37.0, long=-122.0)
                    return original_compute_geo_pose(self, satellite_image, matching_center)
                except Exception as e:
                    logging.error(f"ERROR in compute_geo_pose: {e}")
                    # Return dummy coordinates
                    return GpsCoordinate(lat=37.0, long=-122.0)
            
            # Apply the patch
            Pipeline.compute_geo_pose = patched_compute_geo_pose
            
            pipeline = Pipeline(
                map_reader=self.map_reader,
                drone_streamer=self.streamer,
                detector=superpoint_algorithm,
                matcher=superglue_matcher,
                query_processor=query_processor,
                config=PipelineConfig(),
                logger=logging.getLogger("Pipeline"),
            )
            
            # Create output directory
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Process query images one by one
            all_preds = []
            streamer_iter = iter(self.streamer)
            
            for i in range(len(self.streamer)):
                # Update progress
                progress = 70 + (i / len(self.streamer)) * 25
                self.progress_bar["value"] = progress
                self.root.update_idletasks()
                
                if hasattr(self, "stop_requested") and self.stop_requested:
                    logging.info("Processing stopped by user")
                    break
                
                try:
                    torch.cuda.empty_cache()
                    logging.info(f"Processing query image {i+1}/{len(self.streamer)}")
                    
                    drone_image = next(streamer_iter)
                    query = query_processor(drone_image)
                    
                    if self.use_fp16_var.get() and torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            pred = pipeline.run_on_image(query, output_path)
                    else:
                        pred = pipeline.run_on_image(query, output_path)
                    
                    all_preds.append(pred)
                    logging.info(f"Processed image {i+1} - Match: {pred['is_match']}")
                except Exception as e:
                    logging.error(f"Error processing image {i+1}: {e}")
                    all_preds.append({"is_match": False, "matched_image": None, "distance": float('inf')})
                
                gc.collect()
            
            self.progress_bar["value"] = 95
            self.root.update_idletasks()
            
            try:
                if all_preds:
                    metrics = pipeline.compute_metrics(all_preds)
                    logging.info(f"Metrics: {metrics}")
                    self.metrics = metrics
                else:
                    logging.warning("No predictions to compute metrics for.")
                    self.metrics = {}
            except Exception as e:
                logging.error(f"Could not compute metrics: {e}")
                self.metrics = {}
            
            self.results = all_preds
            self.progress_bar["value"] = 100
            self.status_var.set("Completed")
            
            self.populate_results()
            
            self.root.after(0, lambda: self.tab_control.select(2))
            
        except Exception as e:
            logging.error(f"Processing failed: {e}")
            self.status_var.set("Failed")
            messagebox.showerror("Processing Error", f"An error occurred: {str(e)}")
        finally:
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
            if hasattr(self, "stop_requested"):
                del self.stop_requested
    
    def stop_processing(self):
        self.stop_requested = True
        self.status_var.set("Stopping...")
        logging.info("User requested to stop processing")
    
    def clear_log(self):
        self.log_box.configure(state='normal')
        self.log_box.delete(1.0, tk.END)
        self.log_box.configure(state='disabled')
    
    def populate_results(self):
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        if not self.results:
            return
        
        for i, pred in enumerate(self.results):
            matched_name = pred.get('matched_image', None)
            if matched_name is not None and hasattr(matched_name, 'name'):
                matched_name = matched_name.name
                
            values = (
                f"Query {i+1}",
                matched_name or "None",
                "Yes" if pred.get('is_match', False) else "No",
                f"{pred.get('distance', 'N/A'):.2f}" if isinstance(pred.get('distance'), (int, float)) else "N/A"
            )
            
            self.results_tree.insert('', 'end', text=str(i+1), values=values)
        
        self.metrics_text.configure(state='normal')
        self.metrics_text.delete(1.0, tk.END)
        
        if hasattr(self, 'metrics') and self.metrics:
            for key, value in self.metrics.items():
                if isinstance(value, (int, float)):
                    self.metrics_text.insert(tk.END, f"{key}: {value:.4f}\n")
                else:
                    self.metrics_text.insert(tk.END, f"{key}: {value}\n")
        else:
            self.metrics_text.insert(tk.END, "No metrics available")
        
        self.metrics_text.configure(state='disabled')
    
    def on_result_select(self, event):
        selection = self.results_tree.selection()
        if not selection:
            return
        
        item = self.results_tree.item(selection[0])
        index = int(item['text']) - 1
        
        if not self.results or index >= len(self.results):
            return
        
        # Get the prediction
        pred = self.results[index]
        
        self.visualize_match(pred, index)
    
    def visualize_match(self, pred, index):
        self.fig.clear()
        
        if not pred.get('is_match', False) or not pred.get('matched_image'):
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No match found", ha='center', va='center', fontsize=14)
            ax.axis('off')
            self.canvas.draw()
            return
        
        try:
            matched_image = pred.get('matched_image')
            if hasattr(matched_image, 'name'):
                matched_name = matched_image.name
            else:
                matched_name = str(matched_image)
            
            ax1 = self.fig.add_subplot(111)
            
            query_folder = Path(self.output_folder_var.get())
            query_images = []
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                query_images.extend(list(query_folder.glob(f'*{ext}')))
            
            if index < len(query_images):
                query_img_path = query_images[index]
            else:
                query_img_path = None
            
            ref_folder = Path(self.ref_folder_var.get())
            output_folder = Path(self.output_folder_var.get())
            
            ref_img_path = None
            for folder in [ref_folder, output_folder]:
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    paths = list(folder.glob(f'*{ext}'))
                    for path in paths:
                        if path.stem == matched_name or path.name == matched_name:
                            ref_img_path = path
                            break
                    if ref_img_path:
                        break
            
            # Load and display images
            if query_img_path and query_img_path.exists():
                query_img = cv2.imread(str(query_img_path))
                if query_img is not None:
                    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
                    ax1.imshow(query_img)
                    ax1.set_title(f"Result Image\n{query_img_path.name}")
                else:
                    ax1.text(0.5, 0.5, "Failed to load image", ha='center', va='center')
            else:
                ax1.text(0.5, 0.5, "Result image not found", ha='center', va='center')
            
            match_text = f"Match Score: {pred.get('num_inliers', 'N/A')}\nDistance: {pred.get('distance', 'N/A'):.2f}m"
            self.fig.suptitle(match_text)
            
            ax1.axis('off')
            
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            logging.error(f"Error visualizing match: {e}")
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error visualizing match: {str(e)}", ha='center', va='center', fontsize=10, wrap=True)
            ax.axis('off')
            self.canvas.draw()
    
    def export_results(self):
        if not self.results:
            messagebox.showinfo("Export", "No results to export")
            return
        
        # Ask for export file
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Results"
        )
        
        if not file_path:
            return
        
        try:
            # Create DataFrame
            data = []
            for i, pred in enumerate(self.results):
                matched_name = pred.get('matched_image', None)
                if matched_name is not None and hasattr(matched_name, 'name'):
                    matched_name = matched_name.name
                
                data.append({
                    "Query Index": i+1,
                    "Matched Reference": matched_name,
                    "Is Match": pred.get('is_match', False),
                    "Distance (m)": pred.get('distance', 'N/A'),
                    "Number of Inliers": pred.get('num_inliers', 'N/A'),
                })
            
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            
            # Add metrics if available
            if hasattr(self, 'metrics') and self.metrics:
                with open(file_path.replace(".csv", "_metrics.csv"), 'w') as f:
                    f.write("Metric,Value\n")
                    for key, value in self.metrics.items():
                        f.write(f"{key},{value}\n")
            
            messagebox.showinfo("Export", f"Results exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")

# Import helper functions
def split_and_save_image(image_path, output_folder, grid_size=(2, 2), overlap=0):
    # Implementation from your original code
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
    return df  # Return the dataframe directly

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
    """Ensure all satellite images have proper metadata"""
    # Look for CSV files
    csv_files = list(Path(ref_folder).glob("*.csv"))
    
    # If no CSV file found, create one
    if not csv_files:
        print("No metadata CSV found. Creating dummy metadata...")
        metadata_df = create_dummy_ref_csv(ref_folder)
    else:
        # Try to load the existing CSV
        csv_path = csv_files[0]
        try:
            metadata_df = pd.read_csv(csv_path)
            print(f"Loaded metadata from {csv_path}")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            metadata_df = create_dummy_ref_csv(ref_folder)
    
    # Check if dataframe has required columns
    required_columns = ["Filename", "Top_left_lat", "Top_left_lon", "Bottom_right_lat", "Bottom_right_long"]
    if not all(col in metadata_df.columns for col in required_columns):
        print(f"CSV missing required columns. Creating new metadata...")
        metadata_df = create_dummy_ref_csv(ref_folder)

    # Now assign the metadata to each image
    for _, row in metadata_df.iterrows():
        filename = row["Filename"]
        if filename in map_reader._image_db:
            satellite_image = map_reader._image_db[filename]
            # Create GPS coordinate objects
            top_left = GpsCoordinate(
                lat=float(row["Top_left_lat"]), 
                long=float(row["Top_left_lon"])
            )
            bottom_right = GpsCoordinate(
                lat=float(row["Bottom_right_lat"]), 
                long=float(row["Bottom_right_long"])
            )
            
            # Directly assign to the image object
            satellite_image.top_left = top_left
            satellite_image.bottom_right = bottom_right
            
            print(f"Set metadata for {filename}: TL={top_left.lat},{top_left.long} BR={bottom_right.lat},{bottom_right.long}")

def checker(path):
    """Check if the folder contains exactly one jpg image"""
    if len(list(path.glob("*.jpg"))) == 1:
        return True
    return False

def ensure_query_images(query_folder, ref_folder):
    """Check if query folder has images, if not, copy from ref_folder"""
    query_folder = Path(query_folder)
    ref_folder = Path(ref_folder)
    
    # Check for images in query folder
    query_images = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        query_images.extend(list(query_folder.glob(f'*{ext}')))
    
    if not query_images:
        print(f"No images found in query folder {query_folder}. Copying from reference folder...")
        
        # Check for images in ref folder
        ref_images = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            ref_images.extend(list(ref_folder.glob(f'*{ext}')))
        
        if not ref_images:
            raise ValueError(f"No images found in reference folder {ref_folder} either!")
        
        # Create query folder if it doesn't exist
        query_folder.mkdir(parents=True, exist_ok=True)
        
        # Copy a few images from ref to query
        for img_path in ref_images[:2]:  # Copy at most 2 images
            dest_path = query_folder / img_path.name
            shutil.copy2(img_path, dest_path)
            print(f"Copied {img_path.name} to query folder")
        
        # Check again
        query_images = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            query_images.extend(list(query_folder.glob(f'*{ext}')))
            
        if not query_images:
            raise ValueError("Failed to copy reference images to query folder!")
    
    return len(query_images)

# Main entry point
if __name__ == "__main__":
    # Create the root window
    root = tk.Tk()
    
    # Set app icon and title
    root.title("SuperGlue Visual Localization")
    
    # Apply a theme (if available)
    try:
        from ttkthemes import ThemedStyle
        style = ThemedStyle(root)
        style.set_theme("arc")  # Use a modern theme
    except ImportError:
        # If ttkthemes is not available, use the default style
        pass
    
    # Create the app
    app = VisualLocalizationApp(root)
    
    # Set window size and position
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = 1024
    window_height = 768
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    
    root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")
    
    # Make the window resizable
    root.resizable(True, True)
    
    # Set minimum size
    root.minsize(800, 600)
    
    root.mainloop() 
    
    
    
    
    





"D:\deep_learning\QATM\superglue\visual_localization\src\output"
"D:\deep_learning\QATM\superglue\visual_localization\src\query"
"D:\deep_learning\QATM\superglue\visual_localization\src\ref"
"D:\deep_learning\QATM\superglue\visual_localization\src\refs"