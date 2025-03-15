import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import os
import sys
from pathlib import Path
import run_with_dummy_csv
from PIL import Image, ImageTk
import cv2

class VisualLocalizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visual Localization App")
        self.root.geometry("700x500")
        self.root.resizable(True, True)
        self.style = ttk.Style()
        self.style.configure("TButton", padding=6, relief="flat", background="#ccc")
        self.style.configure("TLabel", padding=6, font=('Helvetica', 10))
        self.style.configure("Header.TLabel", font=('Helvetica', 12, 'bold'))
        
        self.create_widgets()
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(main_frame, text="Visual Localization Tool", style="Header.TLabel")
        title_label.pack(pady=10)
        
        folder_frame = ttk.LabelFrame(main_frame, text="Folder Selection", padding="10")
        folder_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ref_frame = ttk.Frame(folder_frame)
        ref_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(ref_frame, text="Reference Folder:").pack(side=tk.LEFT)
        self.ref_folder_var = tk.StringVar()
        ttk.Entry(ref_frame, textvariable=self.ref_folder_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(ref_frame, text="Browse", command=self.browse_ref_folder).pack(side=tk.LEFT)
        
        target_frame = ttk.Frame(folder_frame)
        target_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(target_frame, text="Target Folder:").pack(side=tk.LEFT)
        self.target_folder_var = tk.StringVar()
        ttk.Entry(target_frame, textvariable=self.target_folder_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(target_frame, text="Browse", command=self.browse_target_folder).pack(side=tk.LEFT)
                
        query_frame = ttk.Frame(folder_frame)
        query_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(query_frame, text="Query Folder:").pack(side=tk.LEFT)
        self.query_folder_var = tk.StringVar()
        ttk.Entry(query_frame, textvariable=self.query_folder_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(query_frame, text="Browse", command=self.browse_query_folder).pack(side=tk.LEFT)
        
        output_frame = ttk.Frame(folder_frame)
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_frame, text="Output Folder:").pack(side=tk.LEFT)
        self.output_folder_var = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_folder_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(output_frame, text="Browse", command=self.browse_output_folder).pack(side=tk.LEFT)
        
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        grid_frame = ttk.Frame(options_frame)
        grid_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(grid_frame, text="Grid Size:").pack(side=tk.LEFT)
        self.grid_size_var = tk.StringVar(value="3x3")
        grid_combo = ttk.Combobox(grid_frame, textvariable=self.grid_size_var, 
                                  values=["2x2", "3x3", "4x4", "5x5"])
        grid_combo.pack(side=tk.LEFT, padx=5)
        
        overlap_frame = ttk.Frame(options_frame)
        overlap_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(overlap_frame, text="Overlap (pixels):").pack(side=tk.LEFT)
        self.overlap_var = tk.IntVar(value=20)
        overlap_spin = ttk.Spinbox(overlap_frame, from_=0, to=100, textvariable=self.overlap_var, width=5)
        overlap_spin.pack(side=tk.LEFT, padx=5)
        
        device_frame = ttk.Frame(options_frame)
        device_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(device_frame, text="Device:").pack(side=tk.LEFT)
        self.device_var = tk.StringVar(value="cuda" if hasattr(run_with_dummy_csv, 'torch') and 
                                      hasattr(run_with_dummy_csv.torch, 'cuda') and 
                                      run_with_dummy_csv.torch.cuda.is_available() else "cpu")
        device_combo = ttk.Combobox(device_frame, textvariable=self.device_var, 
                                   values=["cuda", "cpu"])
        device_combo.pack(side=tk.LEFT, padx=5)
        
        weights_frame = ttk.Frame(options_frame)
        weights_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(weights_frame, text="Model Weights:").pack(side=tk.LEFT)
        self.weights_var = tk.StringVar(value="outdoor")
        weights_combo = ttk.Combobox(weights_frame, textvariable=self.weights_var, 
                                    values=["outdoor", "indoor"])
        weights_combo.pack(side=tk.LEFT, padx=5)
        
        run_frame = ttk.Frame(main_frame)
        run_frame.pack(pady=10)
        
        self.run_button = ttk.Button(run_frame, text="Run Localization", command=self.run_localization)
        self.run_button.pack(pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.pack(pady=5)
        
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(log_frame, height=10, width=80, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        sys.stdout = TextRedirector(self.log_text)
        
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
            
    def run_localization(self):
        ref_folder = self.ref_folder_var.get()
        target_folder = self.target_folder_var.get()
        query_folder = self.query_folder_var.get()
        output_folder = self.output_folder_var.get()
        
        if not ref_folder or not target_folder or not query_folder:
            messagebox.showerror("Error", "Please select all required folders")
            return
        
        grid_size_str = self.grid_size_var.get()
        grid_size = tuple(map(int, grid_size_str.split('x')))
        
        self.run_button.config(state=tk.DISABLED)
        self.status_var.set("Running...")
        self.progress_var.set(0)
        
        self.log_text.delete(1.0, tk.END)
        
        thread = threading.Thread(target=self.run_localization_thread, 
                                 args=(ref_folder, target_folder, query_folder, output_folder, grid_size))
        thread.daemon = True
        thread.start()
        
    def run_localization_thread(self, ref_folder, target_folder, query_folder, output_folder, grid_size):
        try:
            original_split_and_save = run_with_dummy_csv.split_and_save_image
            
            def patched_split_and_save(image_path, output_folder, grid_size=grid_size, overlap=self.overlap_var.get()):
                return original_split_and_save(image_path, output_folder, grid_size, overlap)
            
            run_with_dummy_csv.split_and_save_image = patched_split_and_save
            
            device = self.device_var.get()
            weights = self.weights_var.get()
            
            if hasattr(run_with_dummy_csv, 'SuperPointConfig'):
                original_superpoint_config = run_with_dummy_csv.SuperPointConfig
                
                def patched_superpoint_config(*args, **kwargs):
                    kwargs['device'] = device
                    return original_superpoint_config(*args, **kwargs)
                
                run_with_dummy_csv.SuperPointConfig = patched_superpoint_config
            
            if hasattr(run_with_dummy_csv, 'SuperGlueConfig'):
                original_superglue_config = run_with_dummy_csv.SuperGlueConfig
                
                def patched_superglue_config(*args, **kwargs):
                    kwargs['device'] = device
                    kwargs['weights'] = weights
                    return original_superglue_config(*args, **kwargs)
                
                run_with_dummy_csv.SuperGlueConfig = patched_superglue_config
            
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)
                print(f"Created output directory: {output_folder}")
            
            results = run_with_dummy_csv.main(ref_folder, target_folder, query_folder, output_folder)
            self.results = results
            self.output_folder = output_folder
            self.root.after(0, self.on_localization_complete, results)
            
        except Exception as e:
            self.root.after(0, self.on_localization_error, str(e))
            
    def on_localization_complete(self, results):
        self.run_button.config(state=tk.NORMAL)
        self.status_var.set(f"Complete - {len(results)} results")
        self.progress_var.set(100)
        messagebox.showinfo("Success", f"Localization completed with {len(results)} results")
        
        if results and len(results) > 0:
            self.show_image_viewer(results)
    
    def on_localization_error(self, error_msg):
        self.run_button.config(state=tk.NORMAL)
        self.status_var.set("Error")
        self.progress_var.set(0)
        messagebox.showerror("Error", f"An error occurred: {error_msg}")
        
    def show_image_viewer(self, results):
        self.viewer_window = tk.Toplevel(self.root)
        self.viewer_window.title("Match Results Viewer")
        self.viewer_window.geometry("800x600")
        self.viewer_window.minsize(600, 400)
        
        self.results = results
        self.current_result_index = 0
        
        main_frame = ttk.Frame(self.viewer_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.image_canvas = tk.Canvas(main_frame, bg="black")
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        
        self.prev_button = ttk.Button(nav_frame, text="Previous", command=self.show_previous_result)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.counter_var = tk.StringVar(value=f"Result 1 of {len(results)}")
        counter_label = ttk.Label(nav_frame, textvariable=self.counter_var)
        counter_label.pack(side=tk.LEFT, expand=True)
        
        self.next_button = ttk.Button(nav_frame, text="Next", command=self.show_next_result)
        self.next_button.pack(side=tk.RIGHT, padx=5)
        
        details_frame = ttk.LabelFrame(main_frame, text="Match Details")
        details_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.details_text = tk.Text(details_frame, height=6, wrap=tk.WORD)
        self.details_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.show_result(0)
        
    def show_result(self, index):
        if not hasattr(self, 'results') or not self.results:
            return
            
        if index < 0 or index >= len(self.results):
            return
            
        self.current_result_index = index
        result = self.results[index]
        
        self.counter_var.set(f"Result {index + 1} of {len(self.results)}")
        
        self.prev_button.config(state=tk.NORMAL if index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if index < len(self.results) - 1 else tk.DISABLED)
        
        self.details_text.delete(1.0, tk.END)
        
        details = f"Query: {result.get('query_name', 'Unknown')}\n"
        if 'matched_image' in result:
            details += f"Matched with: {result['matched_image']}\n"
        if 'is_match' in result:
            details += f"Match: {'Yes' if result['is_match'] else 'No'}\n"
        if 'inliers' in result:
            details += f"Inliers: {result['inliers']}\n"
        if 'confidence' in result:
            details += f"Confidence: {result['confidence']:.4f}\n"
        
        self.details_text.insert(tk.END, details)
        
        self.display_result_image(result)
        
    def display_result_image(self, result):
        self.image_canvas.delete("all")
        
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        if canvas_width <= 1:
            canvas_width = 700
        if canvas_height <= 1:
            canvas_height = 400
        
        try:
            output_path_str = getattr(self, 'output_folder', None) or self.output_folder_var.get()
            
            query_path = result.get('query_name', '')
            query_name = os.path.basename(query_path)
            query_name = os.path.splitext(query_name)[0] 
            
            print(f"Looking for visualization for query: {query_name}")
            print(f"In output directory: {output_path_str}")
            
            if not os.path.exists(output_path_str):
                print(f"Output directory does not exist: {output_path_str}")
                os.makedirs(output_path_str, exist_ok=True)
                print(f"Created output directory: {output_path_str}")
            
            all_files = []
            try:
                all_files = [f for f in os.listdir(output_path_str) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                print(f"Found {len(all_files)} image files in output directory")
                for f in all_files:
                    print(f"  - {f}")
            except Exception as e:
                print(f"Error listing directory: {e}")
            
            vis_path = None
            
            if 'matched_image' in result and result['matched_image']:
                match_name = None
                if hasattr(result['matched_image'], 'path'):
                    match_name = os.path.basename(result['matched_image'].path)
                elif hasattr(result['matched_image'], 'name'):
                    match_name = result['matched_image'].name
                elif isinstance(result['matched_image'], str):
                    match_name = os.path.basename(result['matched_image'])
                
                if match_name:
                    match_name = os.path.splitext(match_name)[0]  
                    
                    for file in all_files:
                        if (query_name in file and match_name in file) or (match_name in file and query_name in file):
                            vis_path = os.path.join(output_path_str, file)
                            print(f"Found match by query and ref names: {vis_path}")
                            break
            
            if not vis_path:
                for file in all_files:
                    if query_name in file:
                        vis_path = os.path.join(output_path_str, file)
                        print(f"Found match by query name: {vis_path}")
                        break
            
            if not vis_path and all_files:
                vis_path = os.path.join(output_path_str, all_files[0])
                print(f"Using first available image: {vis_path}")
            
            if vis_path and os.path.exists(vis_path):
                try:
                    img = Image.open(vis_path)
                    
                    img_width, img_height = img.size
                    scale = min(canvas_width / img_width, canvas_height / img_height)
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    
                    self.photo = ImageTk.PhotoImage(image=img)
                    
                    self.image_canvas.create_image(canvas_width//2, canvas_height//2, 
                                                  image=self.photo, anchor=tk.CENTER)
                    print(f"Successfully displayed image: {vis_path}")
                except Exception as e:
                    print(f"Error loading image {vis_path}: {e}")
                    self.image_canvas.create_text(canvas_width//2, canvas_height//2, 
                                                text=f"Error loading image: {e}", 
                                                fill="white", anchor=tk.CENTER)
            else:
                if query_path and os.path.exists(query_path):
                    try:
                        img = Image.open(query_path)
                        
                        img_width, img_height = img.size
                        scale = min(canvas_width / img_width, canvas_height / img_height)
                        new_width = int(img_width * scale)
                        new_height = int(img_height * scale)
                        img = img.resize((new_width, new_height), Image.LANCZOS)
                        
                        self.photo = ImageTk.PhotoImage(image=img)
                        
                        self.image_canvas.create_image(canvas_width//2, canvas_height//2, 
                                                      image=self.photo, anchor=tk.CENTER)
                        
                        self.image_canvas.create_text(10, 10, text="Query Image (no match visualization)", 
                                                     fill="white", anchor=tk.NW)
                        print(f"Displayed query image instead: {query_path}")
                    except Exception as e:
                        print(f"Error loading query image: {e}")
                        self.image_canvas.create_text(canvas_width//2, canvas_height//2, 
                                                    text=f"Error loading query image: {e}", 
                                                    fill="white", anchor=tk.CENTER)
                else:
                    self.image_canvas.create_text(canvas_width//2, canvas_height//2, 
                                                 text=f"No visualization image available\nOutput path: {output_path_str}\nQuery: {query_name}", 
                                                 fill="white", anchor=tk.CENTER)
                    
                    print(f"Could not find any suitable image to display")
                    print(f"Query path: {query_path} (exists: {os.path.exists(query_path) if query_path else False})")
        except Exception as e:
            print(f"Error in display_result_image: {e}")
            import traceback
            traceback.print_exc()
            self.image_canvas.create_text(canvas_width//2, canvas_height//2, 
                                         text=f"Error displaying image: {e}", 
                                         fill="white", anchor=tk.CENTER)
    
    def show_next_result(self):
        if hasattr(self, 'results') and hasattr(self, 'current_result_index'):
            next_index = self.current_result_index + 1
            if next_index < len(self.results):
                self.show_result(next_index)
                print(f"Showing next result (index: {next_index})")
    
    def show_previous_result(self):
        if hasattr(self, 'results') and hasattr(self, 'current_result_index'):
            prev_index = self.current_result_index - 1
            if prev_index >= 0:
                self.show_result(prev_index)
                print(f"Showing previous result (index: {prev_index})")


class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ""
        
    def write(self, string):
        self.buffer += string
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        
    def flush(self):
        pass


if __name__ == "__main__":
    root = tk.Tk()
    app = VisualLocalizationApp(root)
    root.mainloop()
