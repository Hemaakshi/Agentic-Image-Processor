import os
import pydicom
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from skimage import exposure
import glob

# ==========================================
# 1. CONFIGURATION (Change these for your PC)
# ==========================================
# Point this to where you have your raw folders (e.g., pneumonia, lung_ct)
INPUT_DIR = r"D:\MedicalProject\data\raw" 
# Point this to where you want the tiny .npy files to live
OUTPUT_DIR = r"D:\MedicalProject\data\processed" 
# Standard resolution for AI training
TARGET_SIZE = (512, 512) 

def process_file(file_path):
    """Handles DICOM, JPEG, PNG and standardizes them."""
    ext = os.path.splitext(file_path)[1].lower()
    
    # Identify category from the folder name (Virtual Tagging)
    category = os.path.basename(os.path.dirname(file_path))
    save_path = os.path.join(OUTPUT_DIR, category)
    os.makedirs(save_path, exist_ok=True)

    try:
        # --- PHASE 1: LOADING ---
        if ext in ['.dcm', '.dicom']:
            ds = pydicom.dcmread(file_path)
            pixels = ds.pixel_array.astype(float)
        elif ext in ['.jpg', '.jpeg', '.png']:
            img = Image.open(file_path).convert('L') # Force grayscale
            pixels = np.array(img).astype(float)
        else:
            return False

        # --- PHASE 2: STANDARDIZATION ---
        # 1. Resize
        pixels = cv2.resize(pixels, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        
        # 2. Normalize 0 to 1
        p_min, p_max = np.min(pixels), np.max(pixels)
        if p_max > p_min:
            pixels = (pixels - p_min) / (p_max - p_min)
        else:
            pixels = np.zeros_like(pixels)

        # 3. Enhance Contrast (CLAHE)
        # This makes nodules and tumors stand out significantly
        pixels = exposure.equalize_adapthist(pixels)

        # --- PHASE 3: STORAGE ---
        file_id = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join(save_path, f"{file_id}.npy")
        np.save(output_file, pixels.astype(np.float32))
        
        # OPTIONAL: Delete raw file to save space
        # os.remove(file_path) 
        
        return True
    except Exception as e:
        print(f"Error on {file_path}: {e}")
        return False

def run_universal_pipeline():
    # Search for all common medical image extensions
    extensions = ['*.dcm', '*.dicom', '*.jpg', '*.jpeg', '*.png']
    files_to_process = []
    for ext in extensions:
        files_to_process.extend(glob.glob(os.path.join(INPUT_DIR, "**", ext), recursive=True))

    print(f"Factory: Found {len(files_to_process)} images across all categories.")
    
    success_count = 0
    for f in tqdm(files_to_process, desc="Processing Images"):
        if process_file(f):
            success_count += 1

    print(f"\nProcessing Complete!")
    print(f"Total Processed: {success_count}")
    print(f"Files are waiting for you in: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_universal_pipeline()