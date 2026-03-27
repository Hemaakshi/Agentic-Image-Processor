import pydicom
import numpy as np
import os
from skimage import exposure
import glob

def process_file(raw_file_path):
    """Processes a single DICOM file: Anonymize -> Normalize -> Save."""
    try:
        # 1. Load Raw
        ds = pydicom.dcmread(raw_file_path)
        pixels = ds.pixel_array.astype(float)

        # 2. Anonymize (Privacy)
        ds.PatientName = "ANONYMOUS"
        ds.PatientID = "000-000"

        # 3. Normalize (Standardize for AI)
        # Rescale and equalize histogram for better feature visibility
        processed_pixels = exposure.equalize_adapthist(pixels / np.max(pixels))

        # 4. Save to data/processed/
        # We keep the sub-folder structure (e.g., pneumonia_triage/file.npy)
        base_name = os.path.basename(raw_file_path).replace('.dcm', '.npy')
        
        # Determine which category folder this belongs to
        category = os.path.basename(os.path.dirname(raw_file_path))
        output_dir = os.path.join('data', 'processed', category)
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, base_name)
        np.save(output_path, processed_pixels)
        return True
    except Exception as e:
        print(f"Error processing {raw_file_path}: {e}")
        return False

def run_bulk_processing():
    """Finds all .dcm files in data/raw/ and processes them."""
    # This looks into every subfolder of data/raw/ for .dcm files
    raw_files = glob.glob(os.path.join('data', 'raw', '**', '*.dcm'), recursive=True)
    
    print(f"Agent: Found {len(raw_files)} files to process.")
    
    success_count = 0
    for f in raw_files:
        if process_file(f):
            success_count += 1
            
    print(f" Bulk Processing Complete! {success_count}/{len(raw_files)} files processed.")

if __name__ == "__main__":
    run_bulk_processing()