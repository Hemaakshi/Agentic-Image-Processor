import pydicom
import numpy as np
import os
from skimage import exposure

def process_raw_to_processed(raw_file_path):
    """
    1. LOAD RAW (.dcm)
    2. ANONYMIZE (Privacy)
    3. NORMALIZE (Brightness)
    4. SAVE TO PROCESSED (.npy)
    """
    # 1. Load Raw
    ds = pydicom.dcmread(raw_file_path)
    pixels = ds.pixel_array.astype(float)

    # 2. Anonymize
    ds.PatientName = "ANONYMOUS"
    ds.PatientID = "000-000"

    # 3. Normalize
    # Standardizes the image so AI can read it clearly
    processed_pixels = exposure.equalize_adapthist(pixels / np.max(pixels))

    # 4. Save to data/processed/
    file_name = os.path.basename(raw_file_path).replace('.dcm', '.npy')
    output_path = os.path.join('data', 'processed', file_name)
    
    np.save(output_path, processed_pixels)
    print(f"File processed and saved to: {output_path}")

if __name__ == "__main__":
    print("Data Manager Script Ready.")