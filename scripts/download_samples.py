import os
import requests
from tqdm import tqdm

# The 4 cases from your metadata_config.json
DATASET_URLS = {
    "pneumonia_triage": "https://raw.githubusercontent.com/ivmartel/dicom-samples/master/dicom-samples/DX/DICOM/P01/IM-0001-0001.dcm",
    "lung_nodule_ct": "https://raw.githubusercontent.com/ivmartel/dicom-samples/master/dicom-samples/CT/DICOM/P01/IM-0001-0001.dcm",
    "brain_tumor_mri": "https://raw.githubusercontent.com/ivmartel/dicom-samples/master/dicom-samples/MR/DICOM/P01/IM-0001-0001.dcm",
    "breast_cancer_screening": "https://raw.githubusercontent.com/ivmartel/dicom-samples/master/dicom-samples/MG/DICOM/P01/IM-0001-0001.dcm"
}

def grab_data_from_url(category, url):
    """
    Streams data from the URL to your local D: drive.
    Because of .gitignore, this data stays off GitHub.
    """
    # Create the specific folder for the case
    target_dir = os.path.join("data", "raw", category)
    os.makedirs(target_dir, exist_ok=True)
    
    file_name = f"{category}_sample.dcm"
    file_path = os.path.join(target_dir, file_name)
    
    print(f"Agent: Grabbing {category} data...")
    
    # stream=True means we don't load the whole 1GB into RAM at once
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(file_path, "wb") as f, tqdm(total=total_size, unit='B', unit_scale=True, desc=category) as pbar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

if __name__ == "__main__":
    for category, url in DATASET_URLS.items():
        grab_data_from_url(category, url)
    print("\n All samples 'grabbed' and stored in data/raw/")