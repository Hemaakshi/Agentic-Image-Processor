# Medical Dataset Setup Guide

To reach the goal of 1000+ images per use case, follow these links to download the raw DICOM/NIfTI data.

### 1. Pneumonia Triage (Chest X-Ray)

- **Source:** [Stanford CheXpert](https://aimi.stanford.edu/shared-datasets)
- **Count:** 224,000+ images.
- **Action:** Register on the site to get the download link for the "Small" or "Full" dataset.

### 2. Lung Nodule CT (Chest CT)

- **Source:** [TCIA LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/)
- **Count:** 1,018 cases (thousands of DICOM slices).
- **Action:** Download the `.tcia` manifest file and use the "NBIA Data Retriever" to pull the images.

### 3. Brain Tumor Analysis (MRI)

- **Source:** [Mendeley MRI Dataset](https://data.mendeley.com/datasets/d73rs38yk6/1)
- **Count:** 1,000+ images.
- **Action:** Direct download the ZIP file and extract into `data/raw/`.

### 4. Breast Cancer Screening (Mammography)

- **Source:** [TCIA CMMD](https://www.cancerimagingarchive.net/collection/cmmd/)
- **Count:** 5,200+ images.
- **Action:** Download the manifest and use the NBIA Retriever.
