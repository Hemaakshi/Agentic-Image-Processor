# Medical Dataset Setup Guide

To reach the goal of 1000+ images per use case, follow these links to download the raw DICOM/NIfTI data.

### 1. Pneumonia Triage (Chest X-Ray)

- **Source:** [Stanford CheXpert](https://aimi.stanford.edu/shared-datasets)
- **Count:** 224,000+ images.
- **Action:** Register on the site to get the download link for the "Small" or "Full" dataset.
- **What it detects:** Bacterial or viral infections in the lungs.
- **Target Finding:** Opacity (cloudy areas) in the lung fields that indicate fluid or inflammation.
- **Agent Task:** Identify DX modality and route to a "Classifier" to distinguish between Normal vs. Pneumonia.

### 2. Lung Nodule CT (Chest CT)

- **Source:** [TCIA LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/)
- **Count:** 1,018 cases (thousands of DICOM slices).
- **Action:** Download the `.tcia` manifest file and use the "NBIA Data Retriever" to pull the images.
- **What it detects:** Early-stage lung cancer.
- **Target Finding:** Small, round "nodules" (spots) within the lung parenchyma.
- **Agent Task:** Identify CT modality and perform 3D slice-by-slice analysis to flag suspicious growths.

### 3. Brain Tumor Analysis (MRI)

- **Source:** [Mendeley MRI Dataset](https://data.mendeley.com/datasets/d73rs38yk6/1)
- **Count:** 1,000+ images.
- **Action:** Direct download the ZIP file and extract into `data/raw/`.
- **What it detects:** Gliomas, Meningiomas, or Pituitary tumors.
- **Target Finding:** Abnormal tissue masses that displace normal brain structures, often appearing brighter (hyperintense) on specific MRI sequences.
- **Agent Task:** Identify MR modality and normalize the contrast to make the tumor boundaries sharper for segmentation.

### 4. Breast Cancer Screening (Mammography)

- **Source:** [TCIA CMMD](https://www.cancerimagingarchive.net/collection/cmmd/)
- **Count:** 5,200+ images.
- **Action:** Download the manifest and use the NBIA Retriever.
- **What it detects:** Malignant vs. Benign breast lesions.
- **Target Finding:** Calcifications (tiny white dots) or dense masses/architectural distortions in the breast tissue.
- **Agent Task:** Identify MG modality and apply high-resolution enhancement to spot tiny calcium deposits.
