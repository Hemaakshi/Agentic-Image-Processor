# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models
import torchvision.transforms as transforms
from PIL import Image as PILImage

# Limit CPU threads to prevent OOM on HF free tier
torch.set_num_threads(2)
from skimage.feature import hog, graycomatrix, graycoprops
import gradio as gr

from huggingface_hub import snapshot_download
import shutil

# Download specialist models from XET storage at runtime
repo_id = "HEMAAKSHI/Mri_scan"
local_dir = snapshot_download(repo_id=repo_id, repo_type="space")
src = os.path.join(local_dir, "specialist_models")
if os.path.exists(src) and not os.path.exists("specialist_models"):
    shutil.copytree(src, "specialist_models")

# DEBUG - remove after fixing
import glob
print("=== DEBUG: local_dir =", local_dir)
print("=== DEBUG: src path =", src)
print("=== DEBUG: src exists =", os.path.exists(src))
print("=== DEBUG: specialist_models exists =", os.path.exists("specialist_models"))
print("=== DEBUG: files in local_dir =", os.listdir(local_dir))
# =====================================================================
# 1. CORE IMAGE PREPROCESSING FUNCTIONS (All Functions Preserved)
# =====================================================================

def load_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_image(img, size=(224, 224)):
    return cv2.resize(img, size)

def normalize_image(img):
    return img.astype(np.float32) / 255.0

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def median_blur(img, ksize=5):
    return cv2.medianBlur(img, ksize)

def bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

def histogram_equalization(gray_img):
    return cv2.equalizeHist(gray_img)

def clahe_enhancement(gray_img, clip_limit=2.0, grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(gray_img)

def canny_edges(gray_img, low=100, high=200):
    return cv2.Canny(gray_img, low, high)

def binary_threshold(gray_img, thresh=127):
    _, out = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY)
    return out

def otsu_threshold(gray_img):
    _, out = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out

def get_kernel(size=3):
    return np.ones((size, size), np.uint8)

def erosion(img, kernel_size=3, iterations=1):
    kernel = get_kernel(kernel_size)
    return cv2.erode(img, kernel, iterations=iterations)

def dilation(img, kernel_size=3, iterations=1):
    kernel = get_kernel(kernel_size)
    return cv2.dilate(img, kernel, iterations=iterations)

def opening(img, kernel_size=3):
    kernel = get_kernel(kernel_size)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def closing(img, kernel_size=3):
    kernel = get_kernel(kernel_size)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def simple_mask(gray_img, thresh=30):
    _, mask = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY)
    return mask

def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)

def sharpen_image(img):
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

# Unified wrappers to translate single-channel filters for 3-channel UI rendering
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def clahe_wrap(img):
    gray = to_gray(img)
    c = cv2.createCLAHE(2.0, (8,8)).apply(gray)
    return cv2.cvtColor(c, cv2.COLOR_GRAY2RGB)

def median_wrap(img):
    return cv2.medianBlur(img, 5)

def gaussian_wrap(img):
    return cv2.GaussianBlur(img, (5,5), 0)

# Master Map for the interactive app rule-execution engine
FUNCTION_MAP = {
    "clahe": clahe_wrap,
    "median": median_wrap,
    "gaussian": gaussian_wrap,
    "sharpen": sharpen_image
}

# =====================================================================
# 2. FEATURE EXTRACTION UTILITIES (Preserved for your notebook use)
# =====================================================================

def extract_hog(gray_img):
    features, hog_img = hog(gray_img, visualize=True)
    return features, hog_img

def extract_glcm_features(gray_img):
    glcm = graycomatrix(gray_img, [1], [0], 256, symmetric=True, normed=True)
    return {
        "contrast": graycoprops(glcm, 'contrast')[0,0],
        "correlation": graycoprops(glcm, 'correlation')[0,0],
        "energy": graycoprops(glcm, 'energy')[0,0],
        "homogeneity": graycoprops(glcm, 'homogeneity')[0,0]
    }

# =====================================================================
# 3. METRIC ANALYZER & DECISION AGENT ENGINE
# =====================================================================

def analyze_image_metrics(img):
    img_uint8 = img.astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    
    brightness = np.mean(gray)
    contrast = np.std(gray)
    noise = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
    
    return {
        "brightness": brightness,
        "contrast": contrast,
        "noise": noise,
        "edge_density": edge_density
    }

def normalize_metrics(metrics):
    return {
        "brightness": metrics["brightness"] / 255.0,
        "contrast": metrics["contrast"] / 128.0,
        "noise": metrics["noise"] / 1000.0,
        "edge_density": metrics["edge_density"]
    }

def decision_agent(metrics):
    m = normalize_metrics(metrics)
    steps = []
    if m["contrast"] < 0.3:
        steps.append(("clahe", {}))
    if m["noise"] > 0.3:
        steps.append(("median", {}))
    if m["edge_density"] < 0.05:
        steps.append(("sharpen", {}))
    return steps

# =====================================================================
# 4. SPECIALIST PYTORCH ARCHITECTURES
# =====================================================================

class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.dw(x)))
        x = F.relu(self.bn2(self.pw(x)))
        return x

class LiteResBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = DSConv(in_channels, out_channels, stride)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))

class LiteBrainNet2(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = LiteResBlock2(32, 64, stride=2)
        self.layer2 = LiteResBlock2(64, 128, stride=2)
        self.layer3 = LiteResBlock2(128, 128, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class InfectiousBrainNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class StableMetabolicNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class NeoplasticBrainNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop2d_1 = nn.Dropout2d(p=0.2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop2d_2 = nn.Dropout2d(p=0.3)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.drop2d_1(x)
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        x = self.drop2d_2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class CustomLiverNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.global_pool(x)
        x = self.fc_layers(x)
        return x

class MicroLiverNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

# =====================================================================
# 5. TWO-STAGE EXPERT HUB INFERENCE WORKFLOW
# =====================================================================

class MedicalAIHub:
    def __init__(self, paths, gen_class_list):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.paths = paths
        self.gen_classes = gen_class_list

        # Init Generalist Base (ResNet50)
        self.gen = torch_models.resnet50()
        num_ftrs = self.gen.fc.in_features
        self.gen.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, len(gen_class_list))
        )
        
        if os.path.exists(paths['generalist']):
            self.gen.load_state_dict(torch.load(paths['generalist'], map_location=self.device, weights_only=True))
            self.gen.to(self.device).eval()
            print("Generalist system weights verified and running.")
        else:
            print(f"[STARTUP WARNING] Generalist weight file NOT found at: {paths['generalist']}")
            print("Predictions will be random until weights are loaded. Commit specialist_models/ to your HF repo.")
            self.gen.to(self.device).eval()  # Keep running in degraded mode rather than crashing

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def get_specialist_instance(self, category):
        cat = category.lower()
        if 'genetic' in cat:
            diseases = ['Fukuyama Muscular Dystrophy', 'NFM 1 with OGIE', 'Tuberous Sclerosis', 'Walker-Warburg Syndrome']
            return LiteBrainNet2(num_classes=4), self.paths['genetic'], diseases
        if 'infectious' in cat:
            diseases = ['Acute Cerebellitis in HIV', 'Acute Unilateral Cerebellitis in HIV', 'Congenital Toxoplasmosis', 'Japanese B Encephalitis or Epstein-Barr Encephalitis', 'Rasmussens Encephalitis']
            return InfectiousBrainNet(num_classes=5), self.paths['infectious'], diseases
        if 'malformations' in cat or 'developmental' in cat:
            diseases = ['Balloon Cell Cortical Dysplasia', 'Pachygyria with Cerebellar Hypoplasia', 'Perisylvian Syndrome']
            return LiteBrainNet2(num_classes=3), self.paths['malformations'], diseases
        if 'metabolic' in cat:
            diseases = ['Osmotic Demyelination Syndrome', 'Typical Adrenoleukodystrophy']
            return StableMetabolicNet(num_classes=2), self.paths['metabolic'], diseases
        if 'neoplastic' in cat or 'tumor' in cat or 'tumour' in cat:
            diseases = ['Optic Glioma', 'Plexiform Neurofibroma with Sphenoid Wing Dysplasia']
            return NeoplasticBrainNet(num_classes=2), self.paths['neoplastic'], diseases
        if 'malignant' in cat:
            diseases = ['Hepatocellular Carcinoma (HCC) and Dysplastic Nodule', 'Hepatocellular_Carcinoma', 'Inferior Vena Cava (IVC) Leiomyosarcoma']
            return CustomLiverNet(num_classes=3), self.paths['malignant'], diseases
        if 'ductal' in cat or 'ductual' in cat:
            diseases = ['Carolis Disease', 'Cholangiocarcinoma']
            return MicroLiverNet(num_classes=2), self.paths['ductal'], diseases
        return None, None, None

    def infer(self, processed_cv2_img):
        # Stage 1: Generalist Category Check
        tensor_img = self.transform(processed_cv2_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            gen_outputs = self.gen(tensor_img)
            gen_probs = F.softmax(gen_outputs, dim=1)
            gen_conf, gen_idx = torch.max(gen_probs, 1)
            
        predicted_category = self.gen_classes[gen_idx.item()]
        
        if predicted_category == 'Healthy' or predicted_category == 'Magnetic Resonance (MR) Brain':
            return f"Generalist Assessment: {predicted_category}", f"{round(gen_conf.item() * 100, 2)}%"

        # Stage 2: Specialist Specific Deep Model Execution
        spec_model, spec_weight_path, disease_classes = self.get_specialist_instance(predicted_category)
        
        if spec_model is not None and os.path.exists(spec_weight_path):
            try:
                spec_model.load_state_dict(torch.load(spec_weight_path, map_location=self.device, weights_only=True))
                spec_model.to(self.device).eval()
                
                with torch.no_grad():
                    spec_outputs = spec_model(tensor_img)
                    spec_probs = F.softmax(spec_outputs, dim=1)
                    spec_conf, spec_idx = torch.max(spec_probs, 1)
                
                final_condition = disease_classes[spec_idx.item()]
                return f"[{predicted_category.upper()}] -> {final_condition}", f"{round(spec_conf.item() * 100, 2)}%"
            except Exception as e:
                return f"Specialist Runtime Error ({predicted_category})", "0.0%"
        else:
            return f"Generalist Assessment: {predicted_category} (No specialist required)", f"{round(gen_conf.item() * 100, 2)}%"

# =====================================================================
# 6. RUNTIME INITIALIZATION
# =====================================================================

BASE_PATH = "specialist_models/"
my_model_paths = {
    'generalist':    BASE_PATH + 'unified_10class_model.pth',
    'genetic':       BASE_PATH + 'brain_genetic_custom_lite_.pth',
    'infectious':    BASE_PATH + 'infectious_custom_specialist.pth',
    'malformations': BASE_PATH + 'developmental_malformations_lite_92plus.pth',
    'metabolic':     BASE_PATH + 'metabolic_custom_specialist.pth',
    'neoplastic':    BASE_PATH + 'neoplastic_custom_specialist.pth',
    'malignant':     BASE_PATH + 'liver_custom_malignant_classifier.pth',
    'ductal':        BASE_PATH + 'liver_custom_Ductual_micro_final.pth'
}

gen_classes = sorted([
    'Healthy', 'genetic', 'vascular', 'Benign',
    'Retinoblastoma with Intracranial Spread Along Cranial Nerve',
    'Ductual', 'Magnetic Resonance (MR) Brain',
    'developmental brain malformations', 'infectious',
    'tumours or neoplastic', 'metabolic', 'Malignant'
])

hub = MedicalAIHub(my_model_paths, gen_classes)

# =====================================================================
# 7. GRADIO INTERFACE EXECUTION LOGIC
# =====================================================================

def diagnose(image):
    if image is None:
        return None, "No image provided", "—", "—"

    metrics = analyze_image_metrics(image)
    steps = decision_agent(metrics)

    processed_img = image.copy()
    for step_name, _ in steps:
        if step_name in FUNCTION_MAP:
            processed_img = FUNCTION_MAP[step_name](processed_img)

    steps_text = "\n".join([f"{i+1}. {s[0].replace('_',' ').title()}"
                            for i, s in enumerate(steps)]) if steps else "No preprocessing needed (Optimal Input)"

    final_diagnosis, confidence_score = hub.infer(processed_img)

    return processed_img, steps_text, final_diagnosis, confidence_score

# =====================================================================
# 8. GRADIO USER INTERFACE LAYOUT
# =====================================================================

with gr.Blocks(theme=gr.themes.Monochrome(), title="NeuroScan") as demo:
    gr.Markdown("# NeuroScan — Hierarchical MRI Diagnostic Workspace")

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload MRI Scan Graphic", type="numpy")
            run_btn = gr.Button("▶ Run Diagnostic Evaluation", variant="primary")
        with gr.Column(scale=1):
            original_img  = gr.Image(label="Original Viewport Input Reference", interactive=False)
            processed_img = gr.Image(label="Agent Normalized Graphic Result Output", interactive=False)

    with gr.Row():
        steps_out    = gr.Textbox(label="Adaptive Agent Pipeline Processing Logs", lines=3)
        pred_out     = gr.Textbox(label="Calculated Hierarchical Diagnostic Output")
        proc_conf_out = gr.Textbox(label="Certainty Confidence Matrix Score")

    run_btn.click(
        fn=lambda img: (img, *diagnose(img)),
        inputs=input_img,
        outputs=[original_img, processed_img, steps_out, pred_out, proc_conf_out]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)