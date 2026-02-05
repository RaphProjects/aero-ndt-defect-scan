# AeroScan — AI-Powered Non-Destructive Testing (NDT) for Surface Defect Detection
<img width="1775" height="661" alt="image" src="https://github.com/user-attachments/assets/ac39230a-0ab2-41e0-aaf6-b187dddefd0d" />


AeroScan is a prototype smart inspection assistant for metallic surfaces (inspired by the aerospace engine manufacturing/maintenance context).  
The goal is to detect **unknown anomalies** (scratches, cracks, inclusions, etc.) using a **non-supervised anomaly detection** approach based on a **convolutional autoencoder**.

The system:
- reconstructs a “healthy-looking” version of the surface,
- computes a **reconstruction error map** (heatmap),
- outputs a simple decision: **OK / ANOMALY DETECTED**.

---

## 1) Business Context

In aerospace production and maintenance, detecting micro-defects on critical parts (blades, casings, etc.) is mandatory to ensure safety and reliability.  
Human visual inspection is:
- expensive,
- affected by fatigue,
- hard to standardize.

AeroScan aims to turn a complex mathematical output (pixel-wise reconstruction error) into an **operator-friendly diagnostic tool**.

---

## 2) Dataset

**Source:** NEU Surface Defect Database (NEU-DET), Northeastern University.  
The dataset contains grayscale-like texture images of hot-rolled steel surfaces with **6 defect categories**:

- `crazing`
- `inclusion`
- `patches`
- `pitted_surface`
- `rolled-in_scale`
- `scratches`

**Format & structure:**
- Images: `.jpg`, size **200×200**
- Annotations: `.xml` (Pascal VOC style bounding boxes)

> Note: The model is trained in a non-supervised way (no manual class labels needed for training).  
> However, XML bounding boxes are used to **extract “clean” patches** (no defect overlap) to approximate “normal” data.

---

## 3) Anomaly Detection Principle (Unsupervised)

1. Train an autoencoder on patches assumed to be **defect-free**.
2. At inference time, reconstruct each patch and compute a per-pixel error.
3. A high reconstruction error indicates an anomaly.

**Prototype anomaly score:**
- pixel error: `|x - x_recon|`
- patch decision: `max(error) > threshold`

---

## 4) Data Pipeline

### 4.1 Clean patch extraction
- Each 200×200 image is split into **50×50 patches** (16 patches per image).
- A patch is considered “clean” if it does **not overlap any defect bounding box** from the XML.
- Output dataset: **7077 clean patches** used for training.

---

## 5) AI Model

### 5.1 Convolutional Autoencoder (CAE)
Input: `(1, 50, 50)` (single-channel grayscale)

Encoder:
- Conv2d (stride=2) → 32 channels
- Conv2d (stride=2) → 64 channels
- Conv2d (stride=2) → 128 channels
- Bottleneck: `(128, 7, 7)`

Decoder:
- ConvTranspose2d (stride=2) → 64 channels
- ConvTranspose2d (stride=2) → 32 channels
- ConvTranspose2d (stride=2) → 1 channel
- Final activation: **Sigmoid** (output range [0, 1])

### 5.2 Optional Spatial Attention
A **spatial attention module** (CBAM-like) was tested at the bottleneck:
- builds an attention map `(1, H, W)` from channel-wise `avg` + `max`,
- applies `x * attention`.

**Experimental result:** limited/non-systematic improvements on this dataset (useful insight: architecture changes do not always overcome dataset/annotation limitations).

---

## 6) Evaluation

### 6.1 Category-level analysis (pixel defect vs error correlation)
On validation data, the model shows:
- very strong signal for `scratches`,
- weaker signal for subtle texture-like defects (e.g., `crazing`),
- sometimes inverted signal for `patches` (patch defects may look more uniform and can be reconstructed well).

### 6.2 Patch-level classification metrics
Each 50×50 patch is treated as:
- **y_true = 1** if the patch overlaps any ground-truth defect bounding box
- **y_pred = 1** if the model flags the patch as anomalous (threshold decision)

Validation results:

| Metric | Score |
|-------:|:------|
| Accuracy | **0.7389** |
| Precision | **0.8083** |
| Recall | **0.8406** |
| F1-score | **0.8241** |

> In industrial inspection, **Recall** is often the most critical metric (better to raise a false alarm than to miss a real defect).

---

## 7) Demo Application (Streamlit)

A Streamlit dashboard provides:
- image upload,
- patch-wise analysis,
- a red heatmap (higher error = more suspicious),
- a final OK / anomaly decision.

---

## 8) Repository Structure (example)
AeroScan/
├── app/
│ └── streamlit_app.py
├── src/
│ ├── annotation_parser.py
│ ├── patch_extractor.py
│ ├── dataset.py
│ ├── autoencoder.py
│ ├── train.py
│ └── detect.py
├── models/ # trained weights
├── data/ # dataset (not versioned)
├── .gitignore
├── requirements.txt
└── README.md


---

## 9) Installation

### 9.1 Dependencies (indicative)
- Python 3.12+ (tested on Windows)
- `torch`
- `numpy`
- `Pillow`
- `matplotlib`
- `streamlit`
- (optional) `scikit-learn` for metrics

> The project can run without a virtual environment (disk constraints), as long as required packages are installed globally.

---

## 10) Usage

### 10.1 Dataset placement
Download NEU-DET and place it for example as:
- `data/raw/NEU-DET/train/images/...`
- `data/raw/NEU-DET/train/annotations/...`
- `data/raw/NEU-DET/validation/images/...`
- `data/raw/NEU-DET/validation/annotations/...`

### 10.2 Generate clean patches
Run your extraction script (implemented in `patch_extractor.py`) to create:
- `data/processed/clean_patches/`

### 10.3 Training
Run `src/train.py` to train the autoencoder on clean patches.  
Weights are saved under `models/`. You can also directly one of the provided models.

### 10.4 Evaluation
Run `src/detect.py` to compute:
- pixel-level correlation analyses,
- patch-level classification metrics (accuracy, precision, recall, f1).

### 10.5 Run the web app
Start Streamlit:
- `streamlit run app/streamlit_app.py`

---

## 11) Git Best Practices (.gitignore)
This repo does **not** version:
- `data/` (large dataset files),
- Python caches, OS files, notebook checkpoints, IDE settings.

---

## 12) Limitations & Future Improvements

### Current limitations
- Annotations are **bounding boxes**: many “defect pixels” in the box are actually normal background → evaluation noise.
- Some defects are close to the normal texture (e.g., `crazing`) → harder separation.
- Simple decision rule `max(error) > threshold`.

### Potential improvements
- Better anomaly scores: quantiles, mean-over-top-k, SSIM-based scores, etc.
- Perceptual / SSIM reconstruction losses instead of MSE.
- Use truly defect-free images (industrial data collection).
- Stronger architectures: multi-scale AE, U-Net AE.

---

## 13) References / Credits
- NEU Surface Defect Database (NEU-DET), Northeastern University.
- Concepts: convolutional autoencoder anomaly detection, reconstruction error maps, spatial attention (CBAM-like).

---

## 14) Author
Project built as a portfolio prototype for an aerospace-oriented machine learning apprenticeship:
- data engineering → PyTorch training → evaluation metrics → Streamlit dashboard,
- focus on explainability (heatmaps) and practical deployment.
