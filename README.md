# Deepfake Ensemble Detection

This repository implements an ensemble of state-of-the-art deepfake detectors, combining spatial, temporal, frequency- and transformer-based branches via both score-level and attention-based feature-level fusion.  

---

## 🔍 Project Overview

1. **Base Models**  
   - **SPSLDetector**: spatial and frequency artifacts via FFT + specialized ResNet  
   - **UCFDetector**: CNN+GAN hybrid for generalizable forgery cues  
   - **UIAViTDetector**: Vision-Transformer for frame-level analysis  
   - **STILDetector**: spatio-temporal Inception-style network  
   - **CNNImageBranch**: vanilla image CNN branch

2. **Fusion Modes**  
   - **Decision (score) fusion**: model-wise vote or learned meta-classifier  
   - **Feature fusion**: attention-weighted combination of branch features  

3. **Datasets**  
   - **Train/Val**: FaceForensics++  
   - **Test**: Celeb-DF-v1, Celeb-DF-v2, DeepFakeDetection  

---

## ⚙️ Requirements

- Python ≥ 3.11.11
- CUDA ≥ 12.8.1
- PyTorch ≥ 2.8.0


# DeepFake Detection Experiment Setup

This repository contains scripts and configurations for preprocessing datasets, training base models, and evaluating an ensemble for deepfake detection.

## Setup Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation

1. Place your datasets into the `datasets/rgb` folder:

   ```
   /workspace/datasets/rgb
   ```
2. Run the preprocessing scripts to extract faces and generate JSON files (for more detailes refer to [DeepFakeBench](https://github.com/SCLBD/DeepfakeBench)):

   ```bash
   python preprocess.py
   python rearrange.py
   ```
3. Download the DLIB 81-point face landmark predictor file:

   * [`shape_predictor_81_face_landmarks.dat`](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/shape_predictor_81_face_landmarks.dat)
4. Note: We used a fixed frame stride of 5 during preprocessing, resulting in \~1.5 million frames across all datasets (FF++ (c23), Celeb-DF v1 & v2, and DeepFakeDetection). Ensure you have sufficient storage.

## Training Base Models

Train models individually on each dataset:

```bash
python3 -m src.training.train \
  --detector_path ./src/training/config/detector/stil.yaml \
  --train_dataset "FaceForensics++" \
  --test_dataset "Celeb-DF-v1" "Celeb-DF-v2" "DeepFakeDetection"

python3 -m src.training.test \
  --detector_path ./src/training/config/detector/stil.yaml \
  --weights_path "/workspace/weights/stil_best.pth" \
  --test_datasets "Celeb-DF-v1" "Celeb-DF-v2" "DeepFakeDetection"
```

## Ensemble Training

```bash
python3 ensembleTraining.py \
  --config /workspace/src/training/config/detector/ensemble.yaml
```

## Ensemble Testing

```bash
python3 test_ensemble.py \
  --config /workspace/src/training/config/detector/ensemble.yaml
```

## Compute Results

Run the result computation script on all prediction CSV files:

```bash
python3 compute_results.py \
  ./predictions/spsl/Celeb-DF-v1.csv \
  ./predictions/spsl/Celeb-DF-v2.csv \
  ./predictions/spsl/DeepFakeDetection.csv \
  ./predictions/ucf/Celeb-DF-v1.csv \
  ./predictions/ucf/Celeb-DF-v2.csv \
  ./predictions/ucf/DeepFakeDetection.csv \
  ./predictions/stil/Celeb-DF-v1.csv \
  ./predictions/stil/Celeb-DF-v2.csv \
  ./predictions/stil/DeepFakeDetection.csv \
  ./predictions/uia_vit/Celeb-DF-v1.csv \
  ./predictions/uia_vit/Celeb-DF-v2.csv \
  ./predictions/uia_vit/DeepFakeDetection.csv \
  ./predictions/decision_fusion/Celeb-DF-v1.csv \
  ./predictions/decision_fusion/Celeb-DF-v2.csv \
  ./predictions/decision_fusion/DeepFakeDetection.csv \
  ./predictions/feature_fusion/Celeb-DF-v1.csv \
  ./predictions/feature_fusion/Celeb-DF-v2.csv \
  ./predictions/feature_fusion/DeepFakeDetection.csv
```
