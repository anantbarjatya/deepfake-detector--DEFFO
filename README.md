# DeepFake Video Detection System

A production-level deepfake video detection system using **Xception CNN + Bidirectional LSTM** architecture.

## Results
| Metric | Score |
|--------|-------|
| ROC-AUC | 83.93% |
| Fake Recall | 100% |

## Architecture
```
Video → 50 Uniform Frames → MTCNN Face Crop
     → Xception CNN (2048-d features)
     → Bidirectional LSTM
     → Fake / Real + Confidence
```

## Tech Stack
- TensorFlow / Keras
- Xception CNN (ImageNet pretrained)
- Bidirectional LSTM
- MTCNN Face Detection
- Flask REST API
- Albumentations Augmentation

## Dataset
Google DeepFakeDetection (DFD) Dataset
- 726 balanced videos (363 real, 363 fake)
- Source: Kaggle

## Setup

### 1. Clone repo
```bash
git clone https://github.com/YOUR_USERNAME/deepfake-detector.git
cd deepfake-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add dataset
```
dataset/
  real/   ← original videos
  fake/   ← deepfake videos
```

### 4. Train model
```bash
python3 deepfake_train.py
```

### 5. Run server
```bash
python3 app.py
```

### 6. Open browser
```
http://127.0.0.1:5001
```

## Model Architecture
- **Backbone:** Xception (pretrained ImageNet)
- **Sequence Model:** Bidirectional LSTM (128 units)
- **Loss:** Focal Loss (γ=2.0)
- **Frames:** 50 uniform samples per video
- **Face Detection:** MTCNN

## Project Structure
```
deepfake-detector/
├── app.py                 # Flask server
├── deepfake_train.py      # Training pipeline
├── requirements.txt       # Dependencies
├── templates/
│   └── index.html         # Frontend UI
└── README.md
```

## Author
Anant Barjatya
