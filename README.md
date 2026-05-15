# Sign Language Detection System

A real-time AI-based Sign Language Detection System built using Python, MediaPipe, OpenCV, and TensorFlow/Keras.

This project allows users to:

* Record custom sign language gestures using webcam
* Train a deep learning model on recorded gestures
* Detect trained signs in real-time using webcam inference

The system works using hand landmark extraction instead of raw image classification, making it lightweight and efficient for real-time gesture recognition.

---

# Project Structure

```text
project/
│
├── trainingrealdata.py
├── realdatatraining.py
│
├── real_asl_data/
│   ├── HELLO/
│   ├── THANK_YOU/
│   ├── LOVE/
│   └── ...
│
├── real_asl_model.h5
├── real_asl_scaler.pkl
├── real_asl_labels.pkl
│
├── README.md
```

---

# Main Files

## 1. `realdatatraining.py`

This file is used to:

* Open webcam
* Record sign gestures
* Extract MediaPipe hand landmarks
* Save landmark sequences
* Train the LSTM deep learning model

### Workflow

```text
Webcam Input
    ↓
Hand Landmark Extraction
    ↓
Sequence Creation
    ↓
Dataset Storage
    ↓
LSTM Model Training
    ↓
Model Saving
```

### Output Files

After training, the following files are generated:

* `real_asl_model.h5`
* `real_asl_scaler.pkl`
* `real_asl_labels.pkl`

These files are required for real-time testing.

---

## 2. `trainingrealdata.py`

This file is used for real-time sign detection.

### Features

* Opens webcam
* Detects hand landmarks live
* Loads trained model
* Predicts trained sign language words
* Displays prediction confidence

### Detection Pipeline

```text
Webcam
    ↓
MediaPipe Hand Tracking
    ↓
Landmark Extraction
    ↓
Sequence Buffer
    ↓
LSTM Prediction
    ↓
Detected Sign Output
```

---

# Dataset Storage

The folder:

```text
real_asl_data/
```

stores all recorded sign gesture sequences.

Each sign has its own folder:

```text
real_asl_data/
│
├── HELLO/
├── LOVE/
├── THANK_YOU/
├── HELP/
└── ...
```

Inside each folder:

* Webcam gesture recordings
* Landmark sequence `.npy` files

These are used for model training.

---

# Technologies Used

* Python
* OpenCV
* MediaPipe
* TensorFlow / Keras
* NumPy
* Scikit-learn

---

# Model Architecture

The project uses:

* MediaPipe Hand Landmark Detection
* LSTM (Long Short-Term Memory) Neural Network

The model learns temporal hand movement patterns for sign recognition.

Each gesture sequence contains:

* 30 frames
* 21 hand landmarks
* 3D coordinates (x, y, z)

Total features per frame:

```text
21 × 3 = 63 features
```

---

# How the System Works

The system does NOT detect objects.

It detects hand gesture movements and sign language patterns based on training data.

Prediction quality depends on:

* Number of training samples
* Consistent gesture recording
* Lighting conditions
* Camera angle
* Hand visibility

The model performs best when testing conditions are similar to training conditions.

---

# How to Run

## Train the Model

```bash
python realtrainingdata.py
```

## Run Real-Time Detection

```bash
python trainingrealdata.py
```

---

# Current Capabilities

The system can:

* Train custom sign words
* Detect trained gestures live
* Perform real-time inference
* Learn user-specific sign movements

Example signs:

* HELLO
* LOVE
* HELP
* BOOK
* CHAIR
* DRINK
* BEAUTIFUL
* DRINK

---

# Future Improvements

Possible future enhancements:

* Multi-hand detection
* Sentence generation
* Transformer-based sequence models
* Higher dataset diversity
* Better generalization for multiple users
* Mobile deployment
* Text-to-speech output

---

# Note

The accuracy of detection depends heavily on the quality and quantity of recorded training data.

More balanced and diverse gesture samples improve real-world performance significantly.
