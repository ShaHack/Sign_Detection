# Advanced Sign Detection using MediaPipe

This project detects hand signs using a webcam and converts them into text and speech.

## Features
- Real-time hand sign detection
- MediaPipe Hands (21 landmarks)
- KNN classifier
- Voice output (Text-to-Speech)
- Snapshot saving per detected sign
- Command logging

## Requirements
- Python 3.9+
- OpenCV
- MediaPipe 0.10.8
- NumPy
- scikit-learn

## Installation
```bash
pip install numpy opencv-python mediapipe==0.10.8 scikit-learn pandas joblib pyttsx3