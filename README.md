# Advanced Hand Sign Detection using MediaPipe

This project implements a real-time hand sign detection system using a webcam.
MediaPipe Hands is used for landmark extraction and a K-Nearest Neighbors (KNN)
classifier is used to recognize hand signs and convert them into text and speech.

## Features
- Real-time hand sign detection using MediaPipe Hands
- Wrist-centered and scale-normalized landmark features
- KNN (K-Nearest Neighbors) classifier
- Live confidence-based prediction display
- Text-to-Speech output
- Snapshot saving for detected signs
- Command logging with timestamps

## Requirements
- Python 3.9 or higher
- Webcam

## Install Dependencies
pip install numpy opencv-python mediapipe==0.10.8 pandas scikit-learn joblib pyttsx3

## Usage

Collect training data:
python Sign_Detect.py collect hello

Train the model:
python Sign_Detect.py train

Run live detection:
python Sign_Detect.py run
