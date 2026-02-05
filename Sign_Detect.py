"""
Advanced Sign Detection (Bug-Fixed Version)
-------------------------------------------
Usage:
  1) Collect samples:  python sign_detect.py collect hello
  2) Train model:      python sign_detect.py train
  3) Run detection:    python sign_detect.py run
"""

import sys, os, csv, time, argparse
from datetime import datetime

import numpy as np
import cv2
import mediapipe as mp
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

import joblib
import pyttsx3


DATA_CSV = "hand_sign_data.csv"
MODEL_FILE = "hand_sign_knn.joblib"
SCALER_FILE = "scaler.joblib"
COMMAND_LOG = "detected_commands.txt"


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ------------------ FEATURE EXTRACTION ------------------

def extract_landmarks(results):
    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]
    lm = []

    for p in hand.landmark:
        lm.extend([p.x, p.y, p.z])

    arr = np.array(lm).reshape(-1, 3)

    wrist = arr[0]
    arr = arr - wrist

    scale = np.max(np.abs(arr))
    if scale == 0:
        scale = 1.0

    arr = arr / scale
    return arr.flatten().tolist()


# ------------------ DATA COLLECTION ------------------

def collect(label):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not found.")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        count = 0
        print(f"Collecting samples for '{label}'")
        print("Press SPACE to save | Press q to quit")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                for hl in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"Label: {label}  Saved: {count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Collect", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                feats = extract_landmarks(res)
                if feats:
                    new_file = not os.path.isfile(DATA_CSV)
                    with open(DATA_CSV, "a", newline="") as f:
                        w = csv.writer(f)
                        if new_file:
                            w.writerow(["label"] + [f"f{i}" for i in range(len(feats))])
                        w.writerow([label] + feats)
                    count += 1
                    print(f"✔ Sample {count} saved")
                else:
                    print("⚠ No hand detected")

            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# ------------------ TRAINING ------------------

def train():
    if not os.path.exists(DATA_CSV):
        print("❌ No dataset found. Collect data first.")
        return

    df = pd.read_csv(DATA_CSV)

    if df["label"].value_counts().min() < 2:
        print("❌ Each label needs at least 2 samples.")
        return

    X = df.drop("label", axis=1).values
    y = df["label"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y, test_size=0.2, stratify=y, random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(Xtr, ytr)

    acc = knn.score(Xte, yte)
    print(f"Hold-out Accuracy: {acc*100:.2f}%")

    cv_scores = cross_val_score(knn, Xs, y, cv=5)
    print(f"Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}%")

    joblib.dump(knn, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    print("✔ Model and scaler saved")


# ------------------ LIVE DETECTION ------------------

def run_detection():
    if not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE)):
        print("❌ Train the model first.")
        return

    knn = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    engine = pyttsx3.init()
    last_label = ""
    last_time = 0
    cooldown = 1.5
    prob_threshold = 0.6
    transcript = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera error.")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        print("Running detection... Press q to quit")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            label = "No hand"
            conf = 0.0

            if res.multi_hand_landmarks:
                for hl in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

                feats = extract_landmarks(res)
                if feats:
                    X = scaler.transform([feats])

                    if hasattr(knn, "predict_proba"):
                        probs = knn.predict_proba(X)[0]
                        idx = np.argmax(probs)
                        label = knn.classes_[idx]
                        conf = probs[idx]
                    else:
                        label = knn.predict(X)[0]
                        conf = 1.0

                    now = time.time()
                    if conf >= prob_threshold and (label != last_label or now - last_time > cooldown):
                        try:
                            engine.say(label)
                            engine.runAndWait()
                        except:
                            pass

                        os.makedirs("detected_words", exist_ok=True)
                        os.makedirs(f"snapshots/{label}", exist_ok=True)

                        with open(COMMAND_LOG, "a") as f:
                            f.write(f"{datetime.now().isoformat()} | {label} | {conf:.2f}\n")

                        with open(f"detected_words/{label}.txt", "a") as f:
                            f.write(f"{datetime.now().isoformat()} | {label}\n")

                        snap = f"snapshots/{label}/{datetime.now().strftime('%H%M%S')}.jpg"
                        cv2.imwrite(snap, frame)

                        transcript.append(label)
                        last_label = label
                        last_time = now

            color = (0, 0, 255)
            if conf > 0.8:
                color = (0, 255, 0)
            elif conf > 0.6:
                color = (0, 255, 255)

            cv2.putText(frame, f"{label} ({conf:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            text = " ".join(transcript[-10:])
            cv2.putText(frame, text,
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

            cv2.imshow("Sign Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# ------------------ MAIN ------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["collect", "train", "run"])
    parser.add_argument("label", nargs="?", default=None)
    args = parser.parse_args()

    if args.mode == "collect":
        if not args.label:
            print("❌ Provide label name")
            return
        collect(args.label)

    elif args.mode == "train":
        train()

    elif args.mode == "run":
        run_detection()


if __name__ == "__main__":
    main()