# AI-Based Real-Time Posture Monitoring System
## overview

A real-time posture monitoring system that uses computer vision, geometric analysis, and machine learning to detect poor sitting posture and provide intelligent, non-intrusive feedback.

This project was designed as an end-to-end applied AI system, not just a model demo: it includes perception, reasoning, personalization, ML training, evaluation, and user feedback.

##  Features

- Real-time webcam posture tracking
- Pose estimation using body landmarks
- Geometric posture metrics (neck angle, shoulder alignment)
- Rule-based posture classification
- Personalized ML-based posture classifier
- Temporal smoothing to avoid noisy predictions
- Audio + visual alerts with cooldown
- User-labeled data collection for ML
- Offline evaluation: rule-based vs ML comparison

<p align="center">
  <img src="https://github.com/user-attachments/assets/b6ad0806-5632-4cee-8fc3-db0437334334" width="32%" />
  <img src="https://github.com/user-attachments/assets/e05f5ee6-85fe-4874-bf94-3e98053a881a" width="32%" />
  <img src="https://github.com/user-attachments/assets/49a2bf23-55bc-4bc5-8991-77db904d25f4" width="32%" />
</p>

## System Architecture

The system is intentionally modular and layered:
```bash
Camera
  ↓
Pose Estimation (MediaPipe)
  ↓
Posture Metrics (Geometry)
  ↓
Classifier (Rule-based OR ML-based)
  ↓
Temporal Smoothing
  ↓
Alert Manager
  ↓
UI + Audio Feedback
```


## Posture Metrics

The system does not classify posture directly from images.

Instead, it computes interpretable geometric features from pose landmarks:

- Neck angle
Measures forward-head posture (slouching)

- Shoulder height difference
Detects leaning or asymmetric posture

These metrics are:

- camera-agnostic
- explainable
- reusable for ML and analytics


## Classification Approaches
1) Rule-Based Classifier
- Uses hand-crafted thresholds on posture metrics:
- Good / Warning / Bad posture
- Simple and interpretable
- Serves as a baseline

2) ML-Based Classifier (Personalized)
A logistic regression model trained on user-labeled posture data:

- Input: posture metrics
- Output: probability of bad posture
- Personalized per user and setup
- Reduces false alerts in borderline cases

At runtime, the user can choose rule-based or ML-based classification.


## Data Collection & Personalization

The system supports personalized data collection:

Metrics are logged at fixed intervals (no images stored)

User labels posture via keyboard:

- G → Good posture
- B → Bad posture
- U → Unknown / ignore

This data is saved as CSV and used to:

- learn personalized posture patterns
- train ML models
- evaluate performance


## Machine Learning Pipeline

1) Collect labeled posture metrics
2) Train logistic regression model
3) Evaluate on held-out data
4) Save model (.pkl)
5) Load model for real-time inference


## Evaluation

The ML-based classifier is evaluated against the rule-based baseline on the same recorded sessions.

### Metrics reported:

- Accuracy
- Precision / Recall / F1-score
- Confusion matrices
- Rule vs ML agreement rate

### Key Result

- High agreement (~94%) on clear cases
- ML differs mainly on borderline postures
- ML improves precision (fewer false alerts) while maintaining recall

This validates the use of ML where rules are brittle, not as a replacement for the entire system.


## Tech Stack
- Python
- OpenCV — real-time video processing
- MediaPipe — pose estimation
- NumPy / Pandas — numerical & data handling
- scikit-learn — ML training & evaluation


## How to Run
1. Install dependencies
```bash 
pip install -r requirements.txt
```
3. Run the app
```bash
python src/main.py
```

Choose classifier mode at startup:

[r] rule-based
[m] ML-based

3. Collect data

Press G, B, U during runtime

CSV saved automatically

4. Train ML model
```bash
python -m src.ml.train_model
```
6. Evaluate models
```bash
python -m src.ml.evaluate_model
```

<img width="612" height="510" alt="image" src="https://github.com/user-attachments/assets/d6bbf2fe-2de2-4d5f-a594-30f9bd5f336b" />
