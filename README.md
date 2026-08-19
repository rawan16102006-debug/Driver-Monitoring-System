🚗 Driver Monitoring System

A Computer Vision system for detecting key driver behaviors and safety indicators from video using YOLO11s.

🔍 What It Detects

- Open Eye
- Closed Eye
- Cigarette
- Phone
- Seatbelt

⚙️ Pipeline

Dataset Audit → Cleaning → YOLO11s Training → Validation → Error Analysis → Test Evaluation → Streamlit Deployment

The dataset was inspected for corrupted images, duplicates, annotation consistency, class distribution, and image/label matching before training.

📊 Final Test Results

| Metric | Score |
| Precision | 90.5% |
| Recall | 85.8% |
| mAP@50 | 91.0% |
| mAP@50-95 | 66.99% |

🌐 Deployment

The final model is deployed using Streamlit, where users can upload a driver-monitoring video, adjust the confidence threshold, run detection, preview the annotated video, and download the processed result.

🛠️ Tech Stack

Python · YOLO11s · Ultralytics · OpenCV · Streamlit . av

▶️ Run Locally

bash
pip install -r requirements.txt
streamlit run app.py