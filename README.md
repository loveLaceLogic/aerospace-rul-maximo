# Aerospace RUL Prediction + IBM Maximo Integration

## Overview

This project implements a Long Short-Term Memory (LSTM) neural network to predict Remaining Useful Life (RUL) of turbofan engines using the NASA CMAPSS FD001 dataset.

Predicted failure windows are automatically transformed into structured IBM Maximo work order payloads, simulating a real-world predictive maintenance workflow.

This project demonstrates:

- Time-series modeling
- Sequence engineering
- Deep learning with PyTorch
- Predictive maintenance logic
- Enterprise system payload generation

---

## Problem Statement

Unexpected engine failure leads to:

- High maintenance costs  
- Operational downtime  
- Safety risks  

The objective of this system is to:

1. Predict Remaining Useful Life (RUL)
2. Detect engines nearing failure
3. Automatically generate work orders before breakdown

---

## Project Architecture

```
data → preprocessing → sequence building → LSTM model → inference → 
failure detection logic → Maximo JSON work order payload
```

---

## Tech Stack

- Python 3.12
- PyTorch
- Pandas
- NumPy
- LSTM (Recurrent Neural Networks)
- JSON payload automation

---

## Project Structure

```
aerospace-rul-maximo/
│
├── data/                # NASA CMAPSS dataset
├── models/              # Trained LSTM model
├── outputs/             # Generated work orders (JSON)
├── src/
│   ├── preprocess.py
│   ├── build_sequences.py
│   ├── model.py
│   ├── train.py
│   ├── infer_fd001_maximo.py
│   ├── infer_and_generate_workorders.py
│   └── push_to_maximo.py
│
└── README.md
```

---

## Model Pipeline

### 1. Preprocessing
- Normalize sensor features
- Group by engine ID
- Construct time-series windows

### 2. Sequence Building
- Fixed-length sliding windows
- RUL label generation
- Train/test separation

### 3. Model Training
- LSTM layers
- Fully connected regression head
- Mean Squared Error (MSE) loss
- Adam optimizer

### 4. Inference
- Predict RUL on test engines
- Apply threshold logic
- Identify critical units

### 5. Work Order Generation
- Build structured Maximo JSON payload
- Assign priority based on predicted RUL
- Export to `outputs/`

---

## Example Output

```json
{
  "assetnum": "Engine_17",
  "description": "Predicted failure within 15 cycles",
  "priority": "High",
  "reportedby": "PredictiveMaintenanceSystem"
}
```

---

## Key Engineering Decisions

- Used LSTM to model temporal dependencies in sensor data
- Implemented sliding window feature engineering
- Designed modular inference-to-payload pipeline
- Separated training and deployment logic
- Structured code for clarity and scalability

---

## Future Improvements

- Attention-based sequence modeling
- Transformer architecture comparison
- REST API endpoint for real-time scoring
- Cloud deployment (AWS, Azure, or GCP)
- Integration with live CMMS system

---

## Author

Kristina Chaleunsak  
Bachelor of Science in Computer Science  
Focus Areas: Machine Learning, Enterprise Systems, Predictive Analytics
