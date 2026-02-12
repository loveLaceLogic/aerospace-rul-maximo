import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch

from src.preprocess import load_data, add_rul, scale_features, make_sequences
from src.model import LSTMRegressor


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def risk_tier(pred_rul: float) -> str:
    """
    Simple risk mapping for maintenance priority.
    Adjust thresholds to match your preferred logic.
    """
    if pred_rul < 30:
        return "high"
    if pred_rul < 75:
        return "medium"
    return "low"


def main():
    BASE_DIR = Path(__file__).resolve().parents[1]
    models_dir = BASE_DIR / "models"
    outputs_dir = BASE_DIR / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    # Load metadata
    meta_path = models_dir / "meta_fd001.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. Run training first: python3 -m scripts.train"
        )

    meta = json.loads(meta_path.read_text())
    seq_len = int(meta["seq_len"])
    rul_cap = int(meta["rul_cap"])
    feature_cols = meta["feature_cols"]

    # Load scaler
    scaler_path = models_dir / meta["scaler_file"]
    scaler = joblib.load(scaler_path)

    # Load model
    model_path = models_dir / meta["model_file"]
    device = get_device()

    model = LSTMRegressor(input_size=len(feature_cols))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Input data (you can swap train_FD001.txt to test_FD001.txt later)
    data_path = BASE_DIR / "data" / "train_FD001.txt"
    df = load_data(str(data_path))
    df = add_rul(df)

    # IMPORTANT: use the TRAINED scaler (no refit)
    df, _ = scale_features(df, feature_cols, scaler=scaler)

    # Build sequences
    X, _ = make_sequences(df, feature_cols, seq_len=seq_len, rul_cap=rul_cap)

    # Predict on a sample for demonstration (change N as you want)
    N = min(50, len(X))
    X_sample = torch.tensor(X[:N], dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(X_sample).cpu().numpy().reshape(-1)

    # Build maintenance events JSON
    events = []
    for i, pred in enumerate(preds, start=1):
        tier = risk_tier(float(pred))
        action = "inspect" if tier != "low" else "monitor"

        events.append(
            {
                "asset_id": f"ENGINE_{i}",
                "predicted_rul_cycles": float(pred),
                "risk_tier": tier,
                "recommended_action": action,
                "generated_utc": datetime.utcnow().isoformat(),
            }
        )

    out_path = outputs_dir / "maintenance_events.json"
    out_path.write_text(json.dumps(events, indent=2))

    print("Using device:", device)
    print(f"Generated {len(events)} maintenance events -> {out_path}")


if __name__ == "__main__":
    main()
