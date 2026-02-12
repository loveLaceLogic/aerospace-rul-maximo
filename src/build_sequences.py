import os
from preprocess import load_data, add_rul, scale_features, make_sequences

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "train_FD001.txt")

df = load_data(TRAIN_PATH)
df = add_rul(df)

feature_cols = ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

df, scaler = scale_features(df, feature_cols)

X, y = make_sequences(df, feature_cols, seq_len=30, rul_cap=130)

print("X shape:", X.shape)  # (samples, 30, features)
print("y shape:", y.shape)  # (samples,)
print("Example y[0]:", y[0])
