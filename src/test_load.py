import os
from preprocess import load_data, add_rul

# Get project root dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "train_FD001.txt")

df = load_data(DATA_PATH)
df = add_rul(df)

print(df.head())
