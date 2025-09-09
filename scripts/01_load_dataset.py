from datasets import load_dataset
import pandas as pd
ds = load_dataset("greengerong/leetcode")
df = ds["train"].to_pandas()
df.to_csv("../data/raw/leetcode_raw.csv", index=False)
print("✅ Raw dataset saved:", df.shape)
print(df.head())
