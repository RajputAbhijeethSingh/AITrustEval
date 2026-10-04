from datasets import load_dataset
import pandas as pd

print("Downloading RAGAS evaluation dataset...")

dataset = load_dataset("shayorshay/ragas-eval", split="train")

df = dataset.to_pandas()

print(f"Downloaded {len(df)} samples.")
print("Columns:", list(df.columns))

df.to_csv(
    "data/sample/ragas_eval_30.csv",
    index=False
)

print("Saved to: data/sample/ragas_eval_30.csv")
