import pandas as pd

file_path = "data/sample/ragas_eval_30.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET INFORMATION ==========")
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== FIRST SAMPLE ==========")
for column in df.columns:
    print(f"\n{column}:")
    print(df.iloc[0][column])

print("\n========== CONTEXT COUNTS ==========")
print(
    df["contexts"].apply(
        lambda x: len(eval(x)) if isinstance(x, str) else 0
    ).value_counts().sort_index()
)

print("\n========== DATASET READY ==========")
