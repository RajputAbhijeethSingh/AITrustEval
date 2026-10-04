import ast
import pandas as pd


REQUIRED_COLUMNS = [
    "question",
    "ground_truths",
    "answer",
    "contexts",
]


def load_dataset(file_path):
    """Load a RAG evaluation dataset."""

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)

    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)

    elif file_path.endswith(".json"):
        df = pd.read_json(file_path)

    else:
        raise ValueError(
            "Unsupported file format. Use CSV, XLSX, or JSON."
        )

    return df


def validate_dataset(df):
    """Validate the basic structure of a RAG dataset."""

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        return False, f"Missing columns: {missing_columns}"

    if df.empty:
        return False, "Dataset is empty."

    missing_values = df[REQUIRED_COLUMNS].isnull().sum()

    if missing_values.any():
        return False, f"Missing values found: {missing_values.to_dict()}"

    return True, "Dataset is valid."


def parse_list(value):
    """Convert a list stored as text into a Python list safely."""

    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return parsed

        return [str(parsed)]

    except (ValueError, SyntaxError):
        return [str(value)]


def normalize_dataset(df):
    """Convert the original RAGAS format into AITrustEval format."""

    normalized = pd.DataFrame()

    normalized["id"] = range(1, len(df) + 1)

    normalized["question"] = df["question"].astype(str)

    normalized["context"] = df["contexts"].apply(parse_list)

    normalized["answer"] = df["answer"].astype(str)

    normalized["ground_truth"] = df["ground_truths"].apply(parse_list)

    normalized["metadata"] = "source=ragas-eval"

    return normalized


if __name__ == "__main__":

    file_path = "data/sample/ragas_eval_30.csv"

    print("\n========== AITrustEval DATA LOADER ==========")

    df = load_dataset(file_path)

    print(f"Original rows: {len(df)}")

    valid, message = validate_dataset(df)

    print(f"Validation: {message}")

    if valid:

        normalized_df = normalize_dataset(df)

        print("\n========== NORMALIZED DATASET ==========")

        print(f"Rows: {len(normalized_df)}")
        print(f"Columns: {list(normalized_df.columns)}")

        print("\nFirst question:")
        print(normalized_df.iloc[0]["question"])

        print("\nNumber of contexts:")
        print(len(normalized_df.iloc[0]["context"]))

        print("\nNumber of ground truths:")
        print(len(normalized_df.iloc[0]["ground_truth"]))

        output_file = "data/processed/normalized_ragas_eval.csv"

        # Store list fields as strings so they can be written to CSV.
        save_df = normalized_df.copy()
        save_df["context"] = save_df["context"].apply(str)
        save_df["ground_truth"] = save_df["ground_truth"].apply(str)

        save_df.to_csv(output_file, index=False)

        print(f"\nNormalized dataset saved to:")
        print(output_file)

        print("\n========== SUCCESS ==========")
