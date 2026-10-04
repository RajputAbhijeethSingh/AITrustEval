import json
from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
    ".json",
}


def load_dataset(file_path):
    """
    Load a dataset from CSV, Excel, or JSON.

    Parameters
    ----------
    file_path : str or Path
        Path to the dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {extension}. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if extension == ".csv":
        df = pd.read_csv(file_path)

    elif extension in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)

    elif extension == ".json":
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            df = pd.DataFrame(data)

        elif isinstance(data, dict):
            df = pd.DataFrame(data)

        else:
            raise ValueError(
                "JSON dataset must contain a list or dictionary."
            )

    return df


def get_dataset_info(df):
    """
    Generate basic information about a dataset.
    """

    missing_values = int(df.isna().sum().sum())

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "missing_values": missing_values,
        "duplicate_rows": int(df.duplicated().sum()),
    }


if __name__ == "__main__":

    print("========== DATASET LOADER TEST ==========")

    input_file = "data/sample/ragas_eval_30.csv"

    df = load_dataset(input_file)

    info = get_dataset_info(df)

    print("\nDataset loaded successfully.")

    print(f"Rows: {info['rows']}")
    print(f"Columns: {info['columns']}")
    print(f"Missing values: {info['missing_values']}")
    print(f"Duplicate rows: {info['duplicate_rows']}")

    print("\nColumns:")

    for column in info["column_names"]:
        print(f"  - {column}")

    print("\n========== TEST COMPLETE ==========")