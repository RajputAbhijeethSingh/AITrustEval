import pandas as pd


# Possible names that datasets may use for each field.
COLUMN_ALIASES = {
    "question": [
        "question",
        "query",
        "user_query",
        "user_question",
        "prompt",
        "input",
    ],
    "answer": [
        "answer",
        "response",
        "generated_answer",
        "model_answer",
        "output",
    ],
    "ground_truth": [
    "ground_truth",
    "ground_truths",
    "groundtruth",
    "groundtruths",
    "reference",
    "references",
    "reference_answer",
    "expected_answer",
    "correct_answer",
  ],
    "context": [
        "context",
        "contexts",
        "retrieved_context",
        "retrieved_contexts",
        "documents",
        "document",
    ],
    "latency": [
        "latency",
        "response_time",
        "response_time_ms",
        "latency_ms",
    ],
}


def normalize_column_name(name):
    """Convert a column name into a comparable format."""
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def detect_columns(df):
    """
    Automatically detect standard AI evaluation fields
    from dataset column names.
    """

    detected = {}

    normalized_columns = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for standard_field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized_columns:
                detected[standard_field] = normalized_columns[alias]
                break

    return detected


def validate_dataset(df):
    """
    Validate the uploaded dataset and determine
    which evaluation capabilities are available.
    """

    errors = []
    warnings = []

    if df.empty:
        errors.append("Dataset is empty.")

    if len(df.columns) == 0:
        errors.append("Dataset contains no columns.")

    detected = detect_columns(df)

    # Question is required for most answer-quality evaluation.
    if "question" not in detected:
        warnings.append(
            "Question/query column was not detected."
        )

    # Answer is required for answer-quality evaluation.
    if "answer" not in detected:
        warnings.append(
            "Answer/response column was not detected."
        )

    # Context and ground truth are needed for several RAG metrics.
    if "context" not in detected:
        warnings.append(
            "Context/document column was not detected."
        )

    if "ground_truth" not in detected:
        warnings.append(
            "Ground-truth/reference column was not detected."
        )

    capabilities = {
        "answer_relevancy": (
            "question" in detected and
            "answer" in detected
        ),
        "answer_correctness": (
            "answer" in detected and
            "ground_truth" in detected
        ),
        "faithfulness": (
            "answer" in detected and
            "context" in detected
        ),
        "context_metrics": (
            "context" in detected and
            "ground_truth" in detected
        ),
        "latency": (
            "latency" in detected
        ),
    }

    # Retrieval metrics require ranked retrieval information.
    retrieval_metrics_available = False

    if "context" in detected:
        context_column = detected["context"]

        # A context field by itself does not guarantee that
        # ranked retrieval evaluation is possible.
        sample_value = df[context_column].dropna()

        if not sample_value.empty:
            first_value = sample_value.iloc[0]

            if isinstance(first_value, (list, tuple)):
                retrieval_metrics_available = True

    capabilities.update({
        "precision_at_k": retrieval_metrics_available,
        "recall_at_k": retrieval_metrics_available,
        "mrr": retrieval_metrics_available,
        "ndcg": retrieval_metrics_available,
        "hit_rate": retrieval_metrics_available,
    })

    return {
        "valid": len(errors) == 0,
        "rows": len(df),
        "columns": list(df.columns),
        "detected_columns": detected,
        "capabilities": capabilities,
        "errors": errors,
        "warnings": warnings,
    }


if __name__ == "__main__":

    print("========== DATASET SCHEMA DETECTOR ==========")

    input_file = "data/sample/ragas_eval_30.csv"

    df = pd.read_csv(input_file)

    result = validate_dataset(df)

    print("\nDataset:")
    print(f"Rows: {result['rows']}")
    print(f"Columns: {result['columns']}")

    print("\nDetected Columns:")
    for field, column in result["detected_columns"].items():
        print(f"  {field} -> {column}")

    print("\nMetric Capabilities:")
    for metric, available in result["capabilities"].items():
        status = "AVAILABLE" if available else "NOT AVAILABLE"
        print(f"  {metric}: {status}")

    print("\nWarnings:")
    for warning in result["warnings"]:
        print(f"  - {warning}")

    print("\nErrors:")
    for error in result["errors"]:
        print(f"  - {error}")

    print("\n========== TEST COMPLETE ==========")