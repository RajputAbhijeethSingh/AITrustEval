import pandas as pd


# ============================================================
# FIELD ALIASES
# ============================================================

FIELD_ALIASES = {

    "question": [
        "question",
        "questions",
        "query",
        "queries",
        "user_query",
        "user_question",
        "prompt",
        "input",
        "user_input"
    ],

    "context": [
        "context",
        "contexts",
        "retrieved_context",
        "retrieved_contexts",
        "retrieved_contexts",
        "documents",
        "document",
        "retrieval_context",
        "retrieval_contexts"
    ],

    "answer": [
        "answer",
        "answers",
        "response",
        "responses",
        "generated_answer",
        "model_answer",
        "assistant_answer",
        "output"
    ],

    "ground_truth": [
        "ground_truth",
        "ground_truths",
        "groundtruth",
        "groundtruths",
        "reference",
        "references",
        "reference_answer",
        "reference_answers",
        "expected_answer",
        "expected_answers",
        "correct_answer"
    ],

    "metadata": [
        "metadata",
        "meta_data",
        "meta",
        "information",
        "additional_metadata"
    ],

    # Optional fields.
    # These are NOT required for the DRDO dataset.
    "latency": [
        "latency",
        "latency_ms",
        "response_time",
        "response_time_ms",
        "response_latency",
        "response_latency_ms",
        "duration_ms"
    ],

    "status": [
        "status",
        "response_status",
        "request_status",
        "success",
        "is_success",
        "error",
        "failed"
    ]
}


# ============================================================
# NORMALIZE COLUMN NAME
# ============================================================

def normalize_column_name(column_name):

    return (
        str(column_name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


# ============================================================
# DETECT COLUMNS
# ============================================================

def detect_columns(df):

    if not isinstance(df, pd.DataFrame):

        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    detected_columns = {}

    normalized_columns = {}

    for column in df.columns:

        normalized_name = normalize_column_name(
            column
        )

        normalized_columns[
            normalized_name
        ] = column

    # --------------------------------------------------------
    # Match logical fields
    # --------------------------------------------------------

    for field, aliases in FIELD_ALIASES.items():

        for alias in aliases:

            normalized_alias = (
                normalize_column_name(alias)
            )

            if normalized_alias in normalized_columns:

                detected_columns[field] = (
                    normalized_columns[
                        normalized_alias
                    ]
                )

                break

    return detected_columns


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(df):

    if not isinstance(df, pd.DataFrame):

        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    detected_columns = detect_columns(df)

    warnings = []
    errors = []

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    required_fields = [
        "question",
        "context",
        "answer",
        "ground_truth"
    ]

    missing_required = [
        field
        for field in required_fields
        if field not in detected_columns
    ]

    if missing_required:

        errors.append(
            "Missing required fields: "
            + ", ".join(missing_required)
        )

    # --------------------------------------------------------
    # Metadata is expected by the DRDO dataset
    # but we do not make it mandatory because some
    # evaluation datasets may not contain it.
    # --------------------------------------------------------

    if "metadata" not in detected_columns:

        warnings.append(
            "Metadata field was not detected. "
            "Metrics that depend on metadata will be unavailable."
        )

    # --------------------------------------------------------
    # Optional fields
    # --------------------------------------------------------

    if "latency" not in detected_columns:

        warnings.append(
            "Latency field was not detected. "
            "Latency metrics cannot be calculated directly "
            "unless latency information exists inside metadata."
        )

    if "status" not in detected_columns:

        warnings.append(
            "Status field was not detected. "
            "Success/failure metrics may be unavailable."
        )

    return {

        "valid": len(errors) == 0,

        "detected_columns":
            detected_columns,

        "missing_required":
            missing_required,

        "warnings":
            warnings,

        "errors":
            errors

    }


# ============================================================
# HUMAN-READABLE SUMMARY
# ============================================================

def print_schema_summary(validation_result):

    print("\n========== AITrustEval Schema ==========\n")

    detected_columns = validation_result.get(
        "detected_columns",
        {}
    )

    if detected_columns:

        print("Detected fields:")

        for field, column in detected_columns.items():

            print(
                f"  {field:15} -> {column}"
            )

    else:

        print(
            "No compatible fields detected."
        )

    print()

    missing_required = validation_result.get(
        "missing_required",
        []
    )

    if missing_required:

        print("Missing required fields:")

        for field in missing_required:

            print(
                f"  - {field}"
            )

    else:

        print(
            "All required fields detected."
        )

    print()

    warnings = validation_result.get(
        "warnings",
        []
    )

    if warnings:

        print("Warnings:")

        for warning in warnings:

            print(
                f"  - {warning}"
            )

    print()

    errors = validation_result.get(
        "errors",
        []
    )

    if errors:

        print("Errors:")

        for error in errors:

            print(
                f"  - {error}"
            )

    print()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_df = pd.DataFrame({

        "question": [
            "What is TCP?"
        ],

        "context": [
            "TCP is a reliable transport protocol."
        ],

        "answer": [
            "TCP is a reliable transport protocol."
        ],

        "metadata": [
            '{"source": "test", "model": "mistral"}'
        ],

        "ground_truth": [
            "TCP is a reliable transport layer protocol."
        ]

    })

    result = validate_dataset(
        test_df
    )

    print_schema_summary(
        result
    )