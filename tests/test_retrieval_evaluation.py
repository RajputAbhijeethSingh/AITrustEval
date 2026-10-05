import pandas as pd

from evaluation.retrieval.evaluator import (
    evaluate_semantic_retrieval,
)


DATASET_PATH = "data/sample/synthetic_5field_dataset.csv"


def main():

    print("=" * 65)
    print("AITrustEval - Semantic Retrieval Evaluation Test")
    print("=" * 65)

    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    print("\nDataset:")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")

    print("\nRunning semantic retrieval evaluation...")

    results = evaluate_semantic_retrieval(
        df,
        k=5,
        threshold=0.60,
    )

    summary = results["summary"]

    print("\n========== SUMMARY ==========")

    for key, value in summary.items():

        print(
            f"{key}: {value}"
        )

    print("\n========== ROW STATUS ==========")

    for row in results["rows"][:5]:

        print(
            f"\nRow {row['row_index']}"
        )

        print(
            f"Status: {row['status']}"
        )

        print(
            f"Context count: "
            f"{row['context_count']}"
        )

        print(
            f"Ranked context: "
            f"{row['ranked_context']}"
        )

        print(
            f"Message: "
            f"{row['message']}"
        )

    print("\n" + "=" * 65)
    print("TEST COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()