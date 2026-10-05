import pandas as pd

from evaluation.retrieval.evaluator import (
    parse_context,
    parse_ground_truth,
)


DATASET_PATH = "data/sample/synthetic_5field_dataset.csv"


def main():

    print("=" * 60)
    print("AITrustEval - 5 Field Dataset Test")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    print("\nDataset shape:")
    print(df.shape)

    # Show columns
    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    # Required fields
    required_columns = [
        "question",
        "context",
        "answer",
        "metadata",
        "ground_truth",
    ]

    print("\nRequired field check:")

    all_present = True

    for column in required_columns:

        if column in df.columns:
            print(f"  ✓ {column}")
        else:
            print(f"  ✗ {column}")
            all_present = False

    if not all_present:
        print("\nERROR: Required fields are missing.")
        return

    # Check context structure
    print("\nChecking context structure...")

    for index in range(min(3, len(df))):

        context = df.iloc[index]["context"]

        chunks, ranked = parse_context(context)

        print(f"\nRow {index}:")

        print(
            f"  Number of context chunks: "
            f"{len(chunks)}"
        )

        print(
            f"  Ranked context: {ranked}"
        )

        for rank, chunk in enumerate(
            chunks,
            start=1
        ):

            preview = (
                chunk[:100]
                .replace("\n", " ")
            )

            print(
                f"  Rank {rank}: "
                f"{preview}..."
            )

        # Ground truth
        ground_truth = df.iloc[index][
            "ground_truth"
        ]

        references = parse_ground_truth(
            ground_truth
        )

        print(
            f"  Ground-truth references: "
            f"{len(references)}"
        )

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()