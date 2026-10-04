import pandas as pd


DEFAULT_THRESHOLD = 0.40


def classify_score(score, threshold=DEFAULT_THRESHOLD):
    """
    Classify an evaluation score.

    These categories are intended for analysis,
    not as formal pass/fail certification limits.
    """

    if score < threshold:
        return "Needs Investigation"

    return "Acceptable"


def analyze_failures(
    results_df,
    thresholds=None,
):
    """
    Identify low-scoring samples.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Sample-level evaluation results.

    thresholds : dict, optional
        Metric-specific investigation thresholds.

    Returns
    -------
    pandas.DataFrame
        Failure analysis results.
    """

    if thresholds is None:
        thresholds = {
            "answer_relevancy": 0.40,
            "answer_correctness": 0.40,
            "context_answer_similarity": 0.40,
        }

    analysis_rows = []

    score_columns = [
        column
        for column in results_df.columns
        if column in thresholds
    ]

    for _, row in results_df.iterrows():

        sample_id = row["id"]

        failed_metrics = []
        scores = {}

        for metric in score_columns:

            score = float(row[metric])

            scores[metric] = score

            if score < thresholds[metric]:
                failed_metrics.append(metric)

        if failed_metrics:

            severity = determine_severity(
                len(failed_metrics),
                len(score_columns),
            )

            analysis_rows.append(
                {
                    "id": sample_id,
                    "failed_metrics": ", ".join(
                        failed_metrics
                    ),
                    "failure_count": len(
                        failed_metrics
                    ),
                    "severity": severity,
                    **scores,
                }
            )

    return pd.DataFrame(analysis_rows)


def determine_severity(
    failure_count,
    total_metrics,
):
    """
    Assign a simple investigation severity.

    This is an analytical classification and
    should not be interpreted as a security rating.
    """

    if total_metrics == 0:
        return "Unknown"

    failure_ratio = (
        failure_count / total_metrics
    )

    if failure_ratio >= 0.75:
        return "High"

    if failure_ratio >= 0.50:
        return "Medium"

    return "Low"


def summarize_failures(failure_df):
    """
    Generate summary statistics for failures.
    """

    if failure_df.empty:

        return {
            "total_failures": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

    return {
        "total_failures": len(failure_df),
        "high": int(
            (
                failure_df["severity"]
                == "High"
            ).sum()
        ),
        "medium": int(
            (
                failure_df["severity"]
                == "Medium"
            ).sum()
        ),
        "low": int(
            (
                failure_df["severity"]
                == "Low"
            ).sum()
        ),
    }


if __name__ == "__main__":

    print(
        "========== FAILURE ANALYSIS TEST =========="
    )

    input_file = (
        "outputs/results/"
        "unified_evaluation_results.csv"
    )

    df = pd.read_csv(input_file)

    failures = analyze_failures(df)

    summary = summarize_failures(
        failures
    )

    print("\nFailure Summary:")

    print(
        f"Total failures: "
        f"{summary['total_failures']}"
    )

    print(
        f"High: {summary['high']}"
    )

    print(
        f"Medium: {summary['medium']}"
    )

    print(
        f"Low: {summary['low']}"
    )

    print(
        "\n========== FAILED SAMPLES =========="
    )

    if failures.empty:

        print(
            "No samples require investigation."
        )

    else:

        print(
            failures.to_string(
                index=False
            )
        )

    print(
        "\n========== TEST COMPLETE =========="
    )