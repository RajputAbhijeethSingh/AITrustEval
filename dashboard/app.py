"""
AITrustEval - Professional Streamlit Dashboard

Offline AI/ML Trustworthiness, Security & Reliability
Evaluation Platform.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AITrustEval",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# IMPORTS
# ============================================================

from evaluation.dataset.data_loader import (
    load_dataset,
    get_dataset_info,
)

from evaluation.engine.evaluator import (
    run_evaluation,
)

from evaluation.answer_quality.basic_metrics import (
    evaluate_basic_answer_quality,
)

from explainability.explainability_engine import (
    run_explainability,
)

# Baseline failure analysis is implemented locally in this dashboard
# so it can safely consume the DataFrame produced by the baseline evaluator.

from explainability.root_cause import (
    generate_root_cause_report,
)


# ============================================================
# REPORT GENERATOR
# ============================================================

REPORT_GENERATOR_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "report_generator.py"
)


_report_spec = (
    importlib.util.spec_from_file_location(
        "aitrust_eval_report_generator",
        REPORT_GENERATOR_PATH,
    )
)


_report_module = (
    importlib.util.module_from_spec(
        _report_spec
    )
)


_report_spec.loader.exec_module(
    _report_module
)


build_report_data = (
    _report_module.build_report_data
)

generate_reports = (
    _report_module.generate_reports
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def safe_value(
    value,
    default="Not Available",
):
    if value is None:
        return default

    try:

        if pd.isna(value):
            return default

    except Exception:

        pass

    return value


def format_sample_id(
    value,
):
    try:

        number = float(value)

        if number.is_integer():

            return str(
                int(number)
            )

    except Exception:

        pass

    return str(value)


def get_original_row(
    df,
    sample_id,
):

    try:

        position = (
            int(
                float(
                    sample_id
                )
            )
            - 1
        )

        if (
            0 <= position < len(df)
        ):

            return df.iloc[
                position
            ]

    except Exception:

        pass

    return None


def normalize_metric_name(
    name,
):

    return (
        str(name)
        .replace(
            "_",
            " ",
        )
        .title()
    )


def _safe_numeric_score(value):
    """
    Convert a baseline metric value to a numeric score.

    Non-numeric, missing, or invalid values are treated as unavailable.
    """
    try:
        if value is None:
            return None

        numeric_value = float(value)

        if pd.isna(numeric_value):
            return None

        return numeric_value

    except Exception:
        return None


def build_baseline_failure_analysis(
    results_df,
    threshold=0.50,
):
    """
    Build a normalized failure-analysis DataFrame directly from the
    baseline sample-level results.

    This avoids assumptions about nested dictionaries/lists inside the
    baseline result rows and prevents errors such as:
        'str' object has no attribute 'get'

    Failure metrics:
        - answer_relevancy
        - answer_correctness
        - context_answer_similarity

    Severity:
        1 failed metric -> Low
        2 failed metrics -> Medium
        3 failed metrics -> High
    """

    if results_df is None:
        return pd.DataFrame()

    if not isinstance(results_df, pd.DataFrame):
        try:
            results_df = pd.DataFrame(results_df)
        except Exception:
            return pd.DataFrame()

    if results_df.empty:
        return pd.DataFrame()

    metric_columns = [
        "answer_relevancy",
        "answer_correctness",
        "context_answer_similarity",
    ]

    available_metrics = [
        metric
        for metric in metric_columns
        if metric in results_df.columns
    ]

    if not available_metrics:
        return pd.DataFrame()

    failure_rows = []

    for position, (_, row) in enumerate(results_df.iterrows(), start=1):

        sample_id = row.get(
            "id",
            position,
        )

        failed_metrics = []

        metric_scores = {}

        for metric in available_metrics:

            score = _safe_numeric_score(
                row.get(metric)
            )

            metric_scores[metric] = score

            if (
                score is not None
                and score < threshold
            ):
                failed_metrics.append(metric)

        if not failed_metrics:
            continue

        failure_count = len(
            failed_metrics
        )

        if failure_count >= 3:
            severity = "High"
        elif failure_count == 2:
            severity = "Medium"
        else:
            severity = "Low"

        failure_rows.append(
            {
                "id": sample_id,
                "severity": severity,
                "failed_metrics": ", ".join(
                    failed_metrics
                ),
                "failure_count": failure_count,
                **metric_scores,
            }
        )

    return pd.DataFrame(
        failure_rows
    )


def summarize_baseline_failures(
    failure_df,
):
    """
    Summarize baseline failure counts by severity.
    """

    if failure_df is None:
        return {
            "total_failures": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

    if not isinstance(
        failure_df,
        pd.DataFrame,
    ):
        try:
            failure_df = pd.DataFrame(
                failure_df
            )
        except Exception:
            return {
                "total_failures": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
            }

    if failure_df.empty:
        return {
            "total_failures": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

    severity_series = (
        failure_df.get(
            "severity",
            pd.Series(
                dtype=str
            ),
        )
        .astype(str)
        .str.strip()
        .str.title()
    )

    return {
        "total_failures": int(
            len(failure_df)
        ),
        "high": int(
            (severity_series == "High").sum()
        ),
        "medium": int(
            (severity_series == "Medium").sum()
        ),
        "low": int(
            (severity_series == "Low").sum()
        ),
    }


def parse_context(
    value,
):

    if isinstance(
        value,
        list,
    ):

        return value

    if isinstance(
        value,
        dict,
    ):

        return [
            value
        ]

    text = safe_value(
        value,
        "",
    )


    try:

        parsed = json.loads(
            str(text)
        )

        if isinstance(
            parsed,
            list,
        ):

            return parsed

        if isinstance(
            parsed,
            dict,
        ):

            return [
                parsed
            ]

    except Exception:

        pass


    return [
        {
            "rank": 1,
            "context": str(text),
        }
    ]


def render_status(
    title,
    result,
):

    if not isinstance(
        result,
        dict,
    ):

        st.info(
            f"{title}: No result."
        )

        return


    status = result.get(
        "status",
        "unknown",
    )


    message = result.get(
        "message",
        "",
    )


    if status in (
        "completed",
        "success",
    ):

        st.success(
            f"**{title}: Completed**"
        )


    elif status == "unavailable":

        st.warning(
            f"**{title}: Unavailable**"
        )


    elif status == "error":

        st.error(
            f"**{title}: Error**"
        )


    else:

        st.info(
            f"**{title}: {status}**"
        )


    if message:

        st.caption(
            message
        )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ AITrustEval</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Offline AI/ML Trustworthiness, Security & Reliability
    Evaluation Platform
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    **Retrieval Quality • RAG/LLM Quality • Security • Reliability •
    Explainability • Failures • Root Causes**
    """
)


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🛡️ AITrustEval"
    )

    st.markdown(
        """
        ### Evaluation Layers

        🔎 Retrieval

        📊 Baseline Answer Quality

        🧠 RAG / LLM

        🛡️ Security

        ⚙️ Reliability

        🧩 Explainability

        🚨 Failure Analysis

        🧬 Root Cause

        📄 Reporting
        """
    )

    st.divider()

    st.caption(
        "Development: Windows"
    )

    st.caption(
        "Deployment: Offline Ubuntu / DRDO"
    )


# ============================================================
# UPLOAD
# ============================================================

st.header(
    "📂 Upload Evaluation Dataset"
)


uploaded_file = st.file_uploader(
    "Upload CSV, Excel or JSON dataset",
    type=[
        "csv",
        "xlsx",
        "xls",
        "json",
        "jsonl",
    ],
)


if uploaded_file is None:

    st.info(
        "👆 Upload an evaluation dataset to begin."
    )

    st.stop()


# ============================================================
# SAVE DATASET
# ============================================================

input_directory = (
    PROJECT_ROOT
    / "data"
    / "input"
)


input_directory.mkdir(
    parents=True,
    exist_ok=True,
)


file_extension = (
    uploaded_file.name
    .split(".")[-1]
    .lower()
)


input_path = (
    input_directory
    / f"uploaded_dataset.{file_extension}"
)


try:

    with open(
        input_path,
        "wb",
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

except Exception as error:

    st.error(
        f"Unable to save dataset: {error}"
    )

    st.stop()


# ============================================================
# LOAD
# ============================================================

try:

    df = load_dataset(
        str(input_path)
    )

except Exception as error:

    st.error(
        f"Unable to load dataset: {error}"
    )

    st.stop()


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.divider()

st.header(
    "📊 Dataset Overview"
)


dataset_info = get_dataset_info(
    df
)


columns = st.columns(
    5
)


with columns[0]:

    st.metric(
        "Rows",
        dataset_info.get(
            "rows",
            len(df),
        ),
    )


with columns[1]:

    st.metric(
        "Columns",
        dataset_info.get(
            "columns",
            len(df.columns),
        ),
    )


with columns[2]:

    st.metric(
        "Missing Values",
        dataset_info.get(
            "missing_values",
            int(
                df.isna()
                .sum()
                .sum()
            ),
        ),
    )


with columns[3]:

    st.metric(
        "Duplicate Rows",
        dataset_info.get(
            "duplicate_rows",
            int(
                df.duplicated()
                .sum()
            ),
        ),
    )


with columns[4]:

    st.metric(
        "File Type",
        file_extension.upper(),
    )


st.caption(
    f"Dataset: `{uploaded_file.name}`"
)


with st.expander(
    "👁️ Preview Dataset"
):

    st.table(df.head(10))


# ============================================================
# ANALYZE
# ============================================================

st.divider()


if "analysis_requested" not in st.session_state:
    st.session_state["analysis_requested"] = False


analyze_button = st.button(
    "🚀 Analyze Dataset",
    type="primary",
    use_container_width=True,
)


if analyze_button:
    st.session_state["analysis_requested"] = True


if not st.session_state["analysis_requested"]:

    st.info(
        "Click **Analyze Dataset** to run the complete evaluation pipeline."
    )

    st.stop()


# ============================================================
# CENTRAL EVALUATION
# ============================================================

with st.spinner(
    "Running AITrustEval evaluation..."
):

    evaluation = run_evaluation(
        df
    )


schema = evaluation.get(
    "schema",
    {},
)


# ============================================================
# BASELINE ANSWER QUALITY
# ============================================================

with st.spinner(
    "Running offline baseline answer-quality evaluation..."
):

    baseline = (
        evaluate_basic_answer_quality(
            df,
            schema,
        )
    )


# Add baseline information to
# centralized evaluation result.

evaluation[
    "baseline_answer_quality"
] = baseline


evaluation[
    "metrics"
] = evaluation.get(
    "metrics",
    {},
)


evaluation[
    "metrics"
][
    "baseline_answer_quality"
] = baseline.get(
    "averages",
    {},
)


# Update metric availability.

metric_availability = evaluation.get(
    "metric_availability",
    {},
)


baseline_available = (
    baseline.get(
        "status"
    )
    == "completed"
)


metric_availability[
    "baseline_answer_quality"
] = (
    baseline_available
)


evaluation[
    "metric_availability"
] = metric_availability


# ============================================================
# BASELINE DATAFRAME
# ============================================================

baseline_results = pd.DataFrame(
    baseline.get(
        "sample_results",
        [],
    )
)


# ============================================================
# FAILURE ANALYSIS
# ============================================================

failure_df = pd.DataFrame()


failure_summary = {}

root_cause_df = pd.DataFrame()


recommendations = []


if not baseline_results.empty:

    try:

        failure_df = build_baseline_failure_analysis(
            baseline_results
        )


        failure_summary = (
            summarize_baseline_failures(
                failure_df
            )
        )


        root_cause_df = (
            generate_root_cause_report(
                failure_df
            )
        )


        if not root_cause_df.empty:

            recommendations = (
                root_cause_df[
                    "recommendation"
                ]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )


    except Exception as error:

        st.warning(
            f"Baseline failure analysis could not be completed: {error}"
        )


# ============================================================
# EXPLAINABILITY
# ============================================================

with st.spinner(
    "Running explainability and evidence analysis..."
):

    explainability = (
        run_explainability(
            df,
            evaluation_result=evaluation,
        )
    )


# ============================================================
# EVALUATION STATUS
# ============================================================

st.divider()

st.header(
    "📌 Evaluation Status"
)


st.success(
    "Evaluation pipeline completed."
)


status_columns = st.columns(
    6
)


with status_columns[0]:

    render_status(
        "🔎 Retrieval",
        evaluation.get(
            "retrieval",
            {},
        ),
    )


with status_columns[1]:

    render_status(
        "📊 Baseline",
        baseline,
    )


with status_columns[2]:

    render_status(
        "🧠 RAGAS",
        evaluation.get(
            "ragas",
            {},
        ),
    )


with status_columns[3]:

    render_status(
        "🛡️ Security",
        evaluation.get(
            "security",
            {},
        ),
    )


with status_columns[4]:

    render_status(
        "⚙️ Reliability",
        evaluation.get(
            "reliability",
            {},
        ),
    )


with status_columns[5]:

    render_status(
        "🧩 Explainability",
        explainability,
    )


# ============================================================
# SCHEMA
# ============================================================

st.divider()

st.header(
    "🔍 Schema Detection"
)


schema_rows = []


for field, column in (
    schema.items()
):

    schema_rows.append(
        {
            "AITrustEval Field":
                field,

            "Detected Dataset Column":
                (
                    column
                    if column
                    else "NOT DETECTED"
                ),
        }
    )


if schema_rows:

    st.table(pd.DataFrame(schema_rows))


# ============================================================
# VALIDATION
# ============================================================

validation = evaluation.get(
    "validation",
    {},
)


st.subheader(
    "Dataset Validation"
)


if validation.get(
    "valid",
    False,
):

    st.success(
        validation.get(
            "message",
            "Dataset validation completed.",
        )
    )

else:

    st.warning(
        validation.get(
            "message",
            "Dataset validation did not pass.",
        )
    )


for warning in validation.get(
    "warnings",
    [],
):

    st.warning(
        warning
    )


# ============================================================
# METRIC AVAILABILITY
# ============================================================

st.divider()

st.header(
    "📋 Metric Availability"
)


availability = evaluation.get(
    "metric_availability",
    {},
)


availability_rows = []


for metric, value in (
    availability.items()
):

    availability_rows.append(
        {
            "Metric":
                normalize_metric_name(
                    metric
                ),

            "Status":
                (
                    "✅ Available"
                    if bool(value)
                    else "❌ Not Available"
                ),
        }
    )


if availability_rows:

    st.table(pd.DataFrame(availability_rows))


# ============================================================
# BASELINE ANSWER QUALITY
# ============================================================

st.divider()

st.header(
    "📊 Baseline Answer-Quality Evaluation"
)


st.caption(
    """
    These are lightweight offline lexical/reference-similarity
    measurements. They are useful for baseline analysis but are
    not equivalent to RAGAS or LLM-as-a-judge evaluation.
    """
)


baseline_averages = baseline.get(
    "averages",
    {},
)


if baseline_averages:

    baseline_columns = st.columns(
        len(
            baseline_averages
        )
    )


    for index, (
        metric,
        value,
    ) in enumerate(
        baseline_averages.items()
    ):

        with baseline_columns[
            index
        ]:

            st.metric(
                normalize_metric_name(
                    metric
                ),
                f"{float(value):.4f}",
            )


    chart_df = pd.DataFrame(
        [
            {
                "Metric":
                    normalize_metric_name(
                        metric
                    ),

                "Score":
                    float(value),
            }

            for metric, value
            in baseline_averages.items()
        ]
    )


    chart = px.bar(
        chart_df,
        x="Metric",
        y="Score",
        title="Average Baseline Evaluation Scores",
        text="Score",
    )


    chart.update_yaxes(
        range=[
            0,
            1,
        ]
    )


    chart.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )


    st.plotly_chart(
        chart,
        use_container_width=True,
    )


    st.subheader(
        "Evaluation Scores by Sample"
    )


    score_columns = [
        "answer_relevancy",
        "answer_correctness",
        "context_answer_similarity",
    ]


    existing_score_columns = [
        column
        for column in score_columns
        if column in baseline_results.columns
    ]


    if existing_score_columns:

        plot_df = baseline_results[
            [
                "id"
            ]
            + existing_score_columns
        ].copy()


        plot_df = plot_df.rename(
            columns={
                column:
                    normalize_metric_name(
                        column
                    )

                for column
                in existing_score_columns
            }
        )


        plot_df = plot_df.melt(
            id_vars=[
                "id"
            ],
            var_name="Metric",
            value_name="Score",
        )


        sample_chart = px.line(
            plot_df,
            x="id",
            y="Score",
            color="Metric",
            markers=True,
            title="Baseline Scores by Sample",
        )


        sample_chart.update_yaxes(
            range=[
                0,
                1,
            ]
        )


        st.plotly_chart(
            sample_chart,
            use_container_width=True,
        )


    with st.expander(
        "📄 View Sample-Level Baseline Results"
    ):

        st.table(baseline_results)


    baseline_csv = (
        baseline_results
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )


    st.download_button(
        label="⬇️ Download Baseline Results",
        data=baseline_csv,
        file_name="baseline_evaluation_results.csv",
        mime="text/csv",
    )


else:

    st.warning(
        baseline.get(
            "message",
            "Baseline metrics are unavailable.",
        )
    )


# ============================================================
# FAILURE ANALYSIS
# ============================================================

st.divider()

st.header(
    "🚨 Failure Analysis"
)


if not failure_df.empty:

    failure_columns = st.columns(
        4
    )


    with failure_columns[0]:

        st.metric(
            "Total Failures",
            failure_summary.get(
                "total_failures",
                0,
            ),
        )


    with failure_columns[1]:

        st.metric(
            "High Severity",
            failure_summary.get(
                "high",
                0,
            ),
        )


    with failure_columns[2]:

        st.metric(
            "Medium Severity",
            failure_summary.get(
                "medium",
                0,
            ),
        )


    with failure_columns[3]:

        st.metric(
            "Low Severity",
            failure_summary.get(
                "low",
                0,
            ),
        )


    st.subheader(
        "Failed Samples"
    )


    display_columns = [
        "id",
        "severity",
        "failed_metrics",
        "failure_count",
    ]


    available_columns = [
        column
        for column
        in display_columns
        if column in failure_df.columns
    ]


    st.table(failure_df[available_columns])


    # --------------------------------------------------------
    # SEVERITY CHART
    # --------------------------------------------------------

    severity_data = pd.DataFrame(
        {
            "Severity": [
                "High",
                "Medium",
                "Low",
            ],

            "Samples": [
                failure_summary.get(
                    "high",
                    0,
                ),

                failure_summary.get(
                    "medium",
                    0,
                ),

                failure_summary.get(
                    "low",
                    0,
                ),
            ],
        }
    )


    severity_chart = px.bar(
        severity_data,
        x="Severity",
        y="Samples",
        title="Failure Severity Distribution",
        text="Samples",
    )


    st.plotly_chart(
        severity_chart,
        use_container_width=True,
    )


else:

    st.info(
        "No baseline failures require investigation."
    )


# ============================================================
# ROOT CAUSE
# ============================================================

st.divider()

st.header(
    "🧬 Root-Cause Analysis"
)


if not root_cause_df.empty:

    root_display_columns = [
        "id",
        "failed_metrics",
        "failure_count",
        "severity",
        "root_cause",
        "recommendation",
    ]


    available_root_columns = [
        column
        for column
        in root_display_columns
        if column in root_cause_df.columns
    ]


    st.table(root_cause_df[available_root_columns])


    root_cause_counts = (
        root_cause_df[
            "root_cause"
        ]
        .value_counts()
        .reset_index()
    )


    root_cause_counts.columns = [
        "Root Cause",
        "Occurrences",
    ]


    root_chart = px.bar(
        root_cause_counts,
        x="Root Cause",
        y="Occurrences",
        title="Root-Cause Distribution",
        text="Occurrences",
    )


    root_chart.update_layout(
        xaxis_tickangle=-30
    )


    st.plotly_chart(
        root_chart,
        use_container_width=True,
    )


    if recommendations:

        st.subheader(
            "💡 Recommendations"
        )


        for recommendation in (
            recommendations
        ):

            st.info(
                recommendation
            )


else:

    st.info(
        "No root-cause findings are currently available."
    )


# ============================================================
# EXPLAINABILITY
# ============================================================

st.divider()

st.header(
    "🧩 Explainability"
)


render_status(
    "Explainability",
    explainability,
)


if baseline_available:

    st.info(
        "Baseline metric-based failure analysis and sample-level evidence analysis are available. "
        "RAGAS/LLM-based aggregate evaluation remains unavailable until the local Ollama/Mistral environment is available."
    )


st.metric(
    "Samples Analyzed",
    explainability.get(
        "samples_analyzed",
        0,
    ),
)


# ============================================================
# OVERALL BASELINE QUALITY
# ============================================================

st.subheader(
    "Overall Baseline Quality"
)


if baseline_averages:

    numeric_scores = [
        float(value)
        for value in
        baseline_averages.values()
        if value is not None
    ]


    if numeric_scores:

        overall_score = sum(
            numeric_scores
        ) / len(
            numeric_scores
        )


        if overall_score >= 0.75:

            classification = "Good"

        elif overall_score >= 0.50:

            classification = "Warning"

        else:

            classification = "Poor"


        quality_columns = (
            st.columns(3)
        )


        with quality_columns[0]:

            st.metric(
                "Score",
                f"{overall_score:.4f}",
            )


        with quality_columns[1]:

            st.metric(
                "Classification",
                classification,
            )


        with quality_columns[2]:

            st.metric(
                "Metrics Analyzed",
                len(
                    numeric_scores
                ),
            )


else:

    st.info(
        "No baseline aggregate scores are available."
    )


def _is_missing_text(value):

    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except Exception:
        pass

    return not str(value).strip()


BASELINE_ROOT_CAUSE_FALLBACKS = {

    "answer_relevancy": (
        "The generated answer may not adequately address the user's question.",
        "Improve question understanding and constrain answer generation to the requested information.",
    ),

    "answer_correctness": (
        "The generated answer may not sufficiently match the expected reference answer.",
        "Improve answer generation and verify generated claims against the expected answer and supporting evidence.",
    ),

    "context_answer_similarity": (
        "The generated answer may have weak alignment with the retrieved supporting context.",
        "Improve retrieval relevance and context coverage, then verify answer claims against retrieved evidence.",
    ),
}


def get_sample_root_cause(
    selected_id,
    selected_result,
    root_cause_df,
    failure_df,
):
    """Return robust sample-level root cause and recommendation details."""

    selected_root = pd.DataFrame()

    if (
        isinstance(root_cause_df, pd.DataFrame)
        and not root_cause_df.empty
        and "id" in root_cause_df.columns
    ):

        selected_root = root_cause_df[
            root_cause_df["id"].astype(str) == str(selected_id)
        ]

    if not selected_root.empty:

        row = selected_root.iloc[0]

        root_cause = row.get("root_cause")
        recommendation = row.get("recommendation")
        failed_metrics = row.get("failed_metrics")
        severity = row.get("severity", "Unknown")

        if (
            not _is_missing_text(root_cause)
            and not _is_missing_text(recommendation)
        ):
            return {
                "severity": severity,
                "failed_metrics": failed_metrics,
                "root_cause": str(root_cause),
                "recommendation": str(recommendation),
            }

    failed_metrics = []

    if (
        isinstance(failure_df, pd.DataFrame)
        and not failure_df.empty
        and "id" in failure_df.columns
    ):

        matching_failure = failure_df[
            failure_df["id"].astype(str) == str(selected_id)
        ]

        if not matching_failure.empty:
            failed_value = matching_failure.iloc[0].get(
                "failed_metrics",
                "",
            )

            if isinstance(failed_value, (list, tuple, set)):
                failed_metrics = [
                    str(metric).strip()
                    for metric in failed_value
                    if str(metric).strip()
                ]
            else:
                failed_metrics = [
                    metric.strip()
                    for metric in str(failed_value).split(",")
                    if metric.strip()
                ]

    if not failed_metrics:

        for metric in BASELINE_ROOT_CAUSE_FALLBACKS:

            try:
                value = float(selected_result.get(metric))
            except Exception:
                continue

            if value < 0.50:
                failed_metrics.append(metric)

    root_causes = []
    recommendations_local = []

    for metric in failed_metrics:

        if metric in BASELINE_ROOT_CAUSE_FALLBACKS:
            cause, recommendation = BASELINE_ROOT_CAUSE_FALLBACKS[metric]
            root_causes.append(cause)
            recommendations_local.append(recommendation)

    failure_count = len(failed_metrics)

    if failure_count >= 3:
        severity = "High"
    elif failure_count == 2:
        severity = "Medium"
    elif failure_count == 1:
        severity = "Low"
    else:
        severity = "Unknown"

    if not root_causes:
        root_causes = [
            "A low baseline metric was detected, but a specific root-cause mapping is not available for this metric."
        ]

    if not recommendations_local:
        recommendations_local = [
            "Review the failed metric together with the question, answer, ground truth and retrieved context."
        ]

    return {
        "severity": severity,
        "failed_metrics": ", ".join(failed_metrics) if failed_metrics else "Not available",
        "root_cause": " | ".join(dict.fromkeys(root_causes)),
        "recommendation": " | ".join(dict.fromkeys(recommendations_local)),
    }


# ============================================================
# SAMPLE INVESTIGATION
# ============================================================

st.divider()

st.header(
    "🔎 Sample Investigation"
)


if not baseline_results.empty:

    sample_ids = (
        baseline_results[
            "id"
        ]
        .tolist()
    )


    selected_id = st.selectbox(
        "Select sample",
        sample_ids,
        format_func=format_sample_id,
    )


    selected_result = (
        baseline_results[
            baseline_results[
                "id"
            ]
            == selected_id
        ]
    )


    if not selected_result.empty:

        selected_result = (
            selected_result.iloc[0]
        )


        score_columns = [
            column
            for column
            in [
                "answer_relevancy",
                "answer_correctness",
                "context_answer_similarity",
            ]
            if column
            in selected_result.index
        ]


        if score_columns:

            score_cards = st.columns(
                len(
                    score_columns
                )
            )


            for index, metric in enumerate(
                score_columns
            ):

                with score_cards[
                    index
                ]:

                    value = (
                        selected_result[
                            metric
                        ]
                    )


                    st.metric(
                        normalize_metric_name(
                            metric
                        ),
                        f"{float(value):.4f}",
                    )


        # ----------------------------------------------------
        # FAILURE INFORMATION
        # ----------------------------------------------------

        sample_root_cause = get_sample_root_cause(
            selected_id=selected_id,
            selected_result=selected_result,
            root_cause_df=root_cause_df,
            failure_df=failure_df,
        )


        severity = sample_root_cause.get(
            "severity",
            "Unknown",
        )


        if severity == "High":

            st.error(
                f"🚨 High Severity — Sample "
                f"{format_sample_id(selected_id)}"
            )

        elif severity == "Medium":

            st.warning(
                f"⚠️ Medium Severity — Sample "
                f"{format_sample_id(selected_id)}"
            )

        elif severity == "Low":

            st.info(
                f"ℹ️ Low Severity — Sample "
                f"{format_sample_id(selected_id)}"
            )

        else:

            st.info(
                f"ℹ️ Sample {format_sample_id(selected_id)}"
            )


        st.markdown(
            "### ❌ Failed Metrics"
        )


        st.write(
            sample_root_cause.get(
                "failed_metrics",
                "Not available",
            )
        )


        st.markdown(
            "### 🧬 Possible Root Cause"
        )


        st.warning(
            sample_root_cause.get(
                "root_cause",
                "Root cause unavailable.",
            )
        )


        st.markdown(
            "### 💡 Recommendation"
        )


        st.info(
            sample_root_cause.get(
                "recommendation",
                "No recommendation available.",
            )
        )


        # ----------------------------------------------------
        # ORIGINAL DATA
        # ----------------------------------------------------

        original_row = (
            get_original_row(
                df,
                selected_id,
            )
        )


        if original_row is not None:

            st.markdown(
                "### 📚 Original Dataset Evidence"
            )


            question_column = (
                schema.get(
                    "question"
                )
            )


            answer_column = (
                schema.get(
                    "answer"
                )
            )


            ground_truth_column = (
                schema.get(
                    "ground_truth"
                )
            )


            context_column = (
                schema.get(
                    "context"
                )
            )


            metadata_column = (
                schema.get(
                    "metadata"
                )
            )


            with st.expander(
                "❓ View Question"
            ):

                st.write(
                    safe_value(
                        original_row.get(
                            question_column
                        )
                    )
                )


            with st.expander(
                "🤖 View Generated Answer"
            ):

                st.write(
                    safe_value(
                        original_row.get(
                            answer_column
                        )
                    )
                )


            with st.expander(
                "🎯 View Ground Truth"
            ):

                st.write(
                    safe_value(
                        original_row.get(
                            ground_truth_column
                        )
                    )
                )


            with st.expander(
                "📄 View Retrieved Context"
            ):

                if context_column:

                    context_chunks = (
                        parse_context(
                            original_row.get(
                                context_column
                            )
                        )
                    )


                    for chunk in (
                        context_chunks
                    ):

                        if isinstance(
                            chunk,
                            dict,
                        ):

                            rank = (
                                chunk.get(
                                    "rank",
                                    "N/A",
                                )
                            )


                            text = (
                                chunk.get(
                                    "context",
                                    chunk.get(
                                        "text",
                                        "",
                                    ),
                                )
                            )


                            st.markdown(
                                f"**Rank {rank}**"
                            )


                            st.write(
                                text
                            )


                            st.divider()

                        else:

                            st.write(
                                chunk
                            )


            if metadata_column:

                with st.expander(
                    "🗂️ View Metadata"
                ):

                    st.write(
                        safe_value(
                            original_row.get(
                                metadata_column
                            )
                        )
                    )


# ============================================================
# RAGAS
# ============================================================

st.divider()

st.header(
    "🧠 RAG / LLM Evaluation"
)


ragas = evaluation.get(
    "ragas",
    {},
)


render_status(
    "RAGAS",
    ragas,
)


if ragas.get(
    "status"
) == "unavailable":

    st.info(
        """
        RAGAS is unavailable in the current Windows
        development environment because Ollama/Mistral
        is not available.

        The approved DRDO offline environment can run
        the same evaluation pipeline using the local
        evaluator model.
        """
    )


# ============================================================
# RETRIEVAL
# ============================================================

st.divider()

st.header(
    "🔎 Retrieval Quality"
)


retrieval = evaluation.get(
    "retrieval",
    {},
)


render_status(
    "Retrieval",
    retrieval,
)


retrieval_summary = retrieval.get(
    "summary",
    {},
)


retrieval_metrics = [
    "precision@5",
    "recall@5",
    "mrr",
    "ndcg@5",
    "hit_rate@5",
]


retrieval_cards = st.columns(
    5
)


for index, metric in enumerate(
    retrieval_metrics
):

    with retrieval_cards[
        index
    ]:

        value = (
            retrieval_summary.get(
                metric
            )
        )


        if value is None:

            st.metric(
                metric.upper(),
                "N/A",
            )

        else:

            st.metric(
                metric.upper(),
                f"{float(value):.4f}",
            )


# ============================================================
# SECURITY
# ============================================================

st.divider()

st.header(
    "🛡️ Security & Safety"
)


security = evaluation.get(
    "security",
    {},
)


render_status(
    "Security",
    security,
)


security_summary = security.get(
    "summary",
    {},
)


if security_summary:

    security_cards = st.columns(
        4
    )


    security_values = [
        (
            "Total Samples",
            "total_samples",
        ),

        (
            "High Risk",
            "high_risk",
        ),

        (
            "Medium Risk",
            "medium_risk",
        ),

        (
            "Low Risk",
            "low_risk",
        ),
    ]


    for index, (
        label,
        key,
    ) in enumerate(
        security_values
    ):

        with security_cards[
            index
        ]:

            st.metric(
                label,
                security_summary.get(
                    key,
                    0,
                ),
            )


    risk_df = pd.DataFrame(
        {
            "Risk Level": [
                "High Risk",
                "Medium Risk",
                "Low Risk",
            ],

            "Samples": [
                security_summary.get(
                    "high_risk",
                    0,
                ),

                security_summary.get(
                    "medium_risk",
                    0,
                ),

                security_summary.get(
                    "low_risk",
                    0,
                ),
            ],
        }
    )


    security_chart = px.bar(
        risk_df,
        x="Risk Level",
        y="Samples",
        title="Security Risk Distribution",
        text="Samples",
    )


    st.plotly_chart(
        security_chart,
        use_container_width=True,
    )


    check_df = pd.DataFrame(
        {
            "Security Check": [
                "Prompt Injection",
                "Jailbreak",
                "Toxicity",
                "Data Leakage",
            ],

            "Detected": [
                security_summary.get(
                    "prompt_injection",
                    0,
                ),

                security_summary.get(
                    "jailbreak",
                    0,
                ),

                security_summary.get(
                    "toxicity",
                    0,
                ),

                security_summary.get(
                    "data_leakage",
                    0,
                ),
            ],
        }
    )


    check_chart = px.bar(
        check_df,
        x="Security Check",
        y="Detected",
        title="Security Detection Results",
        text="Detected",
    )


    st.plotly_chart(
        check_chart,
        use_container_width=True,
    )


# ============================================================
# RELIABILITY
# ============================================================

st.divider()

st.header(
    "⚙️ Reliability & Performance"
)


reliability = evaluation.get(
    "reliability",
    {},
)


render_status(
    "Reliability",
    reliability,
)


available_reliability = (
    reliability.get(
        "available_metrics",
        [],
    )
)


unavailable_reliability = (
    reliability.get(
        "unavailable_metrics",
        [],
    )
)


if available_reliability:

    st.success(
        "Available: "
        + ", ".join(
            map(
                str,
                available_reliability,
            )
        )
    )


if unavailable_reliability:

    reliability_rows = []


    for item in (
        unavailable_reliability
    ):

        if isinstance(
            item,
            dict,
        ):

            reliability_rows.append(
                {
                    "Metric":
                        item.get(
                            "metric",
                            "Unknown",
                        ),

                    "Reason":
                        item.get(
                            "reason",
                            "Not available.",
                        ),
                }
            )

        else:

            reliability_rows.append(
                {
                    "Metric":
                        str(item),

                    "Reason":
                        "Not available.",
                }
            )


    if reliability_rows:

        st.table(pd.DataFrame(reliability_rows))


# ============================================================
# EXPORT EVALUATION JSON
# ============================================================

st.divider()

st.header(
    "📥 Export Results"
)


evaluation_json = json.dumps(
    evaluation,
    indent=2,
    default=str,
)


st.download_button(
    label="⬇️ Download Evaluation JSON",
    data=evaluation_json,
    file_name="aitrust_eval_results.json",
    mime="application/json",
)


explainability_json = json.dumps(
    explainability,
    indent=2,
    default=str,
)


st.download_button(
    label="⬇️ Download Explainability JSON",
    data=explainability_json,
    file_name="aitrust_eval_explainability.json",
    mime="application/json",
)


# ============================================================
# REPORT
# ============================================================

st.divider()

st.header(
    "📄 Generate Evaluation Report"
)


st.markdown(
    """
    Generate the complete AITrustEval HTML and PDF report.

    The report includes the baseline answer-quality metrics,
    RAGAS status, security, reliability, failure analysis,
    root-cause analysis and recommendations.
    """
)


if st.button(
    "📄 Generate HTML + PDF Report",
    use_container_width=True,
):

    try:

        report_data = build_report_data(
            evaluation_result=evaluation,

            dataset_info=dataset_info,

            schema_mapping=schema,

            metric_availability=
                evaluation.get(
                    "metric_availability",
                    {},
                ),

            baseline_metrics=
                baseline,

            retrieval_result=
                evaluation.get(
                    "retrieval",
                    {},
                ),

            ragas_result=
                evaluation.get(
                    "ragas",
                    {},
                ),

            security_result=
                evaluation.get(
                    "security",
                    {},
                ),

            reliability_result=
                evaluation.get(
                    "reliability",
                    {},
                ),

            failure_summary=
                failure_summary,

            root_cause_report=
                root_cause_df,

            recommendations=
                recommendations,

            explainability_result=
                explainability,
        )


        report_paths = (
            generate_reports(
                report_data
            )
        )


        st.success(
            "✅ HTML and PDF reports generated successfully."
        )


        html_path = Path(
            report_paths[
                "html"
            ]
        )


        pdf_path = Path(
            report_paths[
                "pdf"
            ]
        )


        if html_path.exists():

            st.download_button(
                label="⬇️ Download HTML Report",

                data=
                    html_path.read_bytes(),

                file_name=
                    html_path.name,

                mime="text/html",
            )


        if pdf_path.exists():

            st.download_button(
                label="⬇️ Download PDF Report",

                data=
                    pdf_path.read_bytes(),

                file_name=
                    pdf_path.name,

                mime="application/pdf",
            )


    except Exception as error:

        st.error(
            f"Report generation failed: {error}"
        )

        st.exception(
            error
        )


# ============================================================
# METHODOLOGY
# ============================================================

st.divider()

st.header(
    "ℹ️ Methodology & Limitations"
)


with st.expander(
    "📖 View Methodology"
):

    st.markdown(
        """
### Baseline Answer Quality

AITrustEval calculates:

- Answer Relevancy
- Answer Correctness
- Context-Answer Similarity

using a lightweight offline TF-IDF cosine-similarity baseline.

These scores are intended for baseline comparison and
failure screening.

They are **not equivalent to RAGAS or an LLM-as-a-judge
evaluation**.

### Retrieval

Semantic retrieval metrics require the local embedding
environment.

When explicit relevance labels are unavailable, retrieval
scores are treated as semantic/proxy retrieval measurements.

### RAG / LLM

RAGAS uses the approved local evaluator environment.

The current development laptop does not require Ollama.

### Security

Security evaluation includes:

- Prompt Injection
- Jailbreak
- Toxicity
- Data Leakage

### Reliability

Latency requires timing information.

Success/failure requires status information.

Consistency requires repeated questions.

Robustness requires controlled prompt variations or
perturbations.

### Explainability

AITrustEval provides:

- Evidence analysis
- Evidence-gap analysis
- Failure analysis
- Root-cause analysis
- Recommendations

### Reporting

Only metrics supported by the dataset and available
evaluation environment are reported.

Unavailable metrics are not fabricated.
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AITrustEval — Offline AI/ML Trustworthiness, Security & Reliability Evaluation Platform"
)

st.caption(
    "Development: Windows | Deployment Target: Approved Offline Ubuntu / DRDO"
)