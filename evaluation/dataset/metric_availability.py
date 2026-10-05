# ============================================================
# AITrustEval
# Metric Availability Detection
# ============================================================


def _extract_schema(validation_result):
    """
    Extract the detected field mapping from different
    schema formats.
    """

    if validation_result is None:
        return {}

    if not isinstance(validation_result, dict):
        return {}

    # Format:
    # {
    #     "detected_columns": {
    #         "question": "question",
    #         ...
    #     }
    # }
    if "detected_columns" in validation_result:

        detected = validation_result.get(
            "detected_columns",
            {}
        )

        if isinstance(detected, dict):
            return detected

    # Direct schema:
    # {
    #     "question": "question",
    #     "context": "context",
    #     ...
    # }

    schema_fields = [
        "question",
        "context",
        "answer",
        "ground_truth",
        "metadata",
        "latency",
        "status",
    ]

    detected = {}

    for field in schema_fields:

        if field in validation_result:

            value = validation_result[field]

            if value is not None:
                detected[field] = value

    return detected


# ============================================================
# BUILD METRIC AVAILABILITY
# ============================================================

def build_metric_availability(
    validation_result,
    schema=None
):
    """
    Determine which metrics can be evaluated.

    Supports:

        build_metric_availability(schema)

    and:

        build_metric_availability(df, schema)
    """

    # --------------------------------------------------------
    # When called as:
    #
    # build_metric_availability(df, schema)
    #
    # use schema.
    # --------------------------------------------------------

    if schema is not None:

        detected_columns = _extract_schema(
            schema
        )

    else:

        detected_columns = _extract_schema(
            validation_result
        )

    # ========================================================
    # IMPORTANT:
    # Convert detected column names into TRUE/FALSE.
    # ========================================================

    has_question = bool(
        detected_columns.get("question")
    )

    has_context = bool(
        detected_columns.get("context")
    )

    has_answer = bool(
        detected_columns.get("answer")
    )

    has_ground_truth = bool(
        detected_columns.get("ground_truth")
    )

    has_metadata = bool(
        detected_columns.get("metadata")
    )

    has_latency = bool(
        detected_columns.get("latency")
    )

    has_status = bool(
        detected_columns.get("status")
    )

    # ========================================================
    # AVAILABILITY DICTIONARY
    # ========================================================

    availability = {}

    # --------------------------------------------------------
    # ANSWER QUALITY
    # --------------------------------------------------------

    availability["answer_relevancy"] = (
        has_question
        and has_answer
    )

    availability["answer_correctness"] = (
        has_answer
        and has_ground_truth
    )

    availability["context_answer_similarity"] = (
        has_context
        and has_answer
    )

    # --------------------------------------------------------
    # RAG / LLM METRICS
    # --------------------------------------------------------

    availability["faithfulness"] = (
        has_context
        and has_answer
    )

    availability["ragas_answer_relevancy"] = (
        has_question
        and has_answer
    )

    availability["ragas_answer_correctness"] = (
        has_answer
        and has_ground_truth
    )

    availability["context_precision"] = (
        has_context
        and has_ground_truth
    )

    availability["context_recall"] = (
        has_context
        and has_ground_truth
    )

    # --------------------------------------------------------
    # SEMANTIC RETRIEVAL
    # --------------------------------------------------------

    semantic_retrieval_ready = (
        has_context
        and has_ground_truth
    )

    availability["semantic_retrieval"] = (
        semantic_retrieval_ready
    )

    # --------------------------------------------------------
    # RETRIEVAL METRICS
    # --------------------------------------------------------

    availability["precision_at_k"] = (
        semantic_retrieval_ready
    )

    availability["recall_at_k"] = (
        semantic_retrieval_ready
    )

    availability["mrr"] = (
        semantic_retrieval_ready
    )

    availability["ndcg"] = (
        semantic_retrieval_ready
    )

    availability["hit_rate"] = (
        semantic_retrieval_ready
    )

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    availability["latency"] = (
        has_latency
    )

    # --------------------------------------------------------
    # RELIABILITY
    # --------------------------------------------------------

    availability["success_failure"] = (
        has_status
    )

    availability["consistency"] = (
        has_question
        and has_answer
    )

    availability["robustness"] = False

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    availability["metadata"] = (
        has_metadata
    )

    return availability


# ============================================================
# DETAILED METRIC INFORMATION
# ============================================================

def get_metric_details(
    validation_result,
    schema=None
):
    """
    Return detailed availability information.
    """

    if schema is not None:

        detected_columns = _extract_schema(
            schema
        )

    else:

        detected_columns = _extract_schema(
            validation_result
        )

    # --------------------------------------------------------
    # Convert values to Boolean
    # --------------------------------------------------------

    has_question = bool(
        detected_columns.get("question")
    )

    has_context = bool(
        detected_columns.get("context")
    )

    has_answer = bool(
        detected_columns.get("answer")
    )

    has_ground_truth = bool(
        detected_columns.get("ground_truth")
    )

    has_metadata = bool(
        detected_columns.get("metadata")
    )

    has_latency = bool(
        detected_columns.get("latency")
    )

    has_status = bool(
        detected_columns.get("status")
    )

    semantic_retrieval_ready = (
        has_context
        and has_ground_truth
    )

    return {

        "Answer Relevancy": {
            "available": (
                has_question
                and has_answer
            ),
            "reason": (
                "Question and answer fields detected."
                if has_question and has_answer
                else
                "Requires question and answer fields."
            )
        },

        "Answer Correctness": {
            "available": (
                has_answer
                and has_ground_truth
            ),
            "reason": (
                "Answer and ground truth detected."
                if has_answer and has_ground_truth
                else
                "Requires answer and ground truth."
            )
        },

        "Context-Answer Similarity": {
            "available": (
                has_context
                and has_answer
            ),
            "reason": (
                "Context and answer fields detected."
                if has_context and has_answer
                else
                "Requires context and answer."
            )
        },

        "Faithfulness": {
            "available": (
                has_context
                and has_answer
            ),
            "reason": (
                "Context and answer detected."
                if has_context and has_answer
                else
                "Requires context and answer."
            )
        },

        "RAGAS Answer Relevancy": {
            "available": (
                has_question
                and has_answer
            ),
            "reason": (
                "Question and answer detected."
                if has_question and has_answer
                else
                "Requires question and answer."
            )
        },

        "RAGAS Answer Correctness": {
            "available": (
                has_answer
                and has_ground_truth
            ),
            "reason": (
                "Answer and ground truth detected."
                if has_answer and has_ground_truth
                else
                "Requires answer and ground truth."
            )
        },

        "Context Precision": {
            "available": semantic_retrieval_ready,
            "reason": (
                "Context and ground truth detected. "
                "Semantic relevance can be derived using "
                "the local embedding model."
                if semantic_retrieval_ready
                else
                "Requires context and ground truth."
            )
        },

        "Context Recall": {
            "available": semantic_retrieval_ready,
            "reason": (
                "Context and ground truth detected. "
                "Semantic coverage can be evaluated."
                if semantic_retrieval_ready
                else
                "Requires context and ground truth."
            )
        },

        "Precision@K": {
            "available": semantic_retrieval_ready,
            "reason": (
                "Context and ground truth are available. "
                "Semantic relevance will be derived using "
                "the local embedding model."
                if semantic_retrieval_ready
                else
                "Requires context and ground truth."
            )
        },

        "Recall@K": {
            "available": semantic_retrieval_ready,
            "reason": (
                "Semantic retrieval evaluation is possible "
                "from context and ground truth."
                if semantic_retrieval_ready
                else
                "Requires context and ground truth."
            )
        },

        "MRR": {
            "available": semantic_retrieval_ready,
            "reason": (
                "Ranked context items can be evaluated "
                "after semantic relevance scoring."
                if semantic_retrieval_ready
                else
                "Requires context containing multiple "
                "retrieved items."
            )
        },

        "NDCG": {
            "available": semantic_retrieval_ready,
            "reason": (
                "Semantic similarity can provide graded "
                "relevance for ranked context items."
                if semantic_retrieval_ready
                else
                "Requires context and ground truth."
            )
        },

        "Hit Rate": {
            "available": semantic_retrieval_ready,
            "reason": (
                "Semantic relevance can determine whether "
                "at least one relevant context item is retrieved."
                if semantic_retrieval_ready
                else
                "Requires context and ground truth."
            )
        },

        "Latency": {
            "available": has_latency,
            "reason": (
                "Latency field detected."
                if has_latency
                else
                "No dedicated latency field detected."
            )
        },

        "Success / Failure": {
            "available": has_status,
            "reason": (
                "Status field detected."
                if has_status
                else
                "No dedicated status field detected."
            )
        },

        "Consistency": {
            "available": (
                has_question
                and has_answer
            ),
            "reason": (
                "Question and answer fields detected."
                if has_question and has_answer
                else
                "Requires question and answer fields."
            )
        },

        "Robustness": {
            "available": False,
            "reason": (
                "Requires controlled input variations "
                "or adversarial test cases."
            )
        },

        "Metadata": {
            "available": has_metadata,
            "reason": (
                "Metadata field detected."
                if has_metadata
                else
                "Metadata field not detected."
            )
        },
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_schema = {

        "question": "question",
        "context": "context",
        "answer": "answer",
        "ground_truth": "ground_truth",
        "metadata": "metadata",

    }

    availability = (
        build_metric_availability(
            test_schema
        )
    )

    details = (
        get_metric_details(
            test_schema
        )
    )

    print(
        "\n========== METRIC AVAILABILITY ==========\n"
    )

    for metric, available in availability.items():

        status = (
            "AVAILABLE"
            if available
            else "NOT AVAILABLE"
        )

        print(
            f"{metric:30} -> {status}"
        )

    print(
        "\n========== DETAILS ==========\n"
    )

    for metric, information in details.items():

        status = (
            "AVAILABLE"
            if information["available"]
            else "NOT AVAILABLE"
        )

        print(
            f"{metric:30} -> {status}"
        )

        print(
            f"  Reason: "
            f"{information['reason']}"
        )