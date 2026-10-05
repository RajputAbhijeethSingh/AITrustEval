# ============================================================
# AITrustEval - Evidence Analysis
# ============================================================

from typing import Any, Dict, List


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(
    text: Any,
) -> str:
    """
    Normalize text for basic evidence comparison.
    """

    if text is None:
        return ""

    return " ".join(
        str(text)
        .lower()
        .strip()
        .split()
    )


# ============================================================
# TOKEN SET
# ============================================================

def get_tokens(
    text: Any,
) -> set:
    """
    Convert text into a basic set of tokens.
    """

    normalized = normalize_text(
        text
    )

    if not normalized:
        return set()

    return set(
        normalized.split()
    )


# ============================================================
# TEXT OVERLAP
# ============================================================

def calculate_text_overlap(
    text_a: Any,
    text_b: Any,
) -> float:
    """
    Calculate simple token overlap between two texts.

    This is an explainability aid, not a replacement
    for embedding-based evaluation.
    """

    tokens_a = get_tokens(
        text_a
    )

    tokens_b = get_tokens(
        text_b
    )

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = (
        tokens_a
        & tokens_b
    )

    return (
        len(intersection)
        / len(tokens_a)
    )


# ============================================================
# CONTEXT EVIDENCE
# ============================================================

def analyze_context_evidence(
    answer: str,
    contexts: List[str],
) -> Dict[str, Any]:
    """
    Determine which retrieved context chunks have the
    greatest textual overlap with the generated answer.
    """

    if not contexts:

        return {

            "available":
                False,

            "message":
                "No retrieved context is available.",

            "chunks":
                [],
        }

    results = []

    for index, context in enumerate(
        contexts,
        start=1,
    ):

        overlap = (
            calculate_text_overlap(
                answer,
                context,
            )
        )

        results.append({

            "rank":
                index,

            "overlap":
                round(
                    overlap,
                    4,
                ),

            "context":
                context,
        })

    results.sort(
        key=lambda item:
            item["overlap"],
        reverse=True,
    )

    return {

        "available":
            True,

        "message":
            "Context evidence analyzed.",

        "chunks":
            results,
    }


# ============================================================
# GROUND TRUTH EVIDENCE
# ============================================================

def analyze_ground_truth_evidence(
    answer: str,
    ground_truth: str,
) -> Dict[str, Any]:
    """
    Compare the answer with the ground-truth text.
    """

    overlap = (
        calculate_text_overlap(
            answer,
            ground_truth,
        )
    )

    return {

        "available":
            bool(ground_truth),

        "overlap":
            round(
                overlap,
                4,
            ),

        "answer":
            answer,

        "ground_truth":
            ground_truth,
    }


# ============================================================
# RETRIEVAL EVIDENCE
# ============================================================

def analyze_retrieval_evidence(
    contexts: List[str],
    ground_truth: str,
) -> Dict[str, Any]:
    """
    Determine which retrieved chunks are most similar
    at the basic lexical level to the ground truth.
    """

    if not contexts:

        return {

            "available":
                False,

            "chunks":
                [],
        }

    results = []

    for index, context in enumerate(
        contexts,
        start=1,
    ):

        overlap = (
            calculate_text_overlap(
                ground_truth,
                context,
            )
        )

        results.append({

            "rank":
                index,

            "overlap":
                round(
                    overlap,
                    4,
                ),

            "context":
                context,
        })

    results.sort(
        key=lambda item:
            item["overlap"],
        reverse=True,
    )

    return {

        "available":
            True,

        "chunks":
            results,
    }


# ============================================================
# IDENTIFY EVIDENCE GAP
# ============================================================

def identify_evidence_gap(
    answer: str,
    contexts: List[str],
    ground_truth: str,
) -> Dict[str, Any]:
    """
    Identify whether the answer has weak supporting evidence.
    """

    answer_context = (
        analyze_context_evidence(
            answer,
            contexts,
        )
    )

    ground_truth_context = (
        analyze_retrieval_evidence(
            contexts,
            ground_truth,
        )
    )

    if not contexts:

        return {

            "gap_detected":
                True,

            "type":
                "no_context",

            "message":
                "No retrieved context was available to support the answer.",
        }

    if not answer_context[
        "chunks"
    ]:

        return {

            "gap_detected":
                True,

            "type":
                "no_evidence",

            "message":
                "No supporting context could be identified.",
        }

    best_answer_overlap = max(

        item["overlap"]

        for item
        in answer_context[
            "chunks"
        ]
    )

    best_ground_truth_overlap = max(

        item["overlap"]

        for item
        in ground_truth_context[
            "chunks"
        ]
    )

    if best_answer_overlap < 0.10:

        return {

            "gap_detected":
                True,

            "type":
                "weak_answer_support",

            "message":
                (
                    "The generated answer has weak "
                    "lexical overlap with the retrieved "
                    "context."
                ),

            "best_answer_context_overlap":
                best_answer_overlap,

            "best_ground_truth_context_overlap":
                best_ground_truth_overlap,
        }

    return {

        "gap_detected":
            False,

        "type":
            "supported",

        "message":
            (
                "The answer has identifiable supporting "
                "evidence in the retrieved context."
            ),

        "best_answer_context_overlap":
            best_answer_overlap,

        "best_ground_truth_context_overlap":
            best_ground_truth_overlap,
    }


# ============================================================
# COMPLETE EVIDENCE ANALYSIS
# ============================================================

def analyze_evidence(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
) -> Dict[str, Any]:
    """
    Complete evidence analysis for one sample.
    """

    context_evidence = (
        analyze_context_evidence(
            answer,
            contexts,
        )
    )

    ground_truth_evidence = (
        analyze_ground_truth_evidence(
            answer,
            ground_truth,
        )
    )

    retrieval_evidence = (
        analyze_retrieval_evidence(
            contexts,
            ground_truth,
        )
    )

    evidence_gap = (
        identify_evidence_gap(
            answer,
            contexts,
            ground_truth,
        )
    )

    return {

        "question":
            question,

        "answer":
            answer,

        "ground_truth":
            ground_truth,

        "context_evidence":
            context_evidence,

        "ground_truth_evidence":
            ground_truth_evidence,

        "retrieval_evidence":
            retrieval_evidence,

        "evidence_gap":
            evidence_gap,
    }


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 65
    )

    print(
        "AITrustEval - Evidence Analysis Test"
    )

    print(
        "=" * 65
    )

    question = (
        "What is TCP congestion control?"
    )

    contexts = [

        (
            "TCP uses congestion control mechanisms "
            "to prevent network congestion."
        ),

        (
            "TCP uses a three-way handshake to "
            "establish a connection."
        ),

        (
            "Congestion control adjusts the sending "
            "rate based on network conditions."
        ),

        (
            "UDP does not establish a connection "
            "before transmitting data."
        ),

        (
            "TCP retransmits lost packets."
        ),
    ]

    answer = (
        "TCP uses congestion control to prevent "
        "network congestion and adjusts its sending "
        "rate according to network conditions."
    )

    ground_truth = (
        "TCP controls congestion by adjusting its "
        "sending rate according to network conditions."
    )

    result = analyze_evidence(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
    )

    # --------------------------------------------------------
    # Context evidence
    # --------------------------------------------------------

    print(
        "\n========== ANSWER / CONTEXT EVIDENCE =========="
    )

    for chunk in result[
        "context_evidence"
    ][
        "chunks"
    ]:

        print(
            f"\nRank: "
            f"{chunk['rank']}"
        )

        print(
            f"Overlap: "
            f"{chunk['overlap']}"
        )

        print(
            f"Context: "
            f"{chunk['context']}"
        )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    print(
        "\n========== GROUND TRUTH EVIDENCE =========="
    )

    gt = result[
        "ground_truth_evidence"
    ]

    print(
        f"Overlap: "
        f"{gt['overlap']}"
    )

    # --------------------------------------------------------
    # Retrieval evidence
    # --------------------------------------------------------

    print(
        "\n========== RETRIEVAL EVIDENCE =========="
    )

    for chunk in result[
        "retrieval_evidence"
    ][
        "chunks"
    ]:

        print(
            f"\nRank: "
            f"{chunk['rank']}"
        )

        print(
            f"Ground-truth overlap: "
            f"{chunk['overlap']}"
        )

    # --------------------------------------------------------
    # Evidence gap
    # --------------------------------------------------------

    print(
        "\n========== EVIDENCE GAP =========="
    )

    gap = result[
        "evidence_gap"
    ]

    print(
        f"Gap detected: "
        f"{gap['gap_detected']}"
    )

    print(
        f"Type: "
        f"{gap['type']}"
    )

    print(
        f"Message: "
        f"{gap['message']}"
    )

    print(
        "\n" + "=" * 65
    )

    print(
        "EVIDENCE ANALYSIS TEST COMPLETE"
    )

    print(
        "=" * 65
    )