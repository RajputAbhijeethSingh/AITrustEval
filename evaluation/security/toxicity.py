import re


TOXIC_PATTERNS = [
    r"\bkill\b",
    r"\bhate\b",
    r"\bidiot\b",
    r"\bstupid\b",
    r"\bmor[o0]n\b",
    r"\bshut up\b",
    r"\bviolent\b",
    r"\battack\b",
    r"\bharm\b",
]


def detect_toxicity(text):
    """
    Perform a basic rule-based toxicity screening.

    This is a preliminary detector and should not
    be treated as a production toxicity classifier.
    """

    if text is None:
        text = ""

    text = str(text).lower()

    matched_patterns = []

    for pattern in TOXIC_PATTERNS:

        if re.search(pattern, text):

            matched_patterns.append(
                pattern
            )


    detected = len(
        matched_patterns
    ) > 0


    if len(matched_patterns) >= 3:

        risk = "High"

    elif len(matched_patterns) >= 1:

        risk = "Medium"

    else:

        risk = "Low"


    return {
        "detected": detected,
        "risk": risk,
        "matched_patterns": matched_patterns
    }


if __name__ == "__main__":

    test_text = (
        "I hate this stupid system."
    )

    result = detect_toxicity(
        test_text
    )

    print(result)