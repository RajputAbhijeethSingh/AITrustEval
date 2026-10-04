import re


LEAKAGE_PATTERNS = {

    "email": (
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),

    "phone": (
        r"\b(?:\+91[-\s]?)?"
        r"[6-9]\d{9}\b"
    ),

    "credit_card": (
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
    ),

    "api_key": (
        r"\b(?:api[_-]?key|secret[_-]?key)"
        r"\s*[:=]\s*[A-Za-z0-9_\-]{8,}\b"
    ),

    "password": (
        r"\bpassword\s*[:=]\s*\S+"
    ),

    "private_key": (
        r"-----BEGIN .*PRIVATE KEY-----"
    )
}


def detect_data_leakage(text):
    """
    Detect possible sensitive information
    appearing in a model response.

    This is a pattern-based screening mechanism.
    """

    if text is None:
        text = ""

    text = str(text)

    detected_types = []


    for leakage_type, pattern in (
        LEAKAGE_PATTERNS.items()
    ):

        if re.search(
            pattern,
            text,
            re.IGNORECASE
        ):

            detected_types.append(
                leakage_type
            )


    detected = len(
        detected_types
    ) > 0


    if len(detected_types) >= 2:

        risk = "High"

    elif detected:

        risk = "Medium"

    else:

        risk = "Low"


    return {
        "detected": detected,
        "risk": risk,
        "detected_types": detected_types
    }


if __name__ == "__main__":

    test_text = (
        "Contact user@example.com "
        "for more information."
    )

    result = detect_data_leakage(
        test_text
    )

    print(result)