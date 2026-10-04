import requests


OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral"


def is_ollama_available():
    """
    Check whether Ollama is running locally.
    """

    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=5
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


def get_available_models():
    """
    Return the models currently available in Ollama.
    """

    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=5
        )

        response.raise_for_status()

        data = response.json()

        return [
            model["name"]
            for model in data.get("models", [])
        ]

    except requests.RequestException:
        return []


def is_model_available(model_name=DEFAULT_MODEL):
    """
    Check whether a specific model is available.
    """

    models = get_available_models()

    return any(
        model == model_name
        or model.startswith(model_name + ":")
        for model in models
    )


def generate(
    prompt,
    model=DEFAULT_MODEL,
    temperature=0.0
):
    """
    Send a prompt to the local Ollama model
    and return the generated response.
    """

    if not is_ollama_available():

        raise RuntimeError(
            "Ollama is not running or is not available."
        )


    if not is_model_available(model):

        available_models = get_available_models()

        raise RuntimeError(
            f"Model '{model}' is not available. "
            f"Available models: {available_models}"
        )


    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }


    try:

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        return data.get("response", "").strip()

    except requests.RequestException as e:

        raise RuntimeError(
            f"Ollama request failed: {e}"
        )


if __name__ == "__main__":

    print("=" * 50)
    print("AITrustEval - Ollama Connection Test")
    print("=" * 50)


    if not is_ollama_available():

        print("❌ Ollama is not available.")
        print(
            "This is expected if Ollama is not installed "
            "on the current machine."
        )

    else:

        print("✅ Ollama is running.")


        models = get_available_models()

        print("\nAvailable Models:")

        for model in models:

            print(f"  - {model}")


        if is_model_available("mistral"):

            print("\n✅ Mistral model found.")

            print("\nTesting Mistral...")

            try:

                answer = generate(
                    "Explain what AI evaluation means in one simple sentence.",
                    model="mistral"
                )

                print("\nMistral Response:")
                print(answer)

            except Exception as e:

                print(
                    f"\n❌ Mistral test failed: {e}"
                )

        else:

            print("\n❌ Mistral model not found.")