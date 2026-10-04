"""
AITrustEval - Local Model Manager

Centralized model configuration and discovery for local Ollama models.

The application can run on a development machine where some DRDO models
are unavailable. In that case, the manager reports them as unavailable
instead of failing the whole application.
"""

from typing import Dict, List, Optional

from evaluation.llm.ollama_client import (
    is_ollama_available,
    get_available_models,
    is_model_available,
)


# ============================================================
# DRDO MODEL INVENTORY
# ============================================================

LLM_MODELS = [
    "mistral",
    "mistral-nemo:12b-instruct-2407-q4-k-m",
    "RAG-nemo",
    "qwen3.5:9b",
    "qwen3.6:35b",
    "gemma-04b-q4k",
    "gemma 3:4b",
    "gemma 3:1b",
    "glm4-5b",
    "glm4",
    "gpt-oss:20b",
    "gpt-oss-20b-131k",
]

EMBEDDING_MODELS = [
    "qwen3-embedding:0.6b",
]

VISION_MODELS = [
    "llama3.2-vision",
    "minicpm-v",
]


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_LLM_MODEL = "mistral"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_INFO = {
    "mistral": {
        "type": "llm",
        "role": "Primary LLM Judge",
        "description": "General-purpose local LLM for evaluation and reasoning.",
    },
    "mistral-nemo:12b-instruct-2407-q4-k-m": {
        "type": "llm",
        "role": "LLM Judge",
        "description": "Instruction-tuned local model for evaluation.",
    },
    "RAG-nemo": {
        "type": "llm",
        "role": "RAG Evaluation",
        "description": "Local model intended for RAG-related workloads.",
    },
    "qwen3.5:9b": {
        "type": "llm",
        "role": "LLM Judge",
        "description": "General-purpose local LLM.",
    },
    "qwen3.6:35b": {
        "type": "llm",
        "role": "LLM Judge",
        "description": "Large local LLM for advanced evaluation.",
    },
    "gemma-04b-q4k": {
        "type": "llm",
        "role": "LLM Judge",
        "description": "Compact local LLM.",
    },
    "gemma 3:4b": {
        "type": "llm",
        "role": "LLM Judge",
        "description": "Compact general-purpose local LLM.",
    },
    "gemma 3:1b": {
        "type": "llm",
        "role": "Lightweight LLM",
        "description": "Small local model for lightweight evaluation.",
    },
    "glm4-5b": {
        "type": "llm",
        "role": "LLM Judge",
        "description": "General-purpose local LLM.",
    },
    "glm4": {
        "type": "llm",
        "role": "LLM Judge",
        "description": "General-purpose local LLM.",
    },
    "gpt-oss:20b": {
        "type": "llm",
        "role": "LLM Judge",
        "description": "Large local language model.",
    },
    "gpt-oss-20b-131k": {
        "type": "llm",
        "role": "Long Context LLM",
        "description": "Local model with long-context capability.",
    },
    "qwen3-embedding:0.6b": {
        "type": "embedding",
        "role": "Embedding Model",
        "description": "Local embedding model for semantic similarity and retrieval.",
    },
    "llama3.2-vision": {
        "type": "vision",
        "role": "Vision Model",
        "description": "Local multimodal/vision model.",
    },
    "minicpm-v": {
        "type": "vision",
        "role": "Vision Model",
        "description": "Local multimodal/vision model.",
    },
}


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize_model_name(name: str) -> str:
    """
    Normalize model names for comparison.

    Ollama model names are generally case-sensitive in usage, so this
    function is only intended for comparison/discovery.
    """
    return str(name).strip().lower()


def _find_actual_model_name(
    requested_name: str,
    available_models: List[str],
) -> Optional[str]:
    """
    Find the actual installed model name matching a configured name.

    Exact match is preferred. A case-insensitive match is used as fallback.
    """

    if requested_name in available_models:
        return requested_name

    requested_normalized = _normalize_model_name(requested_name)

    for model in available_models:
        if _normalize_model_name(model) == requested_normalized:
            return model

    return None


# ============================================================
# MODEL CATALOG
# ============================================================

def get_model_catalog() -> Dict[str, Dict]:
    """
    Return the configured DRDO model catalog.
    """

    catalog = {}

    for model in LLM_MODELS:
        catalog[model] = MODEL_INFO.get(
            model,
            {
                "type": "llm",
                "role": "LLM",
                "description": "Local language model.",
            },
        )

    for model in EMBEDDING_MODELS:
        catalog[model] = MODEL_INFO.get(
            model,
            {
                "type": "embedding",
                "role": "Embedding Model",
                "description": "Local embedding model.",
            },
        )

    for model in VISION_MODELS:
        catalog[model] = MODEL_INFO.get(
            model,
            {
                "type": "vision",
                "role": "Vision Model",
                "description": "Local vision model.",
            },
        )

    return catalog


# ============================================================
# AVAILABILITY
# ============================================================

def get_model_status() -> Dict:
    """
    Detect Ollama and determine which configured models are available.

    Returns a structured dictionary suitable for CLI and Streamlit.
    """

    ollama_available = is_ollama_available()

    result = {
        "ollama_available": ollama_available,
        "available_models": [],
        "llm_models": [],
        "embedding_models": [],
        "vision_models": [],
        "unavailable_models": [],
    }

    if not ollama_available:
        for model in (
            LLM_MODELS
            + EMBEDDING_MODELS
            + VISION_MODELS
        ):
            result["unavailable_models"].append(model)

        return result

    try:
        available_models = get_available_models()
    except Exception:
        available_models = []

    result["available_models"] = available_models

    for configured_model in LLM_MODELS:
        actual_model = _find_actual_model_name(
            configured_model,
            available_models,
        )

        if actual_model:
            result["llm_models"].append(actual_model)
        else:
            result["unavailable_models"].append(configured_model)

    for configured_model in EMBEDDING_MODELS:
        actual_model = _find_actual_model_name(
            configured_model,
            available_models,
        )

        if actual_model:
            result["embedding_models"].append(actual_model)
        else:
            result["unavailable_models"].append(configured_model)

    for configured_model in VISION_MODELS:
        actual_model = _find_actual_model_name(
            configured_model,
            available_models,
        )

        if actual_model:
            result["vision_models"].append(actual_model)
        else:
            result["unavailable_models"].append(configured_model)

    return result


# ============================================================
# MODEL SELECTION
# ============================================================

def get_available_llm_models() -> List[str]:
    """
    Return only currently available configured LLM models.
    """

    status = get_model_status()
    return status["llm_models"]


def get_available_embedding_models() -> List[str]:
    """
    Return only currently available configured embedding models.
    """

    status = get_model_status()
    return status["embedding_models"]


def get_available_vision_models() -> List[str]:
    """
    Return only currently available configured vision models.
    """

    status = get_model_status()
    return status["vision_models"]


def select_llm_model(
    preferred_model: Optional[str] = None,
) -> Optional[str]:
    """
    Select an available LLM model.

    Priority:
    1. User-selected model
    2. Default Mistral
    3. First available configured LLM
    """

    available = get_available_llm_models()

    if not available:
        return None

    if preferred_model:
        actual = _find_actual_model_name(
            preferred_model,
            available,
        )

        if actual:
            return actual

    default_actual = _find_actual_model_name(
        DEFAULT_LLM_MODEL,
        available,
    )

    if default_actual:
        return default_actual

    return available[0]


def select_embedding_model(
    preferred_model: Optional[str] = None,
) -> Optional[str]:
    """
    Select an available embedding model.
    """

    available = get_available_embedding_models()

    if not available:
        return None

    if preferred_model:
        actual = _find_actual_model_name(
            preferred_model,
            available,
        )

        if actual:
            return actual

    default_actual = _find_actual_model_name(
        DEFAULT_EMBEDDING_MODEL,
        available,
    )

    if default_actual:
        return default_actual

    return available[0]


# ============================================================
# VALIDATION
# ============================================================

def validate_model(
    model_name: str,
    model_type: Optional[str] = None,
) -> Dict:
    """
    Validate whether a model exists and is currently available.

    model_type can be:
        - llm
        - embedding
        - vision
    """

    status = get_model_status()

    available = status["available_models"]

    actual_model = _find_actual_model_name(
        model_name,
        available,
    )

    configured_info = MODEL_INFO.get(model_name)

    if actual_model is None:
        return {
            "valid": False,
            "model": model_name,
            "actual_model": None,
            "type": (
                configured_info.get("type")
                if configured_info
                else None
            ),
            "reason": "Model is not currently available in Ollama.",
        }

    detected_type = (
        configured_info.get("type")
        if configured_info
        else None
    )

    if model_type and detected_type != model_type:
        return {
            "valid": False,
            "model": model_name,
            "actual_model": actual_model,
            "type": detected_type,
            "reason": (
                f"Model is registered as '{detected_type}', "
                f"not '{model_type}'."
            ),
        }

    return {
        "valid": True,
        "model": model_name,
        "actual_model": actual_model,
        "type": detected_type,
        "reason": "Model is available.",
    }


# ============================================================
# SUMMARY
# ============================================================

def get_model_summary() -> Dict:
    """
    Create a compact summary for the dashboard.
    """

    status = get_model_status()

    return {
        "ollama_available": status["ollama_available"],
        "total_detected": len(status["available_models"]),
        "llm_count": len(status["llm_models"]),
        "embedding_count": len(status["embedding_models"]),
        "vision_count": len(status["vision_models"]),
        "available_llms": status["llm_models"],
        "available_embeddings": status["embedding_models"],
        "available_vision_models": status["vision_models"],
        "selected_default_llm": select_llm_model(),
        "selected_default_embedding": select_embedding_model(),
    }


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AITrustEval - Model Manager")
    print("=" * 60)

    summary = get_model_summary()

    print(f"\nOllama available: {summary['ollama_available']}")

    print(
        f"Detected configured models: "
        f"{summary['total_detected']}"
    )

    print(
        f"\nAvailable LLM models: "
        f"{summary['llm_count']}"
    )

    for model in summary["available_llms"]:
        print(f"  ✓ {model}")

    print(
        f"\nAvailable embedding models: "
        f"{summary['embedding_count']}"
    )

    for model in summary["available_embeddings"]:
        print(f"  ✓ {model}")

    print(
        f"\nAvailable vision models: "
        f"{summary['vision_count']}"
    )

    for model in summary["available_vision_models"]:
        print(f"  ✓ {model}")

    print("\nDefault LLM:")
    print(f"  {summary['selected_default_llm']}")

    print("\nDefault Embedding:")
    print(f"  {summary['selected_default_embedding']}")

    print("\n" + "=" * 60)