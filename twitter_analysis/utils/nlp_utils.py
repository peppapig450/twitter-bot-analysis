import importlib.util
import logging

import spacy
from spacy.language import Language
from spacy.util import is_package

# Check NLTK availability at module level
NLTK_AVAILABLE: bool = bool(importlib.util.find_spec("nltk"))


def load_nlp_model(model_name: str) -> Language | None:
    """
    Load a spaCy language model with robust error handling.

    Args:
        model_name: Name of the spaCy model to load (e.g., 'en_core_web_sm').

    Returns:
        Loaded spaCy Language object if successful, None otherwise.

    Notes:
        Assumes model_name is a valid string (e.g., from ConfigModel).
        Logs warnings or errors if the model is not installed or fails to load.

    """
    if not is_package(model_name):
        logging.warning("spaCy model %s is not installed", model_name)
        return None

    try:
        return spacy.load(model_name)
    except OSError:
        logging.exception("Failed to load spaCy model '%s'", model_name)
        return None
    except ValueError:
        logging.warning("Invalid spaCy model name '%s'", model_name)
        return None
