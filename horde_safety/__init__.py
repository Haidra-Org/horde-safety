"""Contains the functionality for client-side safety measures for the AI Horde."""

import os

from loguru import logger


def get_cache_folder_path() -> str:
    """Determine the cache folder path for Hugging Face Transformers.

    Priority:
    1. HF_HOME
    2. TRANSFORMERS_CACHE
    3. Default path

    Returns:
        str: The path to the appropriate huggingface cache folder.
    """
    hf_home = os.getenv("HF_HOME")
    transformers_cache = os.getenv("TRANSFORMERS_CACHE")
    default_path = os.path.expanduser("~/.cache/huggingface/")
    if hf_home:
        logger.debug(f"Using HF_HOME: {hf_home}")
        return hf_home
    elif transformers_cache:
        logger.debug(f"Using TRANSFORMERS_CACHE: {transformers_cache}")
        return transformers_cache
    else:
        logger.debug(f"Using default cache path: {default_path}")
        return default_path


def setup_cache_paths() -> str:
    """Set up the cache paths for Hugging Face Transformers based on environment variables.

    Returns:
        str: The path to the cache folder. This may be a horde-specific path if AIWORKER_CACHE_HOME is set.
            Else, it will use the default Hugging Face cache paths.
    """
    aiworker_cache_home = os.getenv("AIWORKER_CACHE_HOME")
    transformers_cache = os.getenv("TRANSFORMERS_CACHE")
    hf_home = os.getenv("HF_HOME")

    if aiworker_cache_home and not (transformers_cache or hf_home):
        cache_folder_path = os.path.join(aiworker_cache_home, "clip_blip")
        os.environ["HF_HOME"] = os.path.join(aiworker_cache_home, "hf_transformers")
        logger.debug(f"Set CACHE_FOLDER_PATH to {cache_folder_path} and HF_HOME to {os.environ['HF_HOME']}")
    else:
        if aiworker_cache_home:
            logger.info("TRANSFORMERS_CACHE or HF_HOME already set, not overriding")
        else:
            logger.info("AIWORKER_CACHE_HOME not set, using default huggingface cache paths.")
        cache_folder_path = get_cache_folder_path()

    os.makedirs(cache_folder_path, exist_ok=True)
    logger.debug(f"Created cache directory: {cache_folder_path}")

    return cache_folder_path


CACHE_FOLDER_PATH: str = setup_cache_paths()

from horde_safety.csam_checker import check_for_csam  # noqa: E402
from horde_safety.deep_danbooru_model import DeepDanbooruModel, get_deep_danbooru_model  # noqa: E402
from horde_safety.interrogate import CAPTION_MODELS, get_interrogator_no_blip  # noqa: E402

__all__ = [
    "get_interrogator_no_blip",
    "check_for_csam",
    "CAPTION_MODELS",
    "DeepDanbooruModel",
    "get_deep_danbooru_model",
]
