"""Contains the functionality for client-side safety measures for the AI Horde."""

# flake8: noqa
import os
from loguru import logger

CACHE_FOLDER_PATH: str = os.path.expanduser("~/.cache/huggingface/")

AIWORKER_CACHE_HOME = os.getenv("AIWORKER_CACHE_HOME")
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE")
HF_HOME = os.getenv("HF_HOME")

if AIWORKER_CACHE_HOME:
    if not TRANSFORMERS_CACHE and not HF_HOME:
        CACHE_FOLDER_PATH = os.path.join(AIWORKER_CACHE_HOME, "clip_blip")
        os.environ["HF_HOME"] = os.path.join(AIWORKER_CACHE_HOME, "hf_transformers")
    else:
        logger.info("TRANSFORMERS_CACHE or HF_HOME already set, not overriding")
else:
    logger.info("AIWORKER_CACHE_HOME not set, using default huggingface cache paths.")
    if HF_HOME:
        CACHE_FOLDER_PATH = HF_HOME
        logger.debug(f"Using HF_HOME: {HF_HOME}")
    elif TRANSFORMERS_CACHE:
        CACHE_FOLDER_PATH = TRANSFORMERS_CACHE
        logger.debug(f"Using TRANSFORMERS_CACHE: {TRANSFORMERS_CACHE}")
    else:
        logger.debug(f"Using default cache path: {CACHE_FOLDER_PATH}")

from horde_safety.interrogate import get_interrogator_no_blip, CAPTION_MODELS
from horde_safety.csam_checker import check_for_csam
from horde_safety.deep_danbooru_model import DeepDanbooruModel, get_deep_danbooru_model

__all__ = [
    "get_interrogator_no_blip",
    "check_for_csam",
    "CAPTION_MODELS",
    "DeepDanbooruModel",
    "get_deep_danbooru_model",
]
