import os
import pathlib

from loguru import logger

CACHE_FOLDER_PATH: str | None = None

AIWORKER_CACHE_HOME = os.getenv("AIWORKER_CACHE_HOME")

if AIWORKER_CACHE_HOME:
    CACHE_FOLDER_PATH = os.path.join(AIWORKER_CACHE_HOME, "clip_blip")
    if os.getenv("TRANSFORMERS_CACHE") is None:
        os.environ["TRANSFORMERS_CACHE"] = AIWORKER_CACHE_HOME + "/hf_transformers"
    else:
        logger.info("TRANSFORMERS_CACHE already set, not overriding")
else:
    logger.info("AIWORKER_CACHE_HOME not set, using default huggingface cache paths.")
