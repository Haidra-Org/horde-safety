import re
from pathlib import Path

from loguru import logger
from unidecode import unidecode

weight_remover_regex = re.compile(r"\((.*?):\d+\.\d+\)")
whitespace_remover_regex = re.compile(r"(\s(\w)){3,}\b")
whitespace_converter_regex = re.compile(r"([^\w\s]|_)")


def get_model_details(model_info: dict | None) -> tuple[bool, list[str]]:
    """Gets the model details from the model info dict

    Args:
        model_info (dict): The model info dict

    Returns:
        tuple[bool, list[str]]: The model's is_nsfw value, and a list of the model's tags
    """
    if not model_info:
        model_info = {}
    model_nsfw = model_info.get("nsfw")
    if model_nsfw is None:
        logger.warning("Model info did not contain nsfw, assuming True (this may increase false positives).")
        logger.warning("Pass in a model reference model entry to prevent this warning.")
        model_nsfw = True
    model_tags = model_info.get("tags")
    if not model_tags:
        model_tags = []

    return model_nsfw, model_tags


def normalize_prompt(prompt: str) -> tuple[str, str | None]:
    """Prepares the prompt to be scanned by the regex, by removing tricks one might use to avoid the filters

    Returns:
        tuple[str, str]: The normalized prompt, and the negative prompt (if it exists)
    """
    # Split the prompt into the main prompt and the negative prompt (if it exists)
    neg_prompt: str | None = None
    if "###" in prompt:
        prompt, neg_prompt = prompt.split("###", 1)

    # Remove weight markers from the prompt
    prompt = weight_remover_regex.sub(r"\1", prompt)

    # Convert all whitespace to spaces
    prompt = whitespace_converter_regex.sub(" ", prompt)

    # Replace trimmed whitespace with no whitespace
    for match in re.finditer(whitespace_remover_regex, prompt):
        trim_match = match.group(0).strip()
        replacement = re.sub(r"\s+", "", trim_match)
        prompt = prompt.replace(trim_match, replacement)

    # Replace multiple spaces with a single space
    prompt = re.sub(r"\s+", " ", prompt)

    # Remove all accents from the prompt
    prompt = unidecode(prompt)

    # Normalize the negative prompt (if it exists)
    if neg_prompt:
        neg_prompt = weight_remover_regex.sub(r"\1", neg_prompt)
        neg_prompt = whitespace_converter_regex.sub(" ", neg_prompt)
        for match in re.finditer(whitespace_remover_regex, neg_prompt):
            trim_match = match.group(0).strip()
            replacement = re.sub(r"\s+", "", trim_match)
            neg_prompt = neg_prompt.replace(trim_match, replacement)
        neg_prompt = re.sub(r"\s+", " ", neg_prompt)
        neg_prompt = unidecode(neg_prompt)

    # Return the normalized prompt and negative prompt (if it exists)
    return prompt, neg_prompt


def add_value_to_dict_array(dict_to_modify, array_key, value) -> None:
    """Adds a value to an array stored in a dict key
    If the key does not exist, it is created
    """
    if array_key not in dict_to_modify:
        dict_to_modify[array_key] = []
    dict_to_modify[array_key].append(value)


def get_image_file_paths(folder: Path) -> list[Path]:
    """Returns a list of image file paths in the folder.

    Detects jpg, jpeg, png, and webp.

    Args:
        folder (Path): The folder to search.

    Returns:
        list[Path]: The list of image file paths.
    """
    folder = Path(folder)
    image_paths = list(folder.glob("*.jpg"))
    image_paths.extend(list(folder.glob("*.jpeg")))
    image_paths.extend(list(folder.glob("*.png")))
    image_paths.extend(list(folder.glob("*.webp")))

    return image_paths
