"""Helper functions for using clip_interrogator."""


from clip_interrogator import Config, Interrogator  # type: ignore
from strenum import StrEnum

from horde_safety import CACHE_FOLDER_PATH

_load_caption_model_func_def = Interrogator.load_caption_model


class CAPTION_MODELS(StrEnum):
    """The available caption models."""

    blip_base = "blip-base"
    blip_large = "blip-large"


class CLIP_MODELS(StrEnum):
    """The available clip models."""

    vit_l_14_open_ai = "ViT-L-14/openai"


def get_interrogator_no_blip(
    clip_model_name: CLIP_MODELS = CLIP_MODELS.vit_l_14_open_ai,
) -> Interrogator:
    """Get an interrogator without the caption model loaded. This will immediately load the clip model in RAM.

    You should use this if you are not using the caption model.

    Args:
        clip_model_name: The name of the clip model to use. Defaults to CLIP_MODELS.vit_l_14_open_ai.

    Returns:
        clip_interrogator.Interrogator: An interrogator without the caption model loaded.
    """
    Interrogator.load_caption_model = lambda _: None
    interrogator = Interrogator(
        Config(
            clip_model_name=clip_model_name,
            cache_path=CACHE_FOLDER_PATH,
            clip_model_path=CACHE_FOLDER_PATH,
        ),
    )
    Interrogator.load_caption_model = _load_caption_model_func_def
    return interrogator


def get_interrogator(
    *,
    blip_model_name=CAPTION_MODELS.blip_large,
    clip_model_name: CLIP_MODELS = CLIP_MODELS.vit_l_14_open_ai,
) -> Interrogator:
    """Get an interrogator with the caption model loaded. This will immediately load the clip and caption models in
    RAM.

    You should use this if you are using the caption model.

    Args:
        blip_model_name: The name of the caption model to use. Defaults to CAPTION_MODELS.blip_large.
        clip_model_name: The name of the clip model to use. Defaults to CLIP_MODELS.vit_l_14_open_ai.

    Returns:
        clip_interrogator.Interrogator: An interrogator with the caption model loaded.
    """
    return Interrogator(
        Config(
            blip_model_name=blip_model_name,
            clip_model_name=clip_model_name,
            clip_model_path=CACHE_FOLDER_PATH,
            cache_path=CACHE_FOLDER_PATH,
        ),
    )
