from clip_interrogator import Config, Interrogator, list_caption_models  # type: ignore
from strenum import StrEnum

from horde_safety import CACHE_FOLDER_PATH

_load_caption_model_func_def = Interrogator.load_caption_model


class CAPTION_MODELS(StrEnum):
    blip_base = "blip-base"
    blip_large = "blip-large"


class HordeInterrogateManager:
    def get_interrogator_no_blip(
        self,
        clip_model_name: str = "ViT-L-14/openai",
    ) -> Interrogator:
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
        self,
        *,
        blip_model_name=CAPTION_MODELS.blip_large,
        clip_model_name: str = "ViT-L-14/openai",
    ) -> Interrogator:
        return Interrogator(
            Config(
                blip_model_name=blip_model_name,
                clip_model_name=clip_model_name,
                clip_model_path=CACHE_FOLDER_PATH,
                cache_path=CACHE_FOLDER_PATH,
            ),
        )
