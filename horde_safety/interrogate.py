"""Helper functions for using clip_interrogator."""

import torch
from clip_interrogator import Config, Interrogator  # type: ignore
from open_clip import CLIP  # type: ignore
from strenum import StrEnum
from torch import Tensor

from horde_safety import CACHE_FOLDER_PATH

_load_caption_model_func_def = Interrogator.load_caption_model
_encode_text_func_def = CLIP.encode_text


class CAPTION_MODELS(StrEnum):
    """The available caption models."""

    blip_base = "blip-base"
    blip_large = "blip-large"


class CLIP_MODELS(StrEnum):
    """The available clip models."""

    vit_l_14_open_ai = "ViT-L-14/openai"
    vit_h_14_open_ai = "ViT-H-14/laion2b_s32b_b79k"


_cached_tokens: list[tuple[Tensor, Tensor]] = []


def _encode_text_hijack(self, tokens: Tensor) -> Tensor:
    for cached_tokens, cached_result in _cached_tokens:
        if cached_tokens is tokens or cached_tokens.equal(tokens):
            return cached_result

    result = _encode_text_func_def(self, tokens)
    _cached_tokens.append((tokens, result))
    return result


_cached_text_array_features: dict[tuple[str, ...], Tensor] = {}
_cached_text_features: dict[str, Tensor] = {}


def _similarity_hijack(self, image_features: torch.Tensor, text: str) -> float:
    self._prepare_clip()

    if text in _cached_text_features:
        text_features = _cached_text_features[text]
    else:
        text_tokens = self.tokenize([text]).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)  # type: ignore
        _cached_text_features[text] = text_features  # type: ignore

    with torch.no_grad(), torch.cuda.amp.autocast():
        similarity = text_features @ image_features.T
    return similarity[0][0].item()


def _similarities_hijack(self, image_features: torch.Tensor, text_array: list[str]) -> list[float]:
    self._prepare_clip()

    text_array_key = tuple(text_array)
    if text_array_key in _cached_text_array_features:
        text_features = _cached_text_array_features[text_array_key]
    else:
        text_tokens = self.tokenize(list(text_array)).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)  # type: ignore
        _cached_text_array_features[text_array_key] = text_features  # type: ignore

    with torch.no_grad(), torch.cuda.amp.autocast():
        similarity = text_features @ image_features.T
    return similarity.T[0].tolist()


def get_interrogator_no_blip(
    clip_model_name: CLIP_MODELS = CLIP_MODELS.vit_l_14_open_ai,
    device: str | None = None,
) -> Interrogator:
    """Get an interrogator without the caption model loaded. This will immediately load the clip model in RAM.

    You should use this if you are not using the caption model.

    Args:
        clip_model_name: The name of the clip model to use. Defaults to CLIP_MODELS.vit_l_14_open_ai.
        device: The device to use. Possible values include `cuda` and `cpu`. \
        Defaults to `None` (which will use the default device for the version of PyTorch).

    Returns:
        clip_interrogator.Interrogator: An interrogator without the caption model loaded.
    """
    Interrogator.load_caption_model = lambda _: None
    interrogator = Interrogator(
        Config(
            clip_model_name=clip_model_name,
            cache_path=CACHE_FOLDER_PATH,
            clip_model_path=CACHE_FOLDER_PATH,
            device=device if device else Config.device,
        ),
    )
    Interrogator.load_caption_model = _load_caption_model_func_def

    Interrogator.similarity = _similarity_hijack
    Interrogator.similarities = _similarities_hijack

    CLIP.encode_text = _encode_text_hijack

    return interrogator


def get_interrogator(
    *,
    caption_model_name=CAPTION_MODELS.blip_large,
    clip_model_name: CLIP_MODELS = CLIP_MODELS.vit_l_14_open_ai,
    device: str | None = None,
) -> Interrogator:
    """Get an interrogator with the caption model loaded. This will immediately load the clip and caption models in
    RAM.

    You should use this if you are using the caption model.

    Args:
        caption_model_name: The name of the caption model to use. Defaults to CAPTION_MODELS.blip_large.
        clip_model_name: The name of the clip model to use. Defaults to CLIP_MODELS.vit_l_14_open_ai.
        device: The device to use. Possible values include `cuda` and `cpu`. \
        Defaults to `None` (which will use the default device for the version of PyTorch).

    Returns:
        clip_interrogator.Interrogator: An interrogator with the caption model loaded.
    """

    Interrogator.similarity = _similarity_hijack
    Interrogator.similarities = _similarities_hijack

    CLIP.encode_text = _encode_text_hijack

    return Interrogator(
        Config(
            caption_model_name=caption_model_name,
            clip_model_name=clip_model_name,
            clip_model_path=CACHE_FOLDER_PATH,
            cache_path=CACHE_FOLDER_PATH,
            device=device if device else Config.device,
        ),
    )
