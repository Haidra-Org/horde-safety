import time
from pathlib import Path

import PIL
from tqdm import tqdm

from horde_safety.deep_danbooru_model import download_deep_danbooru_model, get_deep_danbooru_model
from horde_safety.interrogate import get_interrogator, get_interrogator_no_blip


def test_get_interrogator_no_blip():
    interrogator = get_interrogator_no_blip()
    assert interrogator is not None
    assert not hasattr(interrogator, "caption_model")


def test_get_interrogator():
    interrogator = get_interrogator()
    assert interrogator is not None
    assert interrogator.caption_model is not None


def test_download_deep_danbooru_model():
    download_deep_danbooru_model()


def test_get_deep_danbooru_model():
    deepdanbooru_model = get_deep_danbooru_model(device="cpu")
    assert deepdanbooru_model is not None
    assert deepdanbooru_model._initial_device == "cpu"

    del deepdanbooru_model

    deepdanbooru_model = get_deep_danbooru_model(device="cuda")
    assert deepdanbooru_model is not None
    assert deepdanbooru_model._initial_device == "cuda"


def test_hijacks(get_one_image_path: Path):
    interrogator = get_interrogator_no_blip()

    image = PIL.Image.open(get_one_image_path)

    image_tensor = interrogator.image_to_features(image)

    start_time = time.time()
    result = interrogator.similarity(image_tensor, "photo")

    assert result is not None
    assert isinstance(result, float)

    print(f"Time taken: {time.time() - start_time}")

    for _ in tqdm(range(100)):
        result = interrogator.similarity(image_tensor, "photo")

        assert result is not None
        assert isinstance(result, float)
