import os
from pathlib import Path

import pytest
from clip_interrogator import Interrogator  # type: ignore

from horde_safety.deep_danbooru_model import DeepDanbooruModel, get_deep_danbooru_model
from horde_safety.interrogate import get_interrogator_no_blip


@pytest.fixture(scope="session")
def test_parent_folder() -> Path:
    HORDE_SAFETY_TEST_FOLDER = os.getenv("HORDE_SAFETY_TEST_FOLDER")

    if HORDE_SAFETY_TEST_FOLDER is None:
        raise Exception("Please set the environment variable HORDE_SAFETY_TEST_FOLDER")

    return Path(HORDE_SAFETY_TEST_FOLDER)


def _get_folders(test_parent_folder: Path, folder_keyword: str, is_pilot: bool = False) -> list[Path]:
    all_keyword_folders = [x for x in test_parent_folder.iterdir() if folder_keyword in x.name.lower()]

    if is_pilot:
        return [x for x in all_keyword_folders if "pilot" in x.name.lower()]
    else:
        return [x for x in all_keyword_folders if "pilot" not in x.name.lower()]


@pytest.fixture(scope="session")
def nsfw_pilot_folders(test_parent_folder: Path) -> list[Path]:
    return _get_folders(test_parent_folder, "nsfw", is_pilot=True)


@pytest.fixture(scope="session")
def lewd_pilot_folders(test_parent_folder: Path) -> list[Path]:
    return _get_folders(test_parent_folder, "lewd", is_pilot=True)


@pytest.fixture(scope="session")
def inappropriate_pilot_folders(test_parent_folder: Path) -> list[Path]:
    return _get_folders(test_parent_folder, "inappropriate", is_pilot=True)


@pytest.fixture(scope="session")
def sfw_pilot_folders(test_parent_folder: Path) -> list[Path]:
    return _get_folders(test_parent_folder, "safe", is_pilot=True)


@pytest.fixture(scope="session")
def neutral_pilot_folders(test_parent_folder: Path) -> list[Path]:
    return _get_folders(test_parent_folder, "neutral", is_pilot=True)


@pytest.fixture(scope="session")
def nsfw_folders(test_parent_folder: Path) -> list[Path]:
    return _get_folders(test_parent_folder, "nsfw")


@pytest.fixture(scope="session")
def sfw_folders(test_parent_folder: Path) -> list[Path]:
    return _get_folders(test_parent_folder, "safe")


@pytest.fixture(scope="session")
def lewd_folders(test_parent_folder: Path) -> list[Path]:
    return _get_folders(test_parent_folder, "lewd")


@pytest.fixture(scope="session")
def inappropriate_folders(test_parent_folder: Path) -> list[Path]:
    return _get_folders(test_parent_folder, "inappropriate")


@pytest.fixture(scope="session")
def neutral_folders(test_parent_folder: Path) -> list[Path]:
    return _get_folders(test_parent_folder, "neutral")


@pytest.fixture(scope="session")
def pilot_folders(
    nsfw_pilot_folders: list[Path],
    lewd_pilot_folders: list[Path],
    inappropriate_pilot_folders: list[Path],
    sfw_pilot_folders: list[Path],
    neutral_pilot_folders: list[Path],
) -> list[Path]:
    return (
        nsfw_pilot_folders
        + lewd_pilot_folders
        + inappropriate_pilot_folders
        + sfw_pilot_folders
        + neutral_pilot_folders
    )


@pytest.fixture(scope="session")
def pilot_folders_as_params(
    nsfw_pilot_folders: list[Path],
    lewd_pilot_folders: list[Path],
    inappropriate_pilot_folders: list[Path],
    sfw_pilot_folders: list[Path],
    neutral_pilot_folders: list[Path],
) -> dict[str, list[Path]]:
    return {
        "nsfw_pilot_folders": nsfw_pilot_folders,
        "lewd_pilot_folders": lewd_pilot_folders,
        "inappropriate_pilot_folders": inappropriate_pilot_folders,
        "sfw_pilot_folders": sfw_pilot_folders,
        "neutral_pilot_folders": neutral_pilot_folders,
    }


@pytest.fixture(scope="session")
def get_one_image_path(pilot_folders: list[Path]) -> Path:
    return pilot_folders[0].iterdir().__next__()


@pytest.fixture(scope="session")
def interrogator_no_blip() -> Interrogator:
    return get_interrogator_no_blip(device="cuda")


@pytest.fixture(scope="session")
def deep_danbooru_model() -> DeepDanbooruModel:
    return get_deep_danbooru_model(device="cuda")
