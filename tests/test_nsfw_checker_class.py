import json
import sys
import time
from enum import Enum
from pathlib import Path

import numpy as np
import PIL.Image
import pytest
from clip_interrogator.clip_interrogator import tqdm
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from torch import device
from transformers.models.clip import CLIPFeatureExtractor

from horde_safety.deep_danbooru_model import DeepDanbooruModel
from horde_safety.interrogate import Interrogator
from horde_safety.nsfw_checker_class import NSFWChecker, NSFWFolderChecker, NSFWFolderResults
from horde_safety.utils import get_image_file_paths


@pytest.fixture()
def stable_diffusion_safety_checker() -> StableDiffusionSafetyChecker:
    return StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker",
        device_map="cuda",
    )


@pytest.fixture()
def clip_feature_extractor() -> CLIPFeatureExtractor:
    return CLIPFeatureExtractor()


def test_stable_diffusion_safety_checker(
    stable_diffusion_safety_checker: StableDiffusionSafetyChecker,
    clip_feature_extractor: CLIPFeatureExtractor,
    nsfw_pilot_folders: list[Path],
    sfw_pilot_folders: list[Path],
    neutral_pilot_folders: list[Path],
    inappropriate_pilot_folders: list[Path],
    nsfw_folders: list[Path],
    sfw_folders: list[Path],
    lewd_folders: list[Path],
    inappropriate_folders: list[Path],
    neutral_folders: list[Path],
):

    all_folders_to_check = [
        *nsfw_pilot_folders,
        *sfw_pilot_folders,
        *neutral_pilot_folders,
        *inappropriate_pilot_folders,
        *nsfw_folders,
        *sfw_folders,
        *lewd_folders,
        *inappropriate_folders,
        *neutral_folders,
    ]

    assert len(all_folders_to_check) == (
        len(nsfw_pilot_folders)
        + len(sfw_pilot_folders)
        + len(neutral_pilot_folders)
        + len(inappropriate_pilot_folders)
        + len(nsfw_folders)
        + len(sfw_folders)
        + len(lewd_folders)
        + len(inappropriate_folders)
        + len(neutral_folders)
    )

    resize_targets = []
    for i in range(128, 769, 128):
        for j in range(128, 769, 128):
            if i == j:
                resize_targets.append((i, j))
            else:
                resize_targets.append((i, j))
                resize_targets.append((j, i))

    def get_closest_resize_target(image: PIL.Image.Image) -> tuple[int, int]:
        width, height = image.size
        closest_target = min(
            resize_targets,
            key=lambda target: abs(target[0] - width) + abs(target[1] - height),
        )
        return closest_target

    import logging

    logging.getLogger("diffusers").setLevel(logging.CRITICAL)

    for folder in all_folders_to_check:

        image_file_paths = get_image_file_paths(folder)

        progress_bar = tqdm(
            total=len(image_file_paths),
            file=sys.stdout,
        )

        total_sfw_images = 0
        total_nsfw_images = 0
        total_errors = 0

        for image_file_path in image_file_paths:
            try:
                image = PIL.Image.open(image_file_path)
                image = image.resize(get_closest_resize_target(image), resample=PIL.Image.LANCZOS)
                image = image.convert("RGB")

                image_features = clip_feature_extractor(images=image, return_tensors="pt").to(device("cuda"))
                _, has_nsfw_concept = stable_diffusion_safety_checker(  # type: ignore
                    clip_input=image_features["pixel_values"],
                    images=[np.asarray(image)],
                )

                if has_nsfw_concept[0]:
                    total_nsfw_images += 1
                else:
                    total_sfw_images += 1

                progress_bar.set_postfix(
                    {
                        "folder": folder.name,
                        "total_sfw_images": total_sfw_images,
                        "sfw_percent": total_sfw_images / (total_sfw_images + total_nsfw_images) * 100,
                        "total_nsfw_images": total_nsfw_images,
                        "nsfw_percent": total_nsfw_images / (total_sfw_images + total_nsfw_images) * 100,
                        "total_errors": total_errors,
                    },
                    refresh=True,
                )

                progress_bar.update(1)

            except Exception as e:
                total_errors += 1
                print(f"Error processing {image_file_path}: {e}")

        progress_bar.close()


class TestNSFWChecker:
    def test_nsfw_checker_init(
        self,
        interrogator_no_blip: Interrogator,
        deep_danbooru_model: DeepDanbooruModel,
    ):
        nsfw_checker = NSFWChecker(interrogator_no_blip, deep_danbooru_model)
        assert nsfw_checker is not None
        assert nsfw_checker.interrogator is not None
        assert nsfw_checker.deep_danbooru_model is not None

    def test_nsfw_checker_get_similarities(
        self,
        interrogator_no_blip: Interrogator,
        deep_danbooru_model: DeepDanbooruModel,
        get_one_image_path: Path,
    ):
        nsfw_checker = NSFWChecker(interrogator_no_blip, deep_danbooru_model)
        results = nsfw_checker.get_nsfw_concept_similarities(PIL.Image.open(get_one_image_path))
        assert len(results) == len(nsfw_checker.all_nsfw_concepts)

    def test_nsfw_checker_check_for_nsfw(
        self,
        interrogator_no_blip: Interrogator,
        deep_danbooru_model: DeepDanbooruModel,
        get_one_image_path: Path,
    ):
        nsfw_checker = NSFWChecker(interrogator_no_blip, deep_danbooru_model)
        result = nsfw_checker.check_for_nsfw(PIL.Image.open(get_one_image_path))
        assert result is not None
        assert result.nsfw_similarity_results is not None
        assert len(result.nsfw_similarity_results) == len(nsfw_checker.all_nsfw_concepts)

    def test_nsfw_folder_checker(
        self,
        interrogator_no_blip: Interrogator,
        deep_danbooru_model: DeepDanbooruModel,
        nsfw_pilot_folders: list[Path],
    ):
        nsfw_folder_checker = NSFWFolderChecker(
            interrogator_no_blip,
            deep_danbooru_model,
            nsfw_pilot_folders=nsfw_pilot_folders,
        )

        all_folder_results = nsfw_folder_checker.check_all_folders()

        assert all_folder_results is not None
        assert len(all_folder_results) == len(nsfw_pilot_folders)

        for folder_result in all_folder_results:
            assert folder_result is not None
            assert folder_result.folder_path is not None
            assert len(folder_result.all_results) > 0

    def test_nsfw_folder_checker_pilot_folders(
        self,
        interrogator_no_blip: Interrogator,
        deep_danbooru_model: DeepDanbooruModel,
        pilot_folders_as_params: dict[str, list[Path]],
    ):
        nsfw_folder_checker = NSFWFolderChecker(
            **{
                "interrogator": interrogator_no_blip,
                "deep_danbooru_model": deep_danbooru_model,
                **pilot_folders_as_params,
            },
        )

        for folder_path, nsfw_type_expected in nsfw_folder_checker.get_all_pilot_folders():
            assert folder_path is not None
            assert nsfw_type_expected is not None

            folder_result = nsfw_folder_checker.check_folder(folder_path, nsfw_type_expected)

            assert folder_result is not None

            self.write_results(folder_result)

    def test_nsfw_folder_checker_all(
        self,
        interrogator_no_blip: Interrogator,
        deep_danbooru_model: DeepDanbooruModel,
        nsfw_pilot_folders: list[Path],
        sfw_pilot_folders: list[Path],
        neutral_pilot_folders: list[Path],
        inappropriate_pilot_folders: list[Path],
        nsfw_folders: list[Path],
        sfw_folders: list[Path],
        lewd_folders: list[Path],
        inappropriate_folders: list[Path],
        neutral_folders: list[Path],
    ):
        nsfw_folder_checker = NSFWFolderChecker(
            interrogator_no_blip,
            deep_danbooru_model,
            nsfw_pilot_folders=nsfw_pilot_folders,
            sfw_pilot_folders=sfw_pilot_folders,
            neutral_pilot_folders=neutral_pilot_folders,
            inappropriate_pilot_folders=inappropriate_pilot_folders,
            nsfw_folders=nsfw_folders,
            sfw_folders=sfw_folders,
            lewd_folders=lewd_folders,
            inappropriate_folders=inappropriate_folders,
            neutral_folders=neutral_folders,
        )

        all_folders_to_check = nsfw_folder_checker.get_all_folders_to_check()

        assert len(all_folders_to_check) == (
            len(nsfw_pilot_folders)
            + len(sfw_pilot_folders)
            + len(neutral_pilot_folders)
            + len(inappropriate_pilot_folders)
            + len(nsfw_folders)
            + len(sfw_folders)
            + len(lewd_folders)
            + len(inappropriate_folders)
            + len(neutral_folders)
        )

        for folder, nsfw_type_expected in all_folders_to_check:
            assert folder is not None
            assert nsfw_type_expected is not None

            folder_result = nsfw_folder_checker.check_folder(folder, nsfw_type_expected)

            assert folder_result is not None
            self.write_results(folder_result)

    def test_nsfw_folder_checker_all_folders_sampled(
        self,
        interrogator_no_blip: Interrogator,
        deep_danbooru_model: DeepDanbooruModel,
        nsfw_pilot_folders: list[Path],
        sfw_pilot_folders: list[Path],
        nsfw_folders: list[Path],
        sfw_folders: list[Path],
        neutral_pilot_folders: list[Path],
        inappropriate_pilot_folders: list[Path],
        lewd_folders: list[Path],
        inappropriate_folders: list[Path],
        neutral_folders: list[Path],
    ):
        nsfw_folder_checker = NSFWFolderChecker(
            interrogator_no_blip,
            deep_danbooru_model=deep_danbooru_model,
            nsfw_pilot_folders=nsfw_pilot_folders,
            sfw_pilot_folders=sfw_pilot_folders,
            neutral_pilot_folders=neutral_pilot_folders,
            inappropriate_pilot_folders=inappropriate_pilot_folders,
            nsfw_folders=nsfw_folders,
            sfw_folders=sfw_folders,
            lewd_folders=lewd_folders,
            inappropriate_folders=inappropriate_folders,
            neutral_folders=neutral_folders,
        )

        all_folders_to_check = nsfw_folder_checker.get_all_folders_to_check()

        # all_folders_to_check = [folder for folder in all_folders_to_check if "anime" not in folder[0].name]  # FIXME

        for folder, nsfw_type_expected in all_folders_to_check:
            assert folder is not None
            assert nsfw_type_expected is not None

            folder_result = nsfw_folder_checker.check_folder(folder, nsfw_type_expected, sample_size=1000)

            assert folder_result is not None
            self.write_results(folder_result)

    def write_results(self, folder_result: NSFWFolderResults):
        time_str = time.strftime("%Y%m%d-%H%M%S")
        results_dict = {
            "folder": str(folder_result.folder_path),
            "nsfw_type_expected": folder_result.nsfw_type_excepted.name,
            "num_images": len(folder_result.all_results),
            "num_nsfw_images": folder_result.num_nsfw_images,
            "num_anime_images": folder_result.num_anime_images,
            "num_human_subject_images": folder_result.num_human_subject_images,
        }

        def default_serializer(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, Enum):
                return obj.value
            return obj.__dict__

        with open(f"results/{folder_result.folder_path.name}_{time_str}.json", "w") as f:
            results_dict["all_results"] = {
                str(image_path): {
                    **result.model_dump(),
                }
                for image_path, result, in folder_result.all_results
                if result is not None
            }

            f.write(json.dumps(results_dict, indent=4, default=default_serializer))

        with open(f"results/{folder_result.folder_path.name}_unexpected_{time_str}.json", "w") as f:
            del results_dict["all_results"]
            unexpected_results = {
                str(image_path): {
                    **result.model_dump(),
                }
                for image_path, result, in folder_result.get_unexpected_results()
                if result is not None
            }

            if len(unexpected_results) > 0:
                results_dict["all_unexpected_results"] = unexpected_results
                f.write(json.dumps(results_dict, indent=4, default=default_serializer))

        with open(f"results/{folder_result.folder_path.name}_csam_{time_str}.json", "w") as f:
            if "all_unexpected_results" in results_dict:
                del results_dict["all_unexpected_results"]
            unexpected_results = {
                str(image_path): {
                    **result.model_dump(),
                }
                for image_path, result, in folder_result.all_results
                if result is not None and result.is_csam
            }

            if len(unexpected_results) > 0:
                results_dict["all_csam_results"] = unexpected_results
                f.write(json.dumps(results_dict, indent=4, default=default_serializer))
