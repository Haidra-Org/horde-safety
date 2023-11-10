from pathlib import Path

import PIL.Image
from loguru import logger
from tqdm import tqdm

from horde_safety.csam_checker import check_for_csam
from horde_safety.deep_danbooru_model import get_deep_danbooru_model
from horde_safety.interrogate import get_interrogator_no_blip
from horde_safety.nsfw_checker_class import NSFWChecker
from horde_safety.utils import get_image_file_paths


def test_check_for_csam_pilot(pilot_folders: list[Path]):
    interrogator = get_interrogator_no_blip()
    deep_danbooru_model = get_deep_danbooru_model()

    nsfw_checker = NSFWChecker(interrogator, deep_danbooru_model)

    for source_image_folder in pilot_folders:
        if not source_image_folder.exists():
            raise Exception(f"Please create the folder {source_image_folder} and add some images to it")

        image_paths = get_image_file_paths(source_image_folder)

        num_csam = 0

        progress_bar = tqdm(total=len(image_paths), desc="Checking images")
        # Iterate through all the images in the folder
        for image_path in image_paths:
            image = PIL.Image.open(image_path)
            # logger.info(f"Checking {image_path}")

            try:
                is_csam, results, info = check_for_csam(
                    interrogator=interrogator,
                    image=image,
                    prompt="",
                    model_info={"nsfw": False, "tags": []},
                    nsfw_checker=nsfw_checker,
                )
                if is_csam:
                    # print(f"CSAM found in {image_path}")
                    num_csam += 1
                    # with open(f"{image_path}.json", "w") as f:
                    #     json.dump(info, f, indent=4)
                    #     f.write("\n")
                    #     json.dump(results, f, indent=4)
                    #     f.write("\n")
                    #     json.dump(nsfw_anime_result.tag_results, f, indent=4)
                # if not is_csam:
                #     print(f"CSAM not found in {image_path}")
                #     with open(f"{image_path}.json", "w") as f:
                #         json.dump(info, f, indent=4)
                #         f.write("\n")
                #         json.dump(results, f, indent=4)
                #         f.write("\n")
                #         json.dump(nsfw_anime_result.tag_results, f, indent=4)
            except Exception as e:
                logger.error(f"Error checking {image_path}: {e}")
                raise e

            progress_bar.update(1)
            progress_bar.desc = (
                f"({source_image_folder.stem}) num_csam={num_csam} ({num_csam / progress_bar.n * 100:.2f}%)"
            )
