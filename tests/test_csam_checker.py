import json
import os
from pathlib import Path

import PIL.Image
from loguru import logger

from horde_safety.csam_checker import check_for_csam
from horde_safety.interrogate import get_interrogator_no_blip


def test_check_for_csam():
    test_folder = os.getenv("HORDE_SAFETY_TEST_FOLDER")

    if test_folder is None:
        raise Exception("Please set the environment variable HORDE_SAFETY_TEST_FOLDER")

    source_image_folder = Path(test_folder)

    if not source_image_folder.exists():
        raise Exception(f"Please create the folder {source_image_folder} and add some images to it")

    image_paths = list(source_image_folder.glob("*.jpg"))
    image_paths.extend(list(source_image_folder.glob("*.png")))

    interrogator = get_interrogator_no_blip()
    # Iterate through all the images in the folder
    for image_path in image_paths:
        image = PIL.Image.open(image_path)
        logger.info(f"Checking {image_path}")
        is_csam, results, info = check_for_csam(
            interrogator=interrogator,
            image=image,
            prompt="",
            model_info={"nsfw": True, "tags": []},
        )
        if is_csam:
            print(json.dumps(info, indent=4))
        del image
        print()
