import PIL.Image

from horde_safety.deep_danbooru_model import get_deep_danbooru_model
from horde_safety.interrogate import get_interrogator_no_blip
from horde_safety.nsfw_checker_class import NSFWChecker, NSFWResult

interrogator = get_interrogator_no_blip()
deep_danbooru_model = get_deep_danbooru_model()

nsfw_checker = NSFWChecker(
    interrogator,
    deep_danbooru_model,  # Optional, significantly improves results for anime images
)

image: PIL.Image.Image = PIL.Image.open("image.jpg")
prompt: str | None = None  # if this is an image generation, you can provide the prompt here
model_info: dict | None = None  # if this is an image generation, you can provide the model info here

nsfw_result: NSFWResult | None = nsfw_checker.check_for_nsfw(image, prompt=prompt, model_info=model_info)

if nsfw_result is None:
    print("No NSFW result (Did it fail to load the image?)")
    exit(1)


if nsfw_result.is_anime:
    print("Anime detected!")

if nsfw_result.is_nsfw:
    print("NSFW detected!")

if nsfw_result.is_csam:
    print("CSAM detected!")
    exit(1)
