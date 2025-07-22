# horde-safety

Provides safety features used by the horde, especially to do with image generation.

## Environment Variables

This library is made with the default AI Horde worker in mind, and relies on the environment variable `AIWORKER_CACHE_HOME` to establish isolation of the clip models on disk. If you do not want to rely on a horde specific folder structure, define `HF_HOME` to define where you'd prefer the models to be. If neither are defined, the default huggingface folder location for the system used, typically `~/.cache`, depending on other environnement variables, [see the official huggingface docs](https://huggingface.co/docs/transformers/installation#cache-setup) for more info.

> Warning: If you use `AIWORKER_CACHE_HOME`, other environment variables can and will be overridden, including `HF_HOME` and potentially any other related variables. Using `AIWORKER_CACHE_HOME` is explictly opting into the horde isolation scheme, which may not be suitable for other contexts and may lead to duplicate (and very large) models on disk. If you want to use this library outside of the horde, it is recommended to set `HF_HOME` instead.

## Installing

Make sure pytorch is installed, preferably with CUDA/ROCM or other GPU support.

```bash
pip install horde_safety
```

## Use

This library currently relies on [clip_interrogator](https://github.com/pharmapsychotic/clip-interrogator). The `check_for_csam` function requires an instance of `clip_interrogator.Interrogator` to be passed. You can pass in on yourself, or use the helper function `get_interrogator_no_blip` (note that calling this function immediately loads the CLIP model). Use the `device` parameter to choose the device to load to and use for interrogation. If you want to only use the CPU but have CUDA pytorch installed, use `get_interrogator_no_blip(device="cpu")`.

```python
import PIL.Image

from horde_safety.deep_danbooru_model import get_deep_danbooru_model
from horde_safety.interrogate import get_interrogator_no_blip
from horde_safety.nsfw_checker_class import NSFWChecker, NSFWResult

interrogator = get_interrogator_no_blip() # Will trigger a download if not on disk (~1.2 gb)
deep_danbooru_model = get_deep_danbooru_model() # Will trigger a download if not on disk (~614 mb)

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


```

If you reject a job as a horde worker for CSAM, you should report `'state': 'csam'` in the generate submit payload.
