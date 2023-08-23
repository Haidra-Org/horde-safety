# horde-safety
Provides safety features used by the horde, especially to do with image generation.

## Note

This library is made with the default AI Horde worker in mind, and relies on the environment variable `AIWORKER_CACHE_HOME` to establish isolation of the clip models on disk. If you do not want to rely on a horde specific folder structure, define `TRANSFORMERS_CACHE` to define where you'd prefer the models to be. If neither are defined, the default huggingface folder location for the system used, typically `~/.cache`, depending on other environnement variables, [see the official huggingface docs](https://huggingface.co/docs/transformers/installation#cache-setup) for more info.

## Installing

Make sure pytorch is installed, preferably with CUDA/ROCM support.

```bash
    pip install horde_safety
```

## Use

This library currently relies on [clip_interrogator](https://github.com/pharmapsychotic/clip-interrogator). The `check_for_csam` function requires an instance of `clip_interrogator.Interrogator` to be passed. You can pass in on yourself, or use the helper function `get_interrogator_no_blip` (note that calling this function immediately loads the CLIP model).

```python
    import PIL.Image

    from horde_safety.csam_checker import check_for_csam
    from horde_safety.interrogate import get_interrogator_no_blip

    interrogator = get_interrogator_no_blip()
    image: PIL.Image.Image
    prompt: str

    (...)

    is_csam, results, info = check_for_csam(
        interrogator=interrogator,
        image=image,
        prompt=prompt,
        model_info={"nsfw": True, "tags": []},
        # model_info can be found at https://github.com/Haidra-Org/AI-Horde-image-model-reference/
    )

    if is_csam:
        reject_job()

```

If you reject a job as a horde worker for CSAM, you should report `'state': 'csam'` in the generate submit payload.
