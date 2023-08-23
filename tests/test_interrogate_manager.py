from horde_safety.interrogate import get_interrogator, get_interrogator_no_blip


def test_get_interrogator_no_blip():
    interrogator = get_interrogator_no_blip()
    assert interrogator is not None
    assert not hasattr(interrogator, "caption_model")


def test_get_interrogator():
    interrogator = get_interrogator()
    assert interrogator is not None
    assert interrogator.caption_model is not None
