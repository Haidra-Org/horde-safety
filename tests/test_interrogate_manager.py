from horde_safety.interrogate import HordeInterrogateManager


def test_get_interrogator_no_blip():
    manager = HordeInterrogateManager()
    interrogator = manager.get_interrogator_no_blip()
    assert interrogator is not None
    assert not hasattr(interrogator, "caption_model")


def test_get_interrogator():
    manager = HordeInterrogateManager()
    interrogator = manager.get_interrogator()
    assert interrogator is not None
    assert interrogator.caption_model is not None
