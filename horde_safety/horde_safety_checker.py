from clip_interrogator import Interrogator  # type: ignore

from horde_safety.csam_checker import check_for_csam


class HordeSafetyChecker:
    def __init__(self, interrogator: Interrogator):
        self.interrogator = interrogator

    def check_for_csam(self):
        check_for_csam(self.interrogator)
