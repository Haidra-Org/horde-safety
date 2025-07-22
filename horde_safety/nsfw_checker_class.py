from contextlib import nullcontext, redirect_stderr, redirect_stdout
from enum import Enum, auto
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

from horde_safety.deep_danbooru_model import DeepDanbooruModel
from horde_safety.interrogate import Interrogator
from horde_safety.utils import get_image_file_paths, normalize_prompt

HUMAN_SUBJECT_KEY = "human_subject"


class NSFWResult(BaseModel):
    """The result of a NSFW check."""

    all_predicate_scores: dict[str, float]
    matched_predicates: dict[str, float]
    nsfw_similarity_results: dict[str, float]
    is_nsfw: bool
    nsfw_concepts_identified: dict[str, float]
    anime_concepts_identified: dict[str, float]
    is_anime: bool

    is_neg_prompt_suspect: bool = False

    is_underage_subject: bool

    is_csam: bool
    csam_predicates_matched: dict[str, float]

    @property
    def found_human_subject(self) -> bool:
        """Returns `True` if the image contains a human subject, `False` otherwise."""
        return len(self.matched_predicates) > 0


class NSFWAnimeScores:
    safe_score: float
    """The score for the safe rating. This is based on a particular sensibility and is not one that
    would be universally agreed upon."""
    safe_score_key: str = "rating:safe"
    """The deepdanbooru specific key for the safe score."""
    safe_score_threshold: float = 0.5
    """The threshold for the safe score. If the safe score is below this, the image is considered NSFW."""

    questionable_score: float
    """A score which indicates the likelihood that the image could be considered risque."""
    questionable_score_key: str = "rating:questionable"
    """The deepdanbooru specific key for the questionable score."""
    questionable_score_threshold: float = 0.185
    """The threshold for the questionable score. If the questionable score is above this, the image is considered
    NSFW."""

    explicit_score: float
    """A score which indicates the likelihood that the image could be considered sexually explicit."""
    explicit_score_key: str = "rating:explicit"
    """The deepdanbooru specific key for the explicit score."""
    explicit_score_threshold: float = 0.125
    """The threshold for the explicit score. If the explicit score is above this, the image is considered NSFW.
    Any value more than a few percent tends to be accurate.
    """

    nsfw_tags: list[str] = []
    """The tags which indicate the image is NSFW."""

    @property
    def nsfw_based_on_tags(self) -> bool:
        """Return `True` if the image is considered NSFW based on the tags, `False` otherwise.
        This does not consider the meta `rating:*` tags."""

        return len(self.nsfw_tags) > 0

    @property
    def is_anime_nsfw(self) -> bool:
        """Return `True` if the image is considered NSFW based on the deepdanbooru model alone, `False` otherwise."""
        if self.nsfw_based_on_tags:
            return True

        if self.explicit_score > self.explicit_score_threshold:
            return True

        if self.questionable_score > self.questionable_score_threshold and self.safe_score < 0.93:
            if self.safe_score >= 0.93:
                pass
            return True

        if self.safe_score < self.safe_score_threshold:
            return True

        return False

    def __init__(
        self,
        safe_score: float,
        questionable_score: float,
        explicit_score: float,
        nsfw_tags: list[str],
    ) -> None:
        """Instantiate an instance of the NSFWAnimeScores class.

        Args:
            safe_score (float): The deepdanbooru safe score.
            questionable_score (float): The deepdanbooru questionable score.
            explicit_score (float): The deepdanbooru explicit score.
            nsfw_tags (list[str]): The tags which indicate the image is NSFW, not including the meta `rating:*` tags.
        """
        self.safe_score = safe_score
        self.questionable_score = questionable_score
        self.explicit_score = explicit_score
        self.nsfw_tags = nsfw_tags

    def to_dict(self) -> dict[str, float]:
        return {
            self.safe_score_key: self.safe_score,
            self.questionable_score_key: self.questionable_score,
            self.explicit_score_key: self.explicit_score,
        }


class NSFWAnimeResult:
    tag_results: dict[str, float]
    """The results of the danbooru tag check."""
    nsfw_anime_scores: NSFWAnimeScores
    """The scores from the deepdanbooru model represented by a `NSFWAnimeScores` object."""

    @property
    def is_anime_nsfw(self) -> bool:
        return self.nsfw_anime_scores.is_anime_nsfw

    def __init__(
        self,
        tag_results: dict[str, float],
        nsfw_anime_scores: NSFWAnimeScores,
    ) -> None:
        """Instantiate an instance of the NSFWAnimeResult class.

        Args:
            tag_results (dict[str, float]): The results of the danbooru tag check.
            nsfw_anime_scores (NSFWAnimeScores): The scores from the deepdanbooru model represented by a
                `NSFWAnimeScores` object.
        """
        self.tag_results = tag_results
        self.nsfw_anime_scores = nsfw_anime_scores


class NSFWFolderResultTypeExpected(Enum):
    nsfw = auto()
    """90% or more of the images are NSFW."""
    sfw = auto()
    """90% or more of the images are SFW."""
    mostly_nsfw = auto()
    """80% or more of the images are NSFW."""
    mostly_sfw = auto()
    """80% or more of the images are SFW."""

    neutral = auto()
    """The folder is neutral, and potentially contains a mix of NSFW and SFW images."""


class NSFWFolderResults:
    folder_path: Path
    """The path to the folder."""
    nsfw_type_excepted: NSFWFolderResultTypeExpected
    """The type of NSFW folder expected."""

    @property
    def expecting_nsfw(self) -> bool:
        """Return `True` if the folder is expected to be NSFW, `False` otherwise."""
        return self.nsfw_type_excepted in [NSFWFolderResultTypeExpected.nsfw, NSFWFolderResultTypeExpected.mostly_nsfw]

    all_results: list[tuple[Path, NSFWResult | None]]
    """All final evaluation results for the folder."""

    def add_result(self, path: Path, result: NSFWResult | None) -> None:
        """Add a result to the folder results, keeping track of the number of NSFW images, etc."""
        self.all_results.append((path, result))

        if result is not None:
            if result.is_nsfw:
                self.num_nsfw_images += 1
            if result.found_human_subject:
                self.num_human_subject_images += 1
            if result.is_underage_subject:
                self.num_underage_subject_images += 1
            if result.is_anime:
                self.num_anime_images += 1
            if result.is_csam:
                self.num_csam_images += 1

    def get_unexpected_results(self) -> list[tuple[Path, NSFWResult | None]]:
        """Return a list of results which are not expected given the `nsfw_type_expected`."""
        if len(self.all_results) == 0:
            return []

        if self.nsfw_type_excepted == NSFWFolderResultTypeExpected.neutral:
            return []

        return [
            (path, result)
            for path, result in self.all_results
            if result is not None and result.is_nsfw != self.expecting_nsfw
        ]

    num_nsfw_images: int = 0
    """The number of probable NSFW images in the folder."""
    num_anime_images: int = 0
    """The number of probable cartoon/anime images in the folder."""
    num_human_subject_images: int = 0
    """The number of images which probably contain a human subject."""
    num_underage_subject_images: int = 0
    """The number of images which probably contain an underage subject."""
    num_csam_images: int = 0
    """The number of potential CSAM images."""

    def __init__(
        self,
        folder_path: Path,
        nsfw_type_excepted: NSFWFolderResultTypeExpected,
    ) -> None:
        """Instantiate an instance of the NSFWFolderResults class."""

        self.folder_path = folder_path
        self.nsfw_type_excepted = nsfw_type_excepted

        self.all_results = []

    @property
    def percent_nsfw(self) -> float:
        """Return the percentage of NSFW images in the folder."""
        if len(self.all_results) == 0:
            return 0

        return self.num_nsfw_images / len(self.all_results) * 100

    @property
    def percent_not_nsfw(self) -> float:
        """Return the percentage of non-NSFW images in the folder."""
        if len(self.all_results) == 0:
            return 0

        return 100 - self.percent_nsfw

    @property
    def percent_csam(self) -> float:
        """Return the percentage of CSAM images in the folder."""
        if len(self.all_results) == 0:
            return 0

        return self.num_csam_images / len(self.all_results) * 100

    def __repr__(self) -> str:
        val = (
            f"{self.folder_path.stem} (is nsfw {self.expecting_nsfw}) ({len(self.all_results)}): "
            f"{self.percent_nsfw:.2f}% nsfw ({self.num_nsfw_images}), "
            f"{self.percent_not_nsfw:.2f}% not-nsfw ({len(self.all_results) - self.num_nsfw_images}) "
            f"(anime: {self.num_anime_images}, human subjects: {self.num_human_subject_images}, "
            f"underage: {self.num_underage_subject_images}) "
            f"csam {self.percent_csam:.2f}% ({self.num_csam_images})) "
        )

        return val


class NSFWChecker:
    interrogator: Interrogator
    """The interrogator to use for safety checks."""
    deep_danbooru_model: DeepDanbooruModel | None
    """The deepdanbooru model to use for anime safety checks."""

    image_description_prefixes: dict[str, float]
    """The prefixes to combine with parts to create image descriptions.
    These include `photo of`, `painting of`, etc."""

    human_predicate_parts: dict[str, float]
    """The parts to combined with prefixes to create human predicates."""

    human_predicates_isolated: dict[str, float]
    """Predicates for determining a human subject which do not need to be combined with a base."""
    underage_predicates: dict[str, float]
    """Predicates for determining if a human subject is underage."""

    underage_danger_concepts: dict[str, float]
    """The thresholds for underage predicates which should automatically flag the image as CSAM."""
    underage_danger_modifiers: dict[str, float]
    """Modifiers for underage predicates which, when combined, should automatically flag the image as CSAM."""

    underage_nsfw_predicates: dict[str, float]
    """Predicates for determining if a human subject is underage and in a potentially NSFW situation."""
    underage_nsfw_danger_modifiers: dict[str, float]
    """Modifiers for underage predicates which, when combined, should automatically flag the image as CSAM."""

    all_human_predicates: dict[str, float]
    """The human predicates, including the isolated ones and the parts combined with prefixes."""

    nsfw_predicates: dict[str, float]
    """Predicates which rely only on a NSFW clip token."""
    nsfw_synth_predicates: dict[str, float]
    """Synthetic predicates using non-natural language tokens."""

    anime_cartoon_predicates: dict[str, float]

    _all_predicates: dict[str, float]
    """All predicates, including nsfw, synth, underage and anime/cartoon predicates."""
    _all_predicate_names_only: list[str]  # This is kept as a known object as an optimization for the similarities call
    """All predicate names, including nsfw, synth, underage and anime/cartoon predicates."""

    # `_all_predicates` is dynamically generated, and the refresh_* calls should be used to update it.
    @property
    def all_predicates(self) -> dict[str, float]:
        """All predicates, including nsfw, synth, underage and anime/cartoon predicates."""
        return self._all_predicates.copy()

    nsfw_isolated_concepts: dict[str, float]
    """NSFW concepts which do not need to be combined with a base."""
    nsfw_concept_parts: dict[str, float]
    """The parts to combined with prefixes to create nsfw concepts."""
    nsfw_synth_concepts: dict[str, float]
    """Synthetic nsfw concepts using non-natural language tokens."""

    _all_nsfw_concepts: dict[str, float]
    """All nsfw concepts, including the isolated concepts, the combined concepts and the synthetic ones."""
    _all_nsfw_concept_names_only: list[
        str
    ]  # This is kept as a known object as an optimization for the similarities call
    """All nsfw concept names, including the isolated concepts, the combined concepts and the synthetic ones."""

    @property
    def all_nsfw_concepts(self) -> dict[str, float]:
        """All nsfw concepts, including the isolated concepts, the combined concepts and the synthetic ones."""
        return self._all_nsfw_concepts.copy()

    predicate_default_threshold: float = 0.202
    """The default threshold for any predicates."""

    resize_targets: list[tuple[int, int]]
    """The targets to resize an anime image to before checking. The evaluation model only support certain sizes."""

    nsfw_deepdanbooru_concepts: dict[str, float]
    """Danbooru tags which add to the NSFW score."""
    nsfw_post_anime_concepts: dict[str, float]
    """These concepts override the non-anime weights usually used. This is to account for the inaccuracy of some
    concepts with anime (keeping in mind they are accurate for realistic/photos)."""

    def __init__(
        self,
        interrogator: Interrogator,
        deep_danbooru_model: DeepDanbooruModel | None,
    ) -> None:
        """Instantiate an instance of the NSFWChecker class.

        If you intend to change the predicates or concepts, you should call `.refresh()` after changing them.

        Args:
            interrogator (Interrogator): The interrogator to use for the check.
            deep_danbooru_model (DeepDanbooruModel): The deepdanbooru model to use for the check.

        """
        self.interrogator = interrogator
        self.deep_danbooru_model = deep_danbooru_model
        self.resize_targets = []
        for i in range(128, 769, 128):
            for j in range(128, 769, 128):
                if i == j:
                    self.resize_targets.append((i, j))
                else:
                    self.resize_targets.append((i, j))
                    self.resize_targets.append((j, i))

        self._set_default_predicates()
        self._refresh_predicates()
        self._set_default_concepts()
        self._refresh_concepts()

    def refresh(self) -> None:
        """Refresh the internal state to use any changed predicates or concepts."""
        self._refresh_predicates()
        self._refresh_concepts()

    def _set_default_predicates(self) -> None:
        self.image_description_prefixes = {
            "photo of": 0,
            "painting of": 0.005,
            "illustration of": 0.075,
            "3d render of": 0.0125,
        }

        self.human_predicate_parts = {
            "a boy": 0,
            "boys": 0,
            "a girl": 0,
            "girls": 0,
            "a teen": 0,
            "a man": 0.01,
            "men": 0,
            "a woman": 0,
            "a couple": 0,
            "couple kissing": 0,
            "a girl in a bikini": 0.01,
            "girls in bikinis": 0.01,
        }

        self.human_predicates_isolated = {
            "boy": 0.194,
            "boys": 0.194,
            "man": 0.194,
            "men": 0.194,
            "girl": 0.194,
            "girls": 0.194,
            "woman": 0.194,
            "women": 0.194,
            "teenager": 0.194,
            "teen": 0.19,
            "teens": 0.19,
        }

        self.underage_predicates = {
            "child": 0.19,
            "children": 0.19,
            "young children": 0.189,
            "a baby": 0.19,
            "an infant": 0.19,
            "infants": 0.19,
            "toddler": 0.19,
            "toddlers": 0.19,
            "tween": 0.19,
            "tweens": 0.19,
            "lolicon": 0.20,
            # "shota": 0.23,
        }

        self.underage_danger_modifiers = {
            "child": 0.035,
            "children": 0.035,
            "young children": 0.03,
            "a baby": 0.0275,
            "an infant": 0.0275,
            "infants": 0.0275,
            "toddler": 0.0275,
            "toddlers": 0.0275,
            "tween": 0.045,
            "tweens": 0.045,
            "lolicon": 0.044,
            # "shota": 0.01,
        }
        self.underage_danger_concepts = {}

        for predicate, predicate_value in self.underage_danger_modifiers.items():
            self.underage_danger_concepts[predicate] = predicate_value + self.underage_predicates[predicate]

        self.underage_nsfw_predicates = {
            # currently not used
        }

        self.underage_nsfw_danger_modifiers = {
            # currently not used
        }

        self.nsfw_predicates = {
            "18+": 0.20,
            "explicit content": 0.21,
            "erotica": 0.21,
            "exposed erection": 0.215,
            "close up of a vagina": 0.24,
            "photo of vagina": 0.22,
            "porn scene": 0.21,
            "lesbian scene": 0.22,
            "gay scene": 0.22,
        }

        self.nsfw_predicates.update(self.underage_nsfw_predicates)

        for predicate in self.underage_nsfw_predicates:
            self.underage_danger_modifiers[predicate] = (
                self.underage_nsfw_danger_modifiers[predicate] + self.underage_nsfw_predicates[predicate]
            )

        self.nsfw_synth_predicates = {
            # "quintesstain corinblazing dean n blondes bronxsmug demi pilebarcelonluisspandex ,... ...#": 0.22,
            # "liam paying awwww hubby ... feeding mr " " omg dropbox brad(@ zaynwoo ….. burr": 0.22,
            # "nel ov lexi lov thwatchin thalaireason zz big oj anticipation บci ov prick": 0.22,
            # "somehow dang rochelle sizzgusta imrankhanhalftime rankings equities exists skynews "
            # "-- - smokin epidemiq": 0.22,
            # "asses zeit brunette brunette bikini meatlessdesired androunprecearomatic limestone "
            # "juxtatweetapicturevalmelee photo": 0.22,
        }

        self.anime_cartoon_predicates = {
            "japanese animation": 0.20,
            "Japanese cartoon": 0.20,
            "anime": 0.20,
            "cartoon": 0.20,
            "manga": 0.205,
        }

    _combined_underage_danger_concepts: dict[str, float]

    def _refresh_predicates(self) -> None:
        """Refresh the internal state to use any changed predicates.

        This includes, but may not be limited to the following fields:
        - all_human_predicates
        - nsfw_predicates
        - nsfw_synth_predicates
        - anime_cartoon_predicates
        - image_description_prefixes
        - underage_predicates
        """
        self._nsfw_combined_predicates = {}
        self._nsfw_combined_predicates.update(self.nsfw_predicates)

        self._combined_underage_predicates = {}
        self._combined_underage_predicates.update(self.underage_predicates)

        self._combined_underage_danger_concepts = {}
        self._combined_underage_danger_concepts.update(self.underage_danger_concepts)

        self.all_human_predicates = self.human_predicates_isolated.copy()
        for prefix, prefix_value in self.image_description_prefixes.items():
            for part, part_value in self.human_predicate_parts.items():
                self.all_human_predicates[f"{prefix} {part}"] = (
                    self.predicate_default_threshold + prefix_value + part_value
                )

            for nsfw_predicate, nsfw_predicate_value in self.nsfw_predicates.items():
                self._nsfw_combined_predicates[f"{prefix} {nsfw_predicate}"] = prefix_value + nsfw_predicate_value

            for danger_predicate, danger_predicate_value in self.underage_danger_concepts.items():
                if danger_predicate == "lolicon":
                    continue

                self._combined_underage_danger_concepts[f"{prefix} {danger_predicate}"] = (
                    prefix_value + danger_predicate_value
                )
                self._combined_underage_predicates[f"{prefix} {danger_predicate}"] = (
                    prefix_value + self.underage_predicates[danger_predicate]
                )

        total_expected_length = (
            len(self.anime_cartoon_predicates)
            + len(self.all_human_predicates)
            + len(self._nsfw_combined_predicates)
            + len(self.nsfw_synth_predicates)
            + len(self._combined_underage_predicates)
        )

        self._all_predicates = {
            **self.anime_cartoon_predicates,
            **self.all_human_predicates,
            **self._combined_underage_predicates,
            **self._nsfw_combined_predicates,
            **self.nsfw_synth_predicates,
        }

        if len(self._all_predicates) != total_expected_length:
            found_keys = []
            dicts: list[dict[str, float]] = [
                self.anime_cartoon_predicates,
                self.all_human_predicates,
                self._combined_underage_predicates,
                self._nsfw_combined_predicates,
                self.nsfw_synth_predicates,
            ]

            for d in dicts:
                for k, v in d.items():
                    if k in found_keys and k not in self.underage_predicates:
                        # raise ValueError(f"Found duplicate key {k} in predicates")
                        logger.warning(f"Found duplicate key {k} in predicates")
                    found_keys.append(k)

            # raise ValueError(
            #     f"Expected {total_expected_length} predicates, but got {len(self._all_predicates)}. "
            #     f"Check for duplicate keys.",
            # )

        self._all_predicate_names_only = list(self._all_predicates.keys())

        logger.debug(f"Num predicates: {len(self._all_predicates)}")

    def _set_default_concepts(self) -> None:
        self.nsfw_isolated_concepts = {
            "sexual": 0.2,
            "sexy": 0.22,
            "uncensored": 0.21,
            "naked": 0.22,
            "nude": 0.22,
            "nsfw": 0.21,
            "nudity": 0.21,
            "erotic": 0.21,
            "fucking": 0.21,
        }

        self.nsfw_concept_parts = {
            "explicit content": 0.20,
            "exposed erection": 0.21,
            "close up of a vagina": 0.24,
            "a womans legs spread, vagina exposed": 0.23,
            "exposed vagina": 0.22,
            "erotica": 0.21,
            "blowjob": 0.22,
            "vagina": 0.21,
            "an erection": 0.21,
            "porn": 0.2,
            "porn scene": 0.20,
            "lesbian scene": 0.22,
            "gay scene": 0.22,
            "a nude man's ass": 0.21,
            "a nude woman's ass": 0.21,
            # "nude child": 0.22,
            # "nude girl": 0.23,
            # "nude girls": 0.23,
            # "nude boy": 0.23,
            # "nude boys": 0.23,
            # "nude man": 0.23,
            # "nude men": 0.23,
            # "nude woman": 0.23,
            # "nude women": 0.23,
            "exposed breasts": 0.22,
            "girl in a bikini": 0.22,
            "underwear": 0.21,
            "lingerie": 0.21,
        }

        synth1 = (
            "somehow dang rochelle sizzgusta imrankhanhalftime rankings equities exists skynews -- - smokin epidemiq"
        )

        synth2 = "asses zeit brunette brunette bikini meatlessdesired androunprecearomatic limestone "
        synth2 += "juxtatweetapicturevalmelee photo"

        self.nsfw_synth_concepts = {
            "quintesstain corinblazing dean n blondes bronxsmug demi pilebarcelonluisspandex ,... ...#": 0.205,
            "liam paying awwww hubby ... feeding mr " " omg dropbox brad(@ zaynwoo ….. burr": 0.205,
            "nel ov lexi lov thwatchin thalaireason zz big oj anticipation บci ov prick": 0.205,
            synth1: 0.205,
            synth2: 0.205,
        }

        self.nsfw_deepdanbooru_concepts = {
            "bikini": 0.6,
        }

        self.nsfw_post_anime_concepts = {
            "nsfw": 0.21,
            "18+": 0.215,
            "nipples": 0.20,
            "nipple": 0.20,
            "naked child": 0.19,
            "naked breasts": 0.205,
            "illustration of naked breasts": 0.205,
            "exposed breasts": 0.205,
            "illustration of exposed breasts": 0.205,
            "nudity": 0.195,
            "nude": 0.195,
            "naked": 0.22,
            "cartoon porn": 0.20,
            "sex": 0.21,
        }

    def _refresh_concepts(self) -> None:
        """Refresh the internal state to use any changed concepts.

        This includes, but may not be limited  to the following fields:
        - nsfw_isolated_concepts
        - nsfw_concept_parts
        - nsfw_synth_concepts
        - image_description_prefixes
        """
        nsfw_prefix_parts_combined = {}

        for prefix, prefix_value in self.image_description_prefixes.items():
            for part, part_value in self.nsfw_concept_parts.items():
                nsfw_prefix_parts_combined[f"{prefix} {part}"] = prefix_value + part_value

        self._all_nsfw_concepts = {
            **self.nsfw_isolated_concepts,
            **self.nsfw_concept_parts,
            **nsfw_prefix_parts_combined,
            **self.nsfw_synth_concepts,
        }

        self._all_nsfw_concept_names_only = list(self._all_nsfw_concepts.keys())

    def get_nsfw_concept_similarities(self, image: PIL.Image.Image) -> dict[str, float]:
        """Get NSFW concept similarities for the given image."""
        similarity_results = self.interrogator.similarities(
            image_features=self.interrogator.image_to_features(image),
            text_array=self._all_nsfw_concept_names_only,
        )
        return {concept: round(float(value), 4) for concept, value in zip(self._all_nsfw_concepts, similarity_results)}

    def get_predicate_similarities(self, image: PIL.Image.Image | torch.Tensor) -> dict[str, float]:
        """Get predicate similarities for the given image."""
        if isinstance(image, PIL.Image.Image):
            similarity_results = self.interrogator.similarities(
                image_features=self.interrogator.image_to_features(image),
                text_array=self._all_predicate_names_only,
            )
        elif isinstance(image, torch.Tensor):
            similarity_results = self.interrogator.similarities(
                image_features=image.unsqueeze(0),
                text_array=self._all_predicate_names_only,
            )
        else:
            raise ValueError("Image must be a PIL Image or a torch Tensor.")

        return {
            predicate: round(float(value), 4)
            for predicate, value in zip(self._all_predicates.keys(), similarity_results)
        }

    def check_for_nsfw(
        self,
        image: PIL.Image.Image,
        prompt: str | None = None,
        model_info: dict | None = None,
        *,
        image_tensor: torch.Tensor | None = None,
    ) -> NSFWResult | None:
        """Checks if an image is potentially NSFW. It is known to have false positive and false negative results.

        Args:
            interrogator (Interrogator): The interrogator to use for the check.
            image (PIL.Image.Image): The image to check.
            prompt (str): The prompt used to create the image.
            model_info (dict, optional): The entry from the model reference for the model used
            to create the image. Defaults to `None`.

        Returns:
            NSFWResult | None: The result of the check, or `None` if the check failed.
        """

        if prompt is None:
            prompt = ""

        pos_prompt, neg_prompt = normalize_prompt(prompt)  # TODO?
        NEGPROMPT_BOOSTS: set = {" mature", ",mature", " old", ",old", "adult", "elderly", "middle aged"}

        is_neg_prompt_suspect = False

        if neg_prompt and any(boost in neg_prompt for boost in NEGPROMPT_BOOSTS):
            is_neg_prompt_suspect = True

        # nsfw_model, model_tags = get_model_details(model_info)

        # Get the similarity results for the predicates
        predicate_similarity_result = self.get_predicate_similarities(
            image=image_tensor if image_tensor is not None else image,
        )

        # Determine if the image is anime or cartoon
        is_anime = False
        found_anime_predicates: dict[str, float] = {}
        for anime_predicate in self.anime_cartoon_predicates:
            if predicate_similarity_result[anime_predicate] > self.anime_cartoon_predicates[anime_predicate]:
                is_anime = True
                found_anime_predicates[anime_predicate] = round(predicate_similarity_result[anime_predicate], 4)

        # The anime tag is a little too broad sometimes, so we want to check
        # if any of the photo tags are above a certain threshold
        # If they are, we will treat the image as not anime
        if is_anime and all(
            anime_predicate_value < 0.208 for anime_predicate_value in found_anime_predicates.values()
        ):
            for result_name, result_value in predicate_similarity_result.items():
                if "photo" in result_name and result_value > 0.24:
                    is_anime = False
                    break

        is_nsfw = False
        nsfw_concepts_identified = {}

        nsfw_similarity_results = {}
        anime_concepts_identified = {}
        # If the image is anime, we can use deepdanbooru to get a more accurate result
        if self.deep_danbooru_model is not None and is_anime:
            deep_danbooru_result = self.check_for_nsfw_anime_only(
                image=image,
                prompt=prompt,
                model_info=model_info,
            )
            if deep_danbooru_result is None:
                logger.error("Could not get anime result")
                return None
            else:
                is_nsfw = is_nsfw or deep_danbooru_result.is_anime_nsfw
                if "realistic" in deep_danbooru_result.tag_results:
                    is_anime = False
                nsfw_concepts_identified.update(deep_danbooru_result.nsfw_anime_scores.to_dict())
                anime_concepts_identified.update(deep_danbooru_result.tag_results)

        # Predicates typically include human figure detection, but may include other terms, such as those
        # which describe a particular body part.
        matched_predicates: dict[str, float] = {}
        contains_subject_requiring_check = False

        for predicate_term in self._all_predicates.keys():
            predicate_similarity = predicate_similarity_result[predicate_term]

            if predicate_similarity > self._all_predicates[predicate_term]:
                matched_predicates[predicate_term] = round(predicate_similarity, 4)
                contains_subject_requiring_check = True
            elif (
                predicate_term in self._combined_underage_danger_concepts
                and predicate_similarity > self._combined_underage_danger_concepts[predicate_term]
            ):
                contains_subject_requiring_check = True

        # Get the similarity results for the nsfw concepts
        nsfw_similarity_results.update(self.get_nsfw_concept_similarities(image=image))

        # If the image wasn't already determined to be a NSFW anime/cartoon image, and it contains a predicate match,
        # we want to check if it is NSFW
        if not is_nsfw and contains_subject_requiring_check:
            for concept in self.underage_nsfw_predicates:
                if predicate_similarity_result[concept] > self._nsfw_combined_predicates[concept]:
                    is_nsfw = True
                    nsfw_concepts_identified[concept] = round(predicate_similarity_result[concept], 4)
                    continue

            for concept in nsfw_similarity_results:
                nsfw_threshold = self._all_nsfw_concepts[concept]

                # # If the image appears to be anime, use the anime nsfw thresholds to catch anything
                # # deepdanbooru didn't.
                if is_anime and concept in self.nsfw_post_anime_concepts:
                    nsfw_threshold = self.nsfw_post_anime_concepts[concept]

                # # We only check for the similarity of certain tags if the image is anime/cartoon
                # # This is because many of the other concepts included are not accurate for anime
                if is_anime and concept not in self.nsfw_post_anime_concepts:
                    continue

                # if nsfw_model:
                # nsfw_threshold -= 0.01
                # pass

                if nsfw_similarity_results[concept] > nsfw_threshold:
                    is_nsfw = True
                    nsfw_concepts_identified[concept] = round(nsfw_similarity_results[concept], 4)

        is_csam = False
        is_underage_subject = False

        found_csam_predicates: dict[str, float] = {}
        # if is_nsfw:
        if True:
            for predicate, predicate_value in self._combined_underage_danger_concepts.items():
                if predicate_similarity_result[predicate] > predicate_value:
                    found_csam_predicates[predicate] = predicate_similarity_result[predicate]

            if len(found_csam_predicates) >= 1:
                is_underage_subject = True

            if not is_underage_subject:
                if is_anime:
                    second_pass_multiplier = 0.05

                    if is_neg_prompt_suspect:
                        second_pass_multiplier -= 0.005

                    teen_adjustment = 0.0

                    # if predicate_similarity_result["shota"] > self._all_predicates["shota"]:
                    # second_pass_multiplier = 0.0

                    if predicate_similarity_result["teen"] > (self._all_predicates["teen"] + teen_adjustment):
                        found_csam_predicates["teen"] = predicate_similarity_result["teen"]

                    if predicate_similarity_result["teens"] > (self._all_predicates["teens"] + teen_adjustment):
                        found_csam_predicates["teens"] = predicate_similarity_result["teens"]

                    for predicate, predicate_value in self.underage_predicates.items():
                        if predicate == "lolicon" or predicate == "shota" or "photo" in predicate:
                            continue

                        adjusted_value = self.underage_predicates[predicate] + (
                            self.underage_danger_modifiers[predicate] * second_pass_multiplier
                        )

                        if predicate_similarity_result[predicate] > adjusted_value:
                            found_csam_predicates[predicate] = predicate_similarity_result[predicate]
                elif not is_anime:
                    second_pass_multiplier = 0.4

                    if is_neg_prompt_suspect:
                        second_pass_multiplier *= 0.5

                    for predicate, predicate_value in self._combined_underage_predicates.items():
                        adjusted_value = predicate_value + (
                            (
                                (
                                    self._combined_underage_danger_concepts[predicate]
                                    - self._combined_underage_predicates[predicate]
                                )
                                * second_pass_multiplier
                            )
                        )

                        if predicate_similarity_result[predicate] > adjusted_value:
                            found_csam_predicates[predicate] = predicate_similarity_result[predicate]

                PAIRS = [
                    ("tween", "tweens"),
                    ("tweens", "tween"),
                    ("teen", "teens"),
                    ("teens", "teen"),
                    ("an infant", "infants"),
                    ("infants", "an infant"),
                    ("toddler", "toddlers"),
                    ("toddlers", "toddler"),
                    ("child", "children"),
                    ("children", "child"),
                ]

                for concept1, concept2 in PAIRS:
                    if concept1 in found_csam_predicates and concept2 in found_csam_predicates:
                        if found_csam_predicates[concept1] > found_csam_predicates[concept2]:
                            del found_csam_predicates[concept2]
                        else:
                            del found_csam_predicates[concept1]

                if is_anime and len(found_csam_predicates) >= 2:
                    is_underage_subject = True
                elif len(found_csam_predicates) >= 2:
                    is_underage_subject = True

        if is_underage_subject and is_nsfw:
            is_csam = True

        return NSFWResult(
            all_predicate_scores=predicate_similarity_result,
            matched_predicates=matched_predicates,
            nsfw_similarity_results=nsfw_similarity_results,
            anime_concepts_identified=anime_concepts_identified,
            is_nsfw=is_nsfw,
            nsfw_concepts_identified=nsfw_concepts_identified,
            is_anime=is_anime,
            is_underage_subject=is_underage_subject,
            is_csam=is_csam,
            csam_predicates_matched=found_csam_predicates,
            is_neg_prompt_suspect=is_neg_prompt_suspect,
        )

    def get_closest_resize_target(self, image: PIL.Image.Image) -> tuple[int, int]:
        """Gets the closest resize target to the image's size.

        Args:
            image (PIL.Image.Image): The image to get the closest resize target for.

        Returns:
            tuple[int, int]: The closest resize target.
        """

        # Return the best aspect ratio match which is closest to the number of pixels in the image

        image_width, image_height = image.size
        image_num_pixels = image_width * image_height

        closest_target: tuple[int, int] = (0, 0)
        closest_target_num_pixels = 0

        for target in self.resize_targets:
            target_width, target_height = target
            target_num_pixels = target_width * target_height

            if closest_target_num_pixels == 0:
                closest_target = target
                closest_target_num_pixels = target_num_pixels

            if abs(target_num_pixels - image_num_pixels) < abs(closest_target_num_pixels - image_num_pixels):
                closest_target = target
                closest_target_num_pixels = target_num_pixels

        return closest_target

    def check_for_nsfw_anime_only(
        self,
        image: PIL.Image.Image,
        prompt: str | None = "",
        model_info: dict | None = None,
    ) -> NSFWAnimeResult | None:
        """Checks if an image is potentially NSFW, but only works reliable with cartoon/anime images.

        Args:
            deep_danbooru_model (DeepDanbooruModel): The deepdanbooru model to use for the check.
            image (PIL.Image.Image): The image to check.
            prompt (str): The prompt used to create the image.
            model_info (dict): The entry from the model reference for the model used to create the image.
        """
        if self.deep_danbooru_model is None:
            return None

        # Resize to one of the target sizes that the width and height are closest to
        closest_target: tuple[int, int] = self.get_closest_resize_target(image=image)

        if closest_target == (0, 0):
            logger.error("Could not find a closest target")
            closest_target = (768, 768)

        image = image.convert("RGB")
        image = image.resize(closest_target, PIL.Image.LANCZOS)

        a = np.expand_dims(np.array(image, dtype=np.float32), 0) / 255

        tensor_to_evaluate = torch.from_numpy(a)

        evaluation_result_numpy = self.deep_danbooru_model.evaluate_tensor(tensor_to_evaluate)

        tag_results = {}

        safe_score = 0.0
        questionable_score = 0.0
        explicit_score = 0.0

        nsfw_tags: list[str] = []

        for i, p in enumerate(evaluation_result_numpy):
            if len(self.deep_danbooru_model.tags) <= i:
                break

            tag_name = self.deep_danbooru_model.tags[i]

            if tag_name == NSFWAnimeScores.safe_score_key:
                safe_score = round(float(p), 4)
            elif tag_name == NSFWAnimeScores.questionable_score_key:
                questionable_score = round(float(p), 4)
            elif tag_name == NSFWAnimeScores.explicit_score_key:
                explicit_score = round(float(p), 4)
            elif tag_name in self.nsfw_deepdanbooru_concepts and p > self.nsfw_deepdanbooru_concepts[tag_name]:
                nsfw_tags.append(tag_name)
            if p > 0.5:
                tag_results[tag_name] = round(float(p), 4)

        # if model_info is not None:
        # is_stable_diffusion_model_nsfw, stable_diffusion_model_tags = get_model_details(model_info)

        # if is_stable_diffusion_model_nsfw:
        #     safe_score -= 0.1
        #     questionable_score += 0.1
        #     explicit_score += 0.1

        return NSFWAnimeResult(
            tag_results=tag_results,
            nsfw_anime_scores=NSFWAnimeScores(
                safe_score=safe_score,
                questionable_score=questionable_score,
                explicit_score=explicit_score,
                nsfw_tags=nsfw_tags,
            ),
        )


class NSFWFolderChecker(NSFWChecker):
    nsfw_pilot_folders: list[Path] | None
    """Folders with images that are good test cases for known NSFW."""
    nsfw_folders: list[Path] | None
    """Folders with images that are known to be NSFW."""
    lewd_pilot_folders: list[Path] | None
    """Folders with images that are weaker test cases for NSFW, but would be ideal if they were flagged as such."""
    lewd_folders: list[Path] | None
    """Folders with images that are weaker test cases for NSFW."""
    inappropriate_pilot_folders: list[Path] | None
    """Folders with images that are weak test cases for NSFW, and may be hard to automatically categorize."""
    inappropriate_folders: list[Path] | None
    """Folders with images that are weak test cases for NSFW, and may be hard to automatically categorize."""
    sfw_pilot_folders: list[Path] | None
    """Folders with images that are good test cases for known SFW, and should mostly *not* be categorized NSFW."""
    sfw_folders: list[Path] | None
    """Folders with images that are known to be SFW."""
    neutral_pilot_folders: list[Path] | None
    """Folders with images that contain a mix of human and non-human subjects, and in situations that generally
    wouldn't fit into the NSFW or SFW categories."""
    neutral_folders: list[Path] | None
    """Folders with images that contain a mix of human and non-human subjects, and in situations that generally
    wouldn't fit into the NSFW or SFW categories."""

    other_folders: list[tuple[Path, NSFWFolderResultTypeExpected]] | None
    """Any other folders that should be checked, and the expected result type."""

    def __init__(
        self,
        interrogator: Interrogator,
        deep_danbooru_model: DeepDanbooruModel | None,
        *,
        nsfw_pilot_folders: list[Path] | None = None,
        nsfw_folders: list[Path] | None = None,
        lewd_pilot_folders: list[Path] | None = None,
        lewd_folders: list[Path] | None = None,
        inappropriate_pilot_folders: list[Path] | None = None,
        inappropriate_folders: list[Path] | None = None,
        sfw_pilot_folders: list[Path] | None = None,
        sfw_folders: list[Path] | None = None,
        neutral_pilot_folders: list[Path] | None = None,
        neutral_folders: list[Path] | None = None,
        other_folders: list[tuple[Path, NSFWFolderResultTypeExpected]] | None = None,
    ) -> None:
        """Instantiate an instance of the NSFWFolderChecker class.

        If you intend to change the predicates or concepts, you should call `refresh_predicates` and `refresh_concepts`
        as appropriate after changing them.

        Args:
            interrogator (Interrogator): The interrogator to use for the check.
            deep_danbooru_model (DeepDanbooruModel): The deepdanbooru model to use for the check.
            pilot_folders (list[Path], optional): Folders with images that are known to be NSFW.
            nsfw_folders (list[Path], optional): Folders with images that are known to be NSFW.
            lewd_folders (list[Path], optional): Folders with images that are known to be NSFW.
            inappropriate_folders (list[Path], optional): Folders with images that are known to be NSFW.
            sfw_folders (list[Path], optional): Folders with images that are known to be SFW.
            neutral_folders (list[Path], optional): Folders with images that are known to be neutral.
        """
        super().__init__(
            interrogator=interrogator,
            deep_danbooru_model=deep_danbooru_model,
        )

        self.nsfw_pilot_folders = nsfw_pilot_folders
        self.nsfw_folders = nsfw_folders
        self.lewd_pilot_folders = lewd_pilot_folders
        self.lewd_folders = lewd_folders
        self.inappropriate_pilot_folders = inappropriate_pilot_folders
        self.inappropriate_folders = inappropriate_folders
        self.sfw_pilot_folders = sfw_pilot_folders
        self.sfw_folders = sfw_folders
        self.neutral_pilot_folders = neutral_pilot_folders
        self.neutral_folders = neutral_folders
        self.other_folders = other_folders

        all_folders = self.get_all_folders_to_check() + self.get_all_pilot_folders()

        for folder_path, nsfw_type_expected in all_folders:
            if not folder_path.exists():
                logger.error(f"Folder path does not exist: {folder_path}")
                raise ValueError(f"Folder path does not exist: {folder_path}")
            if not folder_path.is_dir():
                logger.error(f"Folder path is not a directory: {folder_path}")
                raise ValueError(f"Folder path is not a directory: {folder_path}")

    def get_all_folders_to_check(self) -> list[tuple[Path, NSFWFolderResultTypeExpected]]:
        all_folders_to_check: list[tuple[Path, NSFWFolderResultTypeExpected]] = []

        all_folders_to_check += self.get_all_pilot_folders()

        if self.nsfw_folders is not None:
            all_folders_to_check += [(folder, NSFWFolderResultTypeExpected.nsfw) for folder in self.nsfw_folders]
        if self.lewd_folders is not None:
            all_folders_to_check += [(folder, NSFWFolderResultTypeExpected.nsfw) for folder in self.lewd_folders]
        if self.inappropriate_folders is not None:
            all_folders_to_check += [
                (folder, NSFWFolderResultTypeExpected.nsfw) for folder in self.inappropriate_folders
            ]
        if self.sfw_folders is not None:
            all_folders_to_check += [(folder, NSFWFolderResultTypeExpected.sfw) for folder in self.sfw_folders]
        if self.neutral_folders is not None:
            all_folders_to_check += [(folder, NSFWFolderResultTypeExpected.sfw) for folder in self.neutral_folders]

        if self.other_folders is not None:
            all_folders_to_check += self.other_folders

        return all_folders_to_check

    def get_all_pilot_folders(self) -> list[tuple[Path, NSFWFolderResultTypeExpected]]:
        pilot_folders: list[tuple[Path, NSFWFolderResultTypeExpected]] = []
        if self.nsfw_pilot_folders is not None:
            pilot_folders += [(folder, NSFWFolderResultTypeExpected.nsfw) for folder in self.nsfw_pilot_folders]
        if self.lewd_pilot_folders is not None:
            pilot_folders += [(folder, NSFWFolderResultTypeExpected.mostly_nsfw) for folder in self.lewd_pilot_folders]
        if self.inappropriate_pilot_folders is not None:
            pilot_folders += [
                (folder, NSFWFolderResultTypeExpected.mostly_nsfw) for folder in self.inappropriate_pilot_folders
            ]
        if self.sfw_pilot_folders is not None:
            pilot_folders += [(folder, NSFWFolderResultTypeExpected.sfw) for folder in self.sfw_pilot_folders]
        if self.neutral_pilot_folders is not None:
            pilot_folders += [(folder, NSFWFolderResultTypeExpected.sfw) for folder in self.neutral_pilot_folders]

        return pilot_folders

    def check_folder(
        self,
        folder_path: Path,
        nsfw_type_expected: NSFWFolderResultTypeExpected,
        *,
        sample_size: int | None = None,
        redirect_console_output: bool = True,
    ):
        image_file_paths = get_image_file_paths(folder_path)

        if sample_size is not None and sample_size < len(image_file_paths):
            import random

            random.seed(0)
            image_file_paths = random.sample(image_file_paths, sample_size)

        folder_result = NSFWFolderResults(folder_path=folder_path, nsfw_type_excepted=nsfw_type_expected)

        progress_bar = tqdm(total=len(image_file_paths))

        def preload_images(paths: list[Path]) -> dict[Path, tuple[PIL.Image.Image, torch.Tensor]]:
            images: dict[Path, tuple[PIL.Image.Image, torch.Tensor]] = {}
            pil_images = []
            valid_paths = []
            for path in paths:
                try:
                    image = PIL.Image.open(path)
                    pil_images.append(image)
                    valid_paths.append(path)
                except Exception as e:
                    logger.trace(f"Error loading image ({type(e)}): {path}")
            # Batch feature extraction if any images loaded
            if pil_images:
                try:
                    # image_to_features supports batch input
                    image_tensors = self.interrogator.image_to_features(pil_images)
                    # image_tensors is a tensor of shape (batch_size, feature_dim)
                    for path, image, tensor in zip(valid_paths, pil_images, image_tensors, strict=True):
                        images[path] = (image, tensor)
                except Exception as e:
                    logger.trace(f"Error in batch feature extraction: {e}")
                    # fallback to per-image extraction if batch fails
                    for path, image in zip(valid_paths, pil_images):
                        try:
                            tensor = self.interrogator.image_to_features(image)
                            images[path] = (image, tensor)
                        except Exception as e2:
                            logger.trace(f"Error extracting features for image ({type(e2)}): {path}")
            return images

        with redirect_stdout(None) if redirect_console_output else nullcontext():
            with redirect_stderr(None) if redirect_console_output else nullcontext():
                batch_size = 500
                for i in range(0, len(image_file_paths), batch_size):
                    batch_paths = image_file_paths[i : i + batch_size]
                    preloaded_images = preload_images(batch_paths)

                    for image_file_path in batch_paths:
                        if not image_file_path.exists():
                            logger.error(f"Image file path does not exist: {image_file_path}")
                            continue

                        try:
                            progress_bar.set_description(str(folder_result))
                            image, image_tensor = preloaded_images.get(image_file_path, (None, None))
                            if image is None:
                                continue
                            result = self.check_for_nsfw(
                                image,
                                image_tensor=image_tensor,
                            )

                            folder_result.add_result(
                                path=image_file_path,
                                result=result,
                            )

                        except Exception as e:
                            logger.trace(f"Error checking image ({type(e)}): {image_file_path}")
                            logger.trace(e)
                        finally:
                            progress_bar.update(1)

        progress_bar.close()

        return folder_result

    def check_pilot_folders(self) -> list[NSFWFolderResults]:
        all_pilot_folder_results = []
        pilot_folders = self.get_all_pilot_folders()

        for folder_path, nsfw_type_expected in pilot_folders:
            folder_result = self.check_folder(
                folder_path=folder_path,
                nsfw_type_expected=nsfw_type_expected,
            )

            all_pilot_folder_results.append(folder_result)

        return all_pilot_folder_results

    def check_all_folders(self) -> list[NSFWFolderResults]:
        all_folder_results: list[NSFWFolderResults] = []

        all_folders_to_check: list[tuple[Path, NSFWFolderResultTypeExpected]]
        all_folders_to_check = self.get_all_folders_to_check()

        for folder_path, nsfw_type_expected in all_folders_to_check:
            folder_result = self.check_folder(
                folder_path=folder_path,
                nsfw_type_expected=nsfw_type_expected,
            )

            all_folder_results.append(folder_result)

        return all_folder_results
