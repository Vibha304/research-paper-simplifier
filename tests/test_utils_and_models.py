from __future__ import annotations

import pytest

from backend.models import Assessment, PaperAnalysis
from services.scoring_service import ScoringService, find_mentions, merge_detected_resources
from utils.exceptions import LLMResponseFormatError
from utils.json_utils import extract_json
from utils.progress import ProgressTracker
from utils.text import dedupe_preserve_order, slugify, to_latin1, truncate


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        'Sure! ```json\n{"a": 1}\n``` hope it helps',
        'prefix {"a": 1,} suffix',
        'text {"a": 1, "s": "brace } inside"} more',
    ],
)
def test_extract_json_recovers_objects(raw):
    assert extract_json(raw)["a"] == 1


@pytest.mark.parametrize("raw", ["", "no json here", "[1, 2, 3]"])
def test_extract_json_raises(raw):
    with pytest.raises(LLMResponseFormatError):
        extract_json(raw)


def test_text_helpers():
    assert slugify("Deep Learning: A Survey!") == "deep-learning-a-survey"
    assert truncate("one two three four", 12) == "one two..."
    assert dedupe_preserve_order(["PyTorch", "pytorch.", " NumPy "]) == ["PyTorch", "NumPy"]
    assert to_latin1("café – “quoted” ∑") == 'café - "quoted" ?'


def test_progress_tracker_maps_stages_and_is_monotonic():
    seen: list[float] = []
    tracker = ProgressTracker(lambda f, _: seen.append(round(f, 3)))
    stage = tracker.stage(0.2, 0.4)
    stage(0.5, "half")
    stage(0.1, "backwards")  # must not move the bar backwards
    assert seen == [0.3, 0.3]


def test_assessment_coercion_and_bounds():
    a = Assessment.model_validate({"difficulty_level": "EXPERT", "difficulty_score": 42, "estimated_hours": "-5"})
    assert a.difficulty_level == "Advanced"
    assert a.difficulty_score == 4  # 42 on a 0-100 scale -> 4/10
    assert a.estimated_hours == 1


def test_find_mentions_word_boundaries():
    text = "We used PyTorch and the CIFAR-10 dataset; ResNet-50 baseline. Results in R."
    assert "PyTorch" in find_mentions(text, ("PyTorch", "TensorFlow"))
    assert find_mentions("CIFAR-100 only", ("CIFAR-10",)) == []


def test_merge_detected_resources_adds_missing_and_drops_placeholders():
    analysis = PaperAnalysis(datasets=["Not explicitly mentioned in the paper."], tools_frameworks=["PyTorch 2.0"])
    merged = merge_detected_resources(analysis, "Experiments on MNIST with PyTorch and NumPy.")
    assert merged.datasets == ["MNIST (detected in text)"]
    assert merged.tools_frameworks == ["PyTorch 2.0", "NumPy (detected in text)"]


def test_scoring_blends_llm_and_heuristics():
    analysis = PaperAnalysis(
        implementation_steps=["a", "b", "c"],
        assessment=Assessment(difficulty_score=9, resume_impact_score=8, estimated_hours=100),
    )
    math_heavy = "Theorem 1. Lemma 2. proof ∑ ∫ α β (1)\n" * 50 + "transformer deep learning " * 10
    result = ScoringService().finalize(math_heavy, 30, analysis)
    assert result.difficulty_level == "Advanced"
    assert 1 <= result.resume_impact_score <= 10
    assert "weeks" in result.estimated_time_label
    easy = ScoringService().finalize("simple words " * 500, 4, PaperAnalysis(assessment=Assessment(difficulty_score=2)))
    assert easy.difficulty_level == "Beginner"
