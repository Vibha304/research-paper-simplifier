"""Explainable scoring: difficulty, resume impact and implementation time.

Final scores blend the LLM's judgement (60 %) with deterministic text
heuristics (40 %). Heuristics keep scores stable and grounded even when a small
model is overly optimistic, and every factor is reported back to the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from backend.models import Assessment, PaperAnalysis
from utils.text import dedupe_preserve_order

LLM_WEIGHT = 0.6

_MATH_SYMBOLS = re.compile(r"[∑∏∫∂∇αβγδεζηθλμνξπρστφχψωΣΩΘΛΦΨ≤≥≈≠∈∉⊂⊆∀∃→↦⊗⊕√∞]")
_EQUATION_NUMBER = re.compile(r"\(\d{1,3}\)\s*$", re.MULTILINE)

ADVANCED_TERMS = (
    "theorem", "lemma", "proof", "corollary", "convex", "stochastic", "variational", "bayesian",
    "markov", "eigen", "lagrangian", "gradient", "manifold", "transformer", "attention mechanism",
    "reinforcement learning", "generative adversarial", "diffusion model", "graph neural",
    "differential equation", "optimization problem", "np-hard", "kernel", "probabilistic",
    "cryptograph", "homomorphic", "quantum", "fourier", "monte carlo", "backpropagation",
)
TRENDING_TERMS = (
    "large language model", "llm", "transformer", "generative", "diffusion", "retrieval-augmented",
    "rag", "federated learning", "edge computing", "iot", "internet of things", "blockchain",
    "cybersecurity", "intrusion detection", "computer vision", "object detection", "autonomous",
    "reinforcement learning", "graph neural", "explainable", "deep learning", "machine learning",
    "5g", "6g", "smart grid", "healthcare", "medical imaging", "nlp", "natural language",
    "recommendation", "cloud", "robotics", "drone", "uav", "quantum",
)

KNOWN_DATASETS = (
    "ImageNet", "CIFAR-10", "CIFAR-100", "MNIST", "Fashion-MNIST", "COCO", "MS COCO", "Pascal VOC",
    "Cityscapes", "KITTI", "SQuAD", "GLUE", "SuperGLUE", "IMDB", "WikiText", "Penn Treebank",
    "LibriSpeech", "Common Voice", "UCI", "Kaggle", "NSL-KDD", "KDD Cup 99", "CICIDS2017", "UNSW-NB15",
    "MIMIC-III", "ChestX-ray14", "CheXpert", "BraTS", "ISIC", "MovieLens", "Amazon Reviews", "Yelp",
    "OpenStreetMap", "nuScenes", "Waymo Open", "Human3.6M", "UCF101", "Kinetics", "AudioSet", "LFW",
    "CelebA", "Market-1501", "SVHN", "Omniglot", "Cora", "Citeseer", "PubMed", "OGB", "MMLU", "GSM8K",
    "HumanEval", "SST-2", "CoNLL-2003", "WMT", "Flickr30k", "Visual Genome", "VQA", "ShapeNet", "ModelNet40",
)
KNOWN_TOOLS = (
    "PyTorch", "TensorFlow", "Keras", "JAX", "scikit-learn", "Hugging Face", "Transformers", "OpenCV",
    "NumPy", "Pandas", "SciPy", "Matplotlib", "MATLAB", "Simulink", "R", "Julia", "CUDA", "cuDNN",
    "Python", "C++", "Java", "Docker", "Kubernetes", "Apache Spark", "Hadoop", "Kafka", "ROS", "Gazebo",
    "NS-2", "NS-3", "OMNeT++", "Mininet", "Cooja", "Contiki", "Arduino", "Raspberry Pi", "FPGA", "Verilog",
    "VHDL", "LabVIEW", "ANSYS", "COMSOL", "Ethereum", "Hyperledger", "Solidity", "LangChain", "Ollama",
    "XGBoost", "LightGBM", "YOLO", "ResNet", "BERT", "GPT", "LLaMA", "Stable Diffusion", "NetworkX",
    "PyTorch Geometric", "DGL", "Weka", "Flask", "FastAPI", "React", "Android", "MongoDB", "PostgreSQL",
    "MySQL", "AWS", "Google Colab", "Jupyter", "NVIDIA",
)
_SHORT_AMBIGUOUS = {"R", "GPT", "AWS"}  # need exact-case, word-boundary match


def find_mentions(text: str, vocabulary: tuple[str, ...]) -> list[str]:
    """Return vocabulary items that literally appear in ``text`` (word-boundary match)."""
    found: list[str] = []
    for term in vocabulary:
        flags = 0 if term in _SHORT_AMBIGUOUS else re.IGNORECASE
        if re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", text, flags):
            found.append(term)
    return found


@dataclass(frozen=True)
class HeuristicFactors:
    math_density: float      # 0..1
    advanced_terms: float    # 0..1
    length: float            # 0..1
    trending: float          # 0..1
    practicality: float      # 0..1

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 2) for k, v in self.__dict__.items()}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class ScoringService:
    def compute_factors(self, text: str, page_count: int, analysis: PaperAnalysis) -> HeuristicFactors:
        words = max(1, len(text.split()))
        lowered = text.lower()
        math_hits = len(_MATH_SYMBOLS.findall(text)) + 3 * len(_EQUATION_NUMBER.findall(text))
        advanced_hits = sum(lowered.count(term) for term in ADVANCED_TERMS)
        trending_terms = sum(1 for term in TRENDING_TERMS if re.search(rf"\b{re.escape(term)}\b", lowered))
        has_resources = bool(analysis.datasets) + bool(analysis.tools_frameworks)
        return HeuristicFactors(
            math_density=_clamp(math_hits / words * 1000 / 25),
            advanced_terms=_clamp(advanced_hits / words * 1000 / 6),
            length=_clamp(page_count / 30),
            trending=_clamp(trending_terms / 5),
            practicality=_clamp(0.3 + 0.35 * has_resources),
        )

    def finalize(
        self, text: str, page_count: int, analysis: PaperAnalysis, *, llm_weight: float = LLM_WEIGHT
    ) -> Assessment:
        """Blend the LLM's assessment with heuristics and fill derived fields.

        ``llm_weight=1.0`` keeps the engine's own scores (used by the built-in engine,
        whose assessment is already heuristic-based).
        """
        factors = self.compute_factors(text, page_count, analysis)
        llm = analysis.assessment

        heuristic_difficulty = 1 + 9 * (
            0.4 * factors.math_density + 0.35 * factors.advanced_terms + 0.25 * factors.length
        )
        difficulty = round(llm_weight * llm.difficulty_score + (1 - llm_weight) * heuristic_difficulty)
        difficulty = int(_clamp(difficulty, 1, 10))
        level = "Beginner" if difficulty <= 3 else "Intermediate" if difficulty <= 6 else "Advanced"

        # Resume impact rewards trending topics and moderately hard, practical projects.
        challenge = 1 - abs(difficulty - 7) / 7
        heuristic_resume = 1 + 9 * (0.5 * factors.trending + 0.25 * challenge + 0.25 * factors.practicality)
        resume = int(_clamp(round(llm_weight * llm.resume_impact_score + (1 - llm_weight) * heuristic_resume), 1, 10))

        hours = self._estimate_hours(level, analysis, llm.estimated_hours)

        rationale_suffix = (
            f" (Heuristics: math density {factors.math_density:.0%}, advanced terminology "
            f"{factors.advanced_terms:.0%}, length {page_count} pages.)"
        )
        return llm.model_copy(
            update={
                "difficulty_level": level,
                "difficulty_score": difficulty,
                "difficulty_rationale": (llm.difficulty_rationale or f"Rated {level}.") + rationale_suffix,
                "resume_impact_score": resume,
                "resume_impact_rationale": llm.resume_impact_rationale
                or f"Topic relevance score {factors.trending:.0%} based on current industry trends.",
                "estimated_hours": hours,
                "estimated_time_label": self.time_label(hours),
                "skills_gained": dedupe_preserve_order(llm.skills_gained)[:8],
            }
        )

    @staticmethod
    def _estimate_hours(level: str, analysis: PaperAnalysis, llm_hours: int) -> int:
        base = {"Beginner": 25, "Intermediate": 50, "Advanced": 90}[level]
        heuristic = base + 3 * len(analysis.implementation_steps) + 4 * len(analysis.build_plan.modules)
        llm_hours = int(_clamp(llm_hours, 8, 400))
        return int(round((llm_hours + heuristic) / 2))

    @staticmethod
    def time_label(hours: int, hours_per_week: int = 10) -> str:
        weeks = max(1, round(hours / hours_per_week))
        upper = weeks + max(1, weeks // 4)
        return f"~{hours} hours ({weeks}-{upper} weeks at {hours_per_week} h/week)"


def merge_detected_resources(analysis: PaperAnalysis, text: str) -> PaperAnalysis:
    """Cross-check LLM-listed resources with exact mentions found in the paper text."""
    detected_datasets = find_mentions(text, KNOWN_DATASETS)
    detected_tools = find_mentions(text, KNOWN_TOOLS)

    def merge(llm_items: list[str], detected: list[str]) -> list[str]:
        meaningful = [i for i in llm_items if "not explicitly mentioned" not in i.lower()]
        already = " ".join(meaningful).lower()
        extra = [d for d in detected if d.lower() not in already]
        return dedupe_preserve_order(meaningful + [f"{d} (detected in text)" for d in extra])

    return analysis.model_copy(
        update={
            "datasets": merge(analysis.datasets, detected_datasets),
            "tools_frameworks": merge(analysis.tools_frameworks, detected_tools),
        }
    )
