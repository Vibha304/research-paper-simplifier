"""Shared fixtures: isolated settings, a deterministic fake LLM and sample PDFs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import pymupdf as fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import Settings, load_settings  # noqa: E402

CANNED: dict[str, dict] = {
    "simple_summary": {
        "simple_summary": "This paper teaches a computer to spot cats in photos, like a friend who learns faces.",
        "problem_statement": "Existing image classifiers are slow on phones.",
        "key_contributions": ["A lightweight CNN", "A new training trick", "Benchmarks on CIFAR-10"],
        "keywords": ["deep learning", "computer vision", "CNN"],
    },
    "methodology": {
        "methodology": "The images pass through convolution layers (filters that detect edges) ...",
        "implementation_steps": "1. Download CIFAR-10\n2. Build the CNN in PyTorch\n3. Train for 50 epochs",
    },
    "datasets": {"datasets": [{"name": "CIFAR-10", "description": "60k small images"}], "tools_frameworks": ["PyTorch"]},
    "limitations": {"limitations": ["Only tested on small images"], "future_work": ["Try ImageNet"]},
    "project_idea": {
        "project_idea": {"title": "PocketVision", "description": "Mobile classifier", "features": ["Camera input"],
                         "tech_stack": ["PyTorch", "Streamlit"], "deliverables": ["Demo app"]},
        "build_plan": {"project_name": "pocket_vision", "objective": "Classify images on-device",
                       "architecture": "Data loader -> CNN -> evaluator", "modules": ["Data Loader: load CIFAR",
                       "model: the CNN", "class: invalid keyword name"], "setup_steps": ["Create venv"],
                       "evaluation_metrics": ["Accuracy"], "stretch_goals": ["Quantization"]},
    },
    "roadmap": {
        "roadmap": [{"phase": "Setup", "duration": "Week 1", "tasks": ["Install PyTorch"]}],
        "difficulty_level": "intermediate", "difficulty_score": "6/10", "difficulty_rationale": "Needs CNN basics.",
        "resume_impact_score": 80, "resume_impact_rationale": "Computer vision is in demand.",
        "estimated_hours": "60 hours", "skills_gained": ["PyTorch", "CNNs"],
    },
    "viva_questions": {"viva_questions": [{"q": "What is a convolution?", "a": "A sliding filter."}]},
    "interview_questions": {"interview_questions": [{"question": "How to deploy?", "answer": "Export to ONNX."}]},
}


class FakeLLM:
    """Deterministic stand-in for Ollama. Embeddings are hashed bag-of-words vectors."""

    model = "fake-llm"
    embedding_model = "fake-embed"

    def __init__(self, *, broken_tasks: Sequence[str] = ()) -> None:
        self.broken_tasks = set(broken_tasks)
        self.calls: list[str] = []

    def chat(self, messages, *, json_mode: bool = False, temperature: float | None = None) -> str:
        prompt = messages[-1]["content"]
        schema = prompt.rsplit("Return a JSON object with exactly these keys:", 1)[-1]
        first_key = re.search(r'"(\w+)"', schema).group(1)
        self.calls.append(first_key)
        if first_key in self.broken_tasks:
            return "Sorry, I cannot produce JSON."
        return "Here you go:\n```json\n" + json.dumps(CANNED[first_key]) + "\n```"

    def stream_chat(self, messages, *, temperature: float | None = None) -> Iterator[str]:
        yield from ["The paper ", "proposes a CNN ", "(p. 1)."]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * 64
            for word in re.findall(r"[a-z]+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    base = load_settings()
    return replace(
        base,
        data_dir=tmp_path,
        sqlite_path=tmp_path / "test.db",
        chroma_dir=tmp_path / "chroma",
        log_dir=tmp_path / "logs",
        max_pages=20,
        chunk_size=600,
        chunk_overlap=100,
    )


@pytest.fixture()
def fake_llm() -> FakeLLM:
    return FakeLLM()


PARAGRAPH = (
    "Convolutional neural networks have become the dominant approach for image recognition. "
    "In this work we study how lightweight models can be trained efficiently on CIFAR-10 using PyTorch. "
    "We evaluate accuracy, latency and memory use on commodity hardware and compare with baselines. "
)


def make_paper_pdf(pages: int = 3, two_column: bool = True) -> bytes:
    """Create a synthetic research paper with headings, columns, and running headers."""
    doc = fitz.open()
    sections = ["Abstract", "1 Introduction", "2 Proposed Method", "3 Experiments", "4 Conclusion", "References"]
    section_iter = iter(sections)
    for page_no in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 30), "IEEE TRANSACTIONS ON TEST, VOL. 1", fontsize=8)
        page.insert_text((290, 820), str(page_no + 1), fontsize=8)
        top = 60
        if page_no == 0:
            page.insert_textbox(fitz.Rect(50, 50, 545, 90), "Efficient Lightweight CNNs for Mobile Vision",
                                fontsize=18, align=1)
            top = 110
        columns = [fitz.Rect(50, top, 290, 800), fitz.Rect(305, top, 545, 800)] if two_column else [
            fitz.Rect(50, top, 545, 800)]
        for rect in columns:
            heading = next(section_iter, None)
            y = rect.y0
            if heading:
                page.insert_textbox(fitz.Rect(rect.x0, y, rect.x1, y + 20), heading, fontsize=11)
                y += 24
            body = "[1] A. Author, Some cited work, 2020." if heading == "References" else PARAGRAPH * 2
            page.insert_textbox(fitz.Rect(rect.x0, y, rect.x1, y + 300), body, fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture()
def paper_pdf() -> bytes:
    return make_paper_pdf()
