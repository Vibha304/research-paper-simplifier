"""One-click "Build this Paper as a Project": generates a ready-to-code starter kit (.zip)."""

from __future__ import annotations

import io
import keyword
import re
import zipfile
from dataclasses import dataclass

from backend.models import PaperAnalysis
from services.export_service import analysis_to_markdown
from utils.text import slugify

# Tech-stack keyword -> pip package.
_PIP_PACKAGES: dict[str, str] = {
    "pytorch geometric": "torch-geometric", "pytorch": "torch", "torch": "torch", "tensorflow": "tensorflow",
    "keras": "keras", "scikit-learn": "scikit-learn", "sklearn": "scikit-learn", "opencv": "opencv-python",
    "numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib", "seaborn": "seaborn", "scipy": "scipy",
    "streamlit": "streamlit", "flask": "flask", "fastapi": "fastapi", "gradio": "gradio",
    "hugging face": "transformers", "transformers": "transformers", "xgboost": "xgboost",
    "lightgbm": "lightgbm", "networkx": "networkx", "nltk": "nltk", "spacy": "spacy",
    "langchain": "langchain", "chromadb": "chromadb", "ollama": "ollama", "yolo": "ultralytics",
    "jupyter": "jupyter", "plotly": "plotly", "pillow": "pillow", "librosa": "librosa",
}
_BASE_PACKAGES = ("numpy", "pandas", "matplotlib", "pytest")


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    purpose: str


def parse_modules(modules: list[str]) -> list[ModuleSpec]:
    """'Data Loader: loads CSVs' -> ModuleSpec('data_loader', 'loads CSVs'), unique & valid identifiers."""
    specs: list[ModuleSpec] = []
    seen: set[str] = set()
    for raw in modules:
        head, _, tail = raw.partition(":")
        name = slugify(head, max_length=40, separator="_")
        name = re.sub(r"\.py$", "", name)
        if not name or name[0].isdigit() or keyword.iskeyword(name):
            name = f"module_{name}" if name else f"module_{len(specs) + 1}"
        if name in seen or name == "main":
            continue
        seen.add(name)
        specs.append(ModuleSpec(name=name, purpose=(tail or head).strip()))
    return specs or [
        ModuleSpec("data_loader", "Load and preprocess the dataset."),
        ModuleSpec("model", "Implement the core method from the paper."),
        ModuleSpec("train", "Training / execution loop."),
        ModuleSpec("evaluate", "Compute evaluation metrics and plots."),
    ]


def requirements_for(tech_stack: list[str]) -> list[str]:
    packages: list[str] = list(_BASE_PACKAGES)
    joined = " ".join(tech_stack).lower()
    for key, package in _PIP_PACKAGES.items():
        if key in joined and package not in packages:
            packages.append(package)
    return packages


class ProjectKitService:
    def build_zip(self, analysis: PaperAnalysis) -> tuple[str, bytes]:
        """Return ``(filename, zip_bytes)`` for a starter project based on the analysis."""
        plan = analysis.build_plan
        project = slugify(plan.project_name or analysis.project_idea.title or analysis.title, 40, "_")
        modules = parse_modules(plan.modules)
        files: dict[str, str] = {
            "README.md": self._readme(analysis, modules),
            "ROADMAP.md": self._roadmap(analysis),
            "PAPER_ANALYSIS.md": analysis_to_markdown(analysis),
            "VIVA_PREP.md": self._viva(analysis),
            "requirements.txt": "\n".join(requirements_for(analysis.project_idea.tech_stack)) + "\n",
            ".gitignore": "__pycache__/\n*.pyc\n.venv/\ndata/raw/\ndata/processed/\noutputs/\n.ipynb_checkpoints/\n",
            "data/README.md": self._data_readme(analysis),
            "src/__init__.py": "",
            "src/config.py": self._config(),
            "src/main.py": self._main(modules),
            "tests/test_smoke.py": self._test(modules),
            "notebooks/.gitkeep": "",
            "outputs/.gitkeep": "",
        }
        for spec in modules:
            files[f"src/{spec.name}.py"] = self._module(spec, analysis)

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path, content in files.items():
                archive.writestr(f"{project}/{path}", content)
        return f"{project}_starter_kit.zip", buffer.getvalue()

    # ------------------------------------------------------------------ files
    @staticmethod
    def _readme(a: PaperAnalysis, modules: list[ModuleSpec]) -> str:
        p, b = a.project_idea, a.build_plan
        bullets = lambda items: "\n".join(f"- {i}" for i in items) or "- TBD"  # noqa: E731
        numbered = lambda items: "\n".join(f"{n}. {i}" for n, i in enumerate(items, 1)) or "1. TBD"  # noqa: E731
        tree = "\n".join(f"│   ├── {m.name}.py  # {m.purpose[:60]}" for m in modules)
        return f"""# {p.title or b.project_name or 'Paper Implementation Project'}

> Based on the paper: **{a.title}**

{p.description}

## Objective
{b.objective or 'Re-implement and extend the core idea of the paper.'}

## Architecture
{b.architecture or 'See PAPER_ANALYSIS.md for the methodology.'}

## Features
{bullets(p.features)}

## Project Structure
```
├── data/            # datasets (see data/README.md)
├── notebooks/       # exploration notebooks
├── outputs/         # models, plots, logs
├── src/
│   ├── main.py      # entry point
│   ├── config.py    # hyper-parameters & paths
{tree}
└── tests/
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m src.main
```

{numbered(b.setup_steps)}

## Evaluation
{bullets(b.evaluation_metrics)}

## Stretch Goals
{bullets(b.stretch_goals)}

## Deliverables
{bullets(p.deliverables)}

**Difficulty:** {a.assessment.difficulty_level} ({a.assessment.difficulty_score}/10) · \
**Estimated time:** {a.assessment.estimated_time_label}
"""

    @staticmethod
    def _roadmap(a: PaperAnalysis) -> str:
        lines = ["# Implementation Roadmap", ""]
        for phase in a.roadmap or []:
            lines.append(f"## {phase.phase}" + (f" — {phase.duration}" if phase.duration else ""))
            lines.extend(f"- [ ] {task}" for task in phase.tasks)
            lines.append("")
        if len(lines) == 2:
            lines += [f"- [ ] {step}" for step in a.implementation_steps] or ["- [ ] Plan the project"]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _viva(a: PaperAnalysis) -> str:
        out = ["# Viva & Interview Preparation", "", "## Viva Questions", ""]
        for i, qa in enumerate(a.viva_questions, 1):
            out += [f"**Q{i}. {qa.question}**", "", qa.answer, ""]
        out += ["## Interview Questions", ""]
        for i, qa in enumerate(a.interview_questions, 1):
            out += [f"**Q{i}. {qa.question}**", "", qa.answer, ""]
        return "\n".join(out)

    @staticmethod
    def _data_readme(a: PaperAnalysis) -> str:
        datasets = "\n".join(f"- {d}" for d in a.datasets) or "- No dataset was named in the paper; choose a public one."
        return f"# Data\n\nDatasets referenced by the paper:\n\n{datasets}\n\nPlace raw files in `data/raw/` " \
               "and processed files in `data/processed/` (both git-ignored).\n"

    @staticmethod
    def _config() -> str:
        return '''"""Central configuration: paths and hyper-parameters."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 10
'''

    @staticmethod
    def _module(spec: ModuleSpec, a: PaperAnalysis) -> str:
        return f'''"""{spec.purpose}

Part of the implementation of: {a.title}
"""

from __future__ import annotations


def run() -> None:
    """Entry point for the `{spec.name}` stage.

    TODO: {spec.purpose}
    """
    raise NotImplementedError("Implement {spec.name}.run()")
'''

    @staticmethod
    def _main(modules: list[ModuleSpec]) -> str:
        imports = "\n".join(f"from src import {m.name}" for m in modules)
        stages = ",\n".join(f'    ("{m.name}", {m.name}.run)' for m in modules)
        return f'''"""Run every stage of the pipeline in order."""

from __future__ import annotations

{imports}

STAGES = [
{stages},
]


def main() -> None:
    for name, stage in STAGES:
        print(f"==> Running stage: {{name}}")
        stage()


if __name__ == "__main__":
    main()
'''

    @staticmethod
    def _test(modules: list[ModuleSpec]) -> str:
        imports = ", ".join(m.name for m in modules)
        return f'''"""Smoke test: every module imports and exposes run()."""

from src import {imports}


def test_modules_expose_run() -> None:
    for module in ({imports},):
        assert callable(module.run)
'''
