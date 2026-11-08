"""Pydantic domain models for a paper analysis.

Small local LLMs are inconsistent about output shapes (a list arrives as a
bullet string, a Q&A pair uses ``q``/``a`` keys, numbers arrive as "7/10" ...).
The ``mode="before"`` validators coerce those variations into the canonical
schema so that the UI can rely on clean, typed data.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DifficultyLevel = Literal["Beginner", "Intermediate", "Advanced"]
NOT_FOUND = "Not explicitly mentioned in the paper."

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)]|[a-z][.)])\s*")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


# ------------------------------------------------------------------ coercion helpers
def _item_to_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        values = [str(v).strip() for v in item.values() if v not in (None, "", [], {})]
        if len(values) == 2:
            return f"{values[0]}: {values[1]}"
        return " - ".join(values)
    if isinstance(item, (list, tuple)):
        return ", ".join(_item_to_text(i) for i in item)
    return "" if item is None else str(item).strip()


def coerce_str_list(value: Any) -> list[str]:
    """Accept list / bullet-string / dict / None and return a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        lines = [_BULLET.sub("", line).strip() for line in value.splitlines()]
        lines = [line for line in lines if line]
        if len(lines) == 1 and ";" in lines[0]:
            lines = [part.strip() for part in lines[0].split(";") if part.strip()]
        return lines
    if isinstance(value, dict):
        return [f"{k}: {_item_to_text(v)}" if v else str(k) for k, v in value.items()]
    if isinstance(value, (list, tuple)):
        return [text for text in (_BULLET.sub("", _item_to_text(i)) for i in value) if text]
    return [str(value)]


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        items = [_item_to_text(v) for v in value if _item_to_text(v)]
        return items[0] if len(items) == 1 else "\n".join(f"- {item}" for item in items)
    if isinstance(value, dict):
        return "\n".join(f"{k}: {_item_to_text(v)}" for k, v in value.items())
    return str(value)


def coerce_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(round(value))
    if isinstance(value, str):
        match = _NUMBER.search(value)
        if match:
            return int(round(float(match.group())))
    return default


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# ------------------------------------------------------------------------- models
class QAItem(_Base):
    question: str
    answer: str = ""

    @classmethod
    def coerce_list(cls, value: Any) -> list[dict[str, str]]:
        if value is None:
            return []
        if isinstance(value, dict):  # {"What is X?": "X is ..."}
            value = [{"question": k, "answer": v} for k, v in value.items()]
        if isinstance(value, str):
            value = coerce_str_list(value)
        items: list[dict[str, str]] = []
        for raw in value:
            if isinstance(raw, str):
                if raw.strip():
                    items.append({"question": raw.strip(), "answer": ""})
                continue
            if isinstance(raw, dict):
                lowered = {str(k).lower(): v for k, v in raw.items()}
                question = lowered.get("question") or lowered.get("q") or lowered.get("query") or ""
                answer = lowered.get("answer") or lowered.get("a") or lowered.get("expected_answer") or ""
                if question:
                    items.append({"question": coerce_text(question), "answer": coerce_text(answer)})
        return items


class ProjectIdea(_Base):
    title: str = ""
    description: str = ""
    features: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    deliverables: list[str] = Field(default_factory=list)

    _lists = field_validator("features", "tech_stack", "deliverables", mode="before")(coerce_str_list)
    _texts = field_validator("title", "description", mode="before")(coerce_text)


class BuildPlan(_Base):
    """The "Build this Paper as a Project" blueprint."""

    project_name: str = ""
    objective: str = ""
    architecture: str = ""
    modules: list[str] = Field(default_factory=list)
    setup_steps: list[str] = Field(default_factory=list)
    evaluation_metrics: list[str] = Field(default_factory=list)
    stretch_goals: list[str] = Field(default_factory=list)

    _lists = field_validator("modules", "setup_steps", "evaluation_metrics", "stretch_goals", mode="before")(
        coerce_str_list
    )
    _texts = field_validator("project_name", "objective", "architecture", mode="before")(coerce_text)


class RoadmapPhase(_Base):
    phase: str
    duration: str = ""
    tasks: list[str] = Field(default_factory=list)

    _tasks = field_validator("tasks", mode="before")(coerce_str_list)
    _texts = field_validator("phase", "duration", mode="before")(coerce_text)


class Assessment(_Base):
    difficulty_level: DifficultyLevel = "Intermediate"
    difficulty_score: int = Field(default=5, ge=1, le=10)
    difficulty_rationale: str = ""
    resume_impact_score: int = Field(default=5, ge=1, le=10)
    resume_impact_rationale: str = ""
    estimated_hours: int = Field(default=40, ge=1, le=2000)
    estimated_time_label: str = ""
    skills_gained: list[str] = Field(default_factory=list)

    @field_validator("difficulty_level", mode="before")
    @classmethod
    def _level(cls, value: Any) -> str:
        text = str(value or "").strip().lower()
        if text.startswith(("beg", "easy", "low")):
            return "Beginner"
        if text.startswith(("adv", "hard", "high", "expert")):
            return "Advanced"
        return "Intermediate"

    @field_validator("difficulty_score", "resume_impact_score", mode="before")
    @classmethod
    def _score(cls, value: Any) -> int:
        score = coerce_int(value, 5)
        if score > 10:  # model answered on a 0-100 scale
            score = round(score / 10)
        return min(10, max(1, score))

    @field_validator("estimated_hours", mode="before")
    @classmethod
    def _hours(cls, value: Any) -> int:
        return min(2000, max(1, coerce_int(value, 40)))

    _texts = field_validator(
        "difficulty_rationale", "resume_impact_rationale", "estimated_time_label", mode="before"
    )(coerce_text)
    _skills = field_validator("skills_gained", mode="before")(coerce_str_list)


class PaperAnalysis(_Base):
    """Complete, student-friendly analysis of one paper."""

    title: str = ""
    simple_summary: str = ""
    problem_statement: str = ""
    key_contributions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    methodology: str = ""
    implementation_steps: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    tools_frameworks: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list)
    project_idea: ProjectIdea = Field(default_factory=ProjectIdea)
    build_plan: BuildPlan = Field(default_factory=BuildPlan)
    roadmap: list[RoadmapPhase] = Field(default_factory=list)
    viva_questions: list[QAItem] = Field(default_factory=list)
    interview_questions: list[QAItem] = Field(default_factory=list)
    assessment: Assessment = Field(default_factory=Assessment)
    warnings: list[str] = Field(default_factory=list)

    _lists = field_validator(
        "key_contributions", "keywords", "implementation_steps", "datasets",
        "tools_frameworks", "limitations", "future_work", "warnings",
        mode="before",
    )(coerce_str_list)
    _texts = field_validator("title", "simple_summary", "problem_statement", "methodology", mode="before")(
        coerce_text
    )

    @field_validator("viva_questions", "interview_questions", mode="before")
    @classmethod
    def _qa(cls, value: Any) -> list[dict[str, str]]:
        return QAItem.coerce_list(value)

    @field_validator("roadmap", mode="before")
    @classmethod
    def _roadmap(cls, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, dict):
            value = [{"phase": k, "tasks": v} for k, v in value.items()]
        if isinstance(value, str):
            value = [{"phase": line} for line in coerce_str_list(value)]
        phases = []
        for index, raw in enumerate(value, start=1):
            if isinstance(raw, str):
                phases.append({"phase": raw})
            elif isinstance(raw, dict):
                lowered = {str(k).lower(): v for k, v in raw.items()}
                phases.append(
                    {
                        "phase": lowered.get("phase") or lowered.get("name") or lowered.get("title") or f"Phase {index}",
                        "duration": lowered.get("duration") or lowered.get("time") or lowered.get("weeks") or "",
                        "tasks": lowered.get("tasks") or lowered.get("steps") or lowered.get("activities") or [],
                    }
                )
        return phases

    @field_validator("project_idea", "build_plan", "assessment", mode="before")
    @classmethod
    def _nested(cls, value: Any) -> Any:
        if isinstance(value, str):  # model returned JSON as a string
            try:
                return json.loads(value)
            except ValueError:
                return {}
        return value or {}
