"""Prompt templates and the declarative list of analysis tasks.

Each ``AnalysisTask`` describes *what* to retrieve (queries + preferred
sections) and *what* to generate (instructions + JSON schema). The pipeline
executes them generically, so adding a new analysis section means adding one
entry here - no pipeline changes required.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

SYSTEM_PROMPT = """You are an expert research mentor who explains academic papers to undergraduate \
engineering students.
Rules:
1. Ground every statement in the provided paper excerpts. Never invent datasets, numbers, tools or results.
2. If the paper does not contain the requested information, write "Not explicitly mentioned in the paper."
3. Use simple language, short sentences and everyday analogies. Explain jargon the first time it appears.
4. Respond ONLY with one valid JSON object that follows the requested schema. No markdown, no commentary."""

TASK_TEMPLATE = """PAPER TITLE: {title}

PAPER OPENING (title, abstract, introduction):
\"\"\"
{lead}
\"\"\"

RELEVANT EXCERPTS (retrieved from the full paper):
\"\"\"
{context}
\"\"\"
{known_facts}
TASK:
{instructions}

Return a JSON object with exactly these keys:
{schema}"""

KNOWN_FACTS_TEMPLATE = """
ALREADY ESTABLISHED ABOUT THIS PAPER (stay consistent with it):
{facts}
"""

JSON_RETRY_SUFFIX = (
    "\n\nIMPORTANT: Your previous answer was not valid JSON. Reply with ONLY the JSON object, "
    "starting with '{' and ending with '}'."
)

CHAT_SYSTEM_PROMPT = """You are a friendly research assistant helping an engineering student understand \
the paper "{title}".
Answer using ONLY the paper excerpts provided in the user's message. Cite pages like (p. 4).
If the excerpts do not contain the answer, say so honestly and suggest what section to read.
Keep answers clear and concise; use bullet points or short code/pseudocode when it helps."""

CHAT_USER_TEMPLATE = """PAPER EXCERPTS:
\"\"\"
{context}
\"\"\"

QUESTION: {question}"""


@dataclass(frozen=True)
class AnalysisTask:
    key: str
    label: str
    queries: tuple[str, ...]
    preferred_sections: tuple[str, ...]
    instructions: str
    schema: str
    output_keys: tuple[str, ...]
    uses_known_facts: bool = False
    top_k: int = 6
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None


def _assessment_transform(data: dict[str, Any]) -> dict[str, Any]:
    """Flat assessment keys from the model -> nested ``assessment`` object."""
    assessment_keys = (
        "difficulty_level", "difficulty_score", "difficulty_rationale", "resume_impact_score",
        "resume_impact_rationale", "estimated_hours", "estimated_time_label", "skills_gained",
    )
    nested = dict(data.get("assessment") or {})
    nested.update({k: data[k] for k in assessment_keys if k in data})
    return {"roadmap": data.get("roadmap", []), "assessment": nested}


ANALYSIS_TASKS: tuple[AnalysisTask, ...] = (
    AnalysisTask(
        key="overview",
        label="Writing the simple summary & problem statement",
        queries=(
            "abstract summary of the paper and main idea",
            "what problem does this paper solve and why is it important",
            "main contributions of this work: we propose, we introduce, our contributions are",
        ),
        preferred_sections=("abstract", "introduction", "conclusion"),
        instructions=(
            "Explain the paper to a second-year engineering student.\n"
            "- simple_summary: 120-180 words in plain English, include one everyday analogy.\n"
            "- problem_statement: 50-100 words: what problem, why existing solutions fall short.\n"
            "- key_contributions: 3-6 short bullet strings, each one concrete contribution.\n"
            "- keywords: 5-8 technical keywords/topics of the paper."
        ),
        schema='{"simple_summary": "string", "problem_statement": "string", '
        '"key_contributions": ["string"], "keywords": ["string"]}',
        output_keys=("simple_summary", "problem_statement", "key_contributions", "keywords"),
    ),
    AnalysisTask(
        key="methodology",
        label="Explaining the methodology",
        queries=(
            "proposed method architecture and algorithm details",
            "system model, model design and how the approach works step by step",
            "training procedure, implementation details, hyperparameters",
        ),
        preferred_sections=("methodology", "implementation", "experiments"),
        instructions=(
            "- methodology: 150-250 words explaining HOW the approach works, step by step, in simple terms. "
            "Define each technical term in brackets the first time it appears.\n"
            "- implementation_steps: 5-10 ordered, actionable steps a student would follow to re-implement "
            "the core idea (start each with a verb)."
        ),
        schema='{"methodology": "string", "implementation_steps": ["string"]}',
        output_keys=("methodology", "implementation_steps"),
        top_k=7,
    ),
    AnalysisTask(
        key="resources",
        label="Finding datasets, tools & frameworks",
        queries=(
            "datasets used for experiments, benchmark, data collection, number of samples",
            "implemented using framework library tools software hardware GPU programming language",
            "experimental setup and evaluation metrics",
        ),
        preferred_sections=("experiments", "implementation", "results"),
        instructions=(
            "List ONLY items explicitly named in the excerpts.\n"
            "- datasets: each item as 'Name: one-line description (size/purpose if stated)'.\n"
            "- tools_frameworks: libraries, frameworks, languages, simulators, hardware mentioned "
            "(e.g. 'PyTorch', 'NS-3', 'MATLAB', 'NVIDIA V100 GPU').\n"
            "Use an empty list if nothing is mentioned."
        ),
        schema='{"datasets": ["string"], "tools_frameworks": ["string"]}',
        output_keys=("datasets", "tools_frameworks"),
    ),
    AnalysisTask(
        key="critique",
        label="Identifying limitations & future work",
        queries=(
            "limitations, drawbacks, weaknesses, assumptions of the approach",
            "future work, future directions, we plan to extend",
            "conclusion of the paper",
        ),
        preferred_sections=("limitations", "future_work", "conclusion", "results"),
        instructions=(
            "- limitations: 3-6 limitations. Prefer ones stated by the authors; you may add obvious ones "
            "implied by the method (e.g. dataset size, compute cost), prefixed with 'Implied: '.\n"
            "- future_work: 3-6 future research directions (stated by authors or natural extensions)."
        ),
        schema='{"limitations": ["string"], "future_work": ["string"]}',
        output_keys=("limitations", "future_work"),
    ),
    AnalysisTask(
        key="project",
        label="Designing a student project (Build this Paper)",
        queries=(
            "proposed method architecture components and modules",
            "implementation details dataset and evaluation metrics",
        ),
        preferred_sections=("methodology", "implementation", "experiments"),
        instructions=(
            "Design a realistic final-year student project inspired by this paper that can run on a "
            "laptop or free cloud GPU.\n"
            "- project_idea: a catchy title, a 60-100 word description, 4-6 features, the tech stack, "
            "and 3-5 deliverables (e.g. demo app, report, notebook).\n"
            "- build_plan: project_name (snake_case), objective (1-2 sentences), architecture (describe the "
            "components and data flow in 3-5 sentences), modules (4-7 items as 'module_name: purpose'), "
            "setup_steps (environment setup and data preparation steps), evaluation_metrics (how to measure "
            "success), stretch_goals (2-3 advanced extensions)."
        ),
        schema='{"project_idea": {"title": "string", "description": "string", "features": ["string"], '
        '"tech_stack": ["string"], "deliverables": ["string"]}, "build_plan": {"project_name": "string", '
        '"objective": "string", "architecture": "string", "modules": ["string"], "setup_steps": ["string"], '
        '"evaluation_metrics": ["string"], "stretch_goals": ["string"]}}',
        output_keys=("project_idea", "build_plan"),
        uses_known_facts=True,
        top_k=5,
    ),
    AnalysisTask(
        key="roadmap",
        label="Building the roadmap & scoring difficulty",
        queries=(
            "mathematical formulation equations theorem complexity of the method",
            "computational requirements training time hardware",
        ),
        preferred_sections=("methodology", "experiments"),
        instructions=(
            "- roadmap: 4-6 phases for implementing the student project. Each phase has a phase name, "
            "a duration (e.g. 'Week 1-2') and 3-5 concrete tasks.\n"
            "- difficulty_level: Beginner, Intermediate or Advanced for an undergraduate student.\n"
            "- difficulty_score: integer 1-10. difficulty_rationale: one sentence.\n"
            "- resume_impact_score: integer 1-10 (how impressive/relevant this project is to recruiters). "
            "resume_impact_rationale: one sentence.\n"
            "- estimated_hours: integer total hours for a student to build the project.\n"
            "- skills_gained: 4-6 skills a student will learn."
        ),
        schema='{"roadmap": [{"phase": "string", "duration": "string", "tasks": ["string"]}], '
        '"difficulty_level": "Beginner|Intermediate|Advanced", "difficulty_score": 5, '
        '"difficulty_rationale": "string", "resume_impact_score": 5, "resume_impact_rationale": "string", '
        '"estimated_hours": 40, "skills_gained": ["string"]}',
        output_keys=("roadmap", "assessment"),
        uses_known_facts=True,
        top_k=4,
        transform=_assessment_transform,
    ),
    AnalysisTask(
        key="viva",
        label="Generating viva questions",
        queries=(
            "core concepts and definitions used in the paper",
            "why this approach design choices compared to baselines",
            "results and performance comparison",
        ),
        preferred_sections=("methodology", "results", "introduction"),
        instructions=(
            "Write 8 viva (oral examination) questions an examiner would ask a student who presents this "
            "paper, from basic to deep. Each answer: 2-4 sentences, grounded in the paper."
        ),
        schema='{"viva_questions": [{"question": "string", "answer": "string"}]}',
        output_keys=("viva_questions",),
    ),
    AnalysisTask(
        key="interview",
        label="Generating interview questions",
        queries=(
            "technical concepts algorithms and models used",
            "implementation challenges trade-offs and evaluation",
        ),
        preferred_sections=("methodology", "implementation", "experiments"),
        instructions=(
            "Write 6 technical job-interview questions a recruiter could ask a candidate who lists a project "
            "based on this paper on their resume. Mix conceptual, practical and system-design questions. "
            "Each answer: 2-4 sentences."
        ),
        schema='{"interview_questions": [{"question": "string", "answer": "string"}]}',
        output_keys=("interview_questions",),
        uses_known_facts=True,
        top_k=5,
    ),
)
