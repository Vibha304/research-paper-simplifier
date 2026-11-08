"""The default zero-setup engine: full pipeline on a realistic paper, no LLM at all."""

from __future__ import annotations

import io
import zipfile

import pytest

from backend.container import build_container
from services.builtin_engine import BuiltinEngine
from services.domain_profiles import detect_domain
from services.nlp_toolkit import explicit_keywords, hashed_embedding, split_into_sentences
from tests.sample_paper import make_realistic_paper_pdf


@pytest.fixture()
def app(settings):
    return build_container(settings, BuiltinEngine())


@pytest.fixture()
def outcome(app):
    return app.pipeline.run(make_realistic_paper_pdf(), "lightids.pdf")


def test_default_container_uses_builtin_engine(settings):
    assert isinstance(build_container(settings).llm, BuiltinEngine)
    assert build_container(settings).llm.health_check().ready


def test_overview_is_grounded_in_the_paper(outcome):
    a = outcome.analysis
    assert not a.warnings, a.warnings
    assert "LightIDS" in a.title
    assert "intrusion" in a.simple_summary.lower()
    assert "Analogy" in a.simple_summary
    assert any(w in a.problem_statement.lower() for w in ("challenge", "suffer", "difficult", "cannot"))
    assert any("propose" in c.lower() or "introduce" in c.lower() for c in a.key_contributions)
    assert "Intrusion detection" in a.keywords  # taken from the explicit "Index Terms" line


def test_method_resources_and_critique(outcome):
    a = outcome.analysis
    assert "mutual information" in a.methodology.lower()
    assert "Jargon decoder" in a.methodology
    assert len(a.implementation_steps) >= 7
    dataset_names = " ".join(a.datasets)
    assert "CICIDS2017" in dataset_names and "UNSW-NB15" in dataset_names
    tools = " ".join(a.tools_frameworks)
    assert "PyTorch" in tools and "scikit-learn" in tools
    assert any("only on two offline datasets" in item for item in a.limitations)
    assert any("federated learning" in item.lower() or "live traffic" in item for item in a.future_work)


def test_project_roadmap_scores_and_questions(outcome, app):
    a = outcome.analysis
    assert a.project_idea.title and "Python" in a.project_idea.tech_stack
    assert len(a.build_plan.modules) >= 4 and a.build_plan.evaluation_metrics
    assert [p.phase for p in a.roadmap][0] == "Understand the paper"
    assert a.roadmap[-1].duration.startswith("Week")
    s = a.assessment
    assert s.difficulty_level in {"Beginner", "Intermediate", "Advanced"}
    assert 1 <= s.resume_impact_score <= 10 and s.estimated_hours > 10
    assert len(a.viva_questions) >= 8 and all(q.answer for q in a.viva_questions)
    assert len(a.interview_questions) >= 6
    name, data = app.project_kit.build_zip(a)
    assert zipfile.ZipFile(io.BytesIO(data)).testzip() is None and name.endswith(".zip")
    assert app.exporter.to_pdf(a).startswith(b"%PDF")


def test_chat_answers_from_the_paper_with_citations(outcome, app):
    answer = app.chat.ask(outcome.paper.id, outcome.analysis.title, "Which datasets were used for evaluation?")
    text = "".join(answer.tokens)
    assert "CICIDS2017" in text and "(p." in text
    unknown = app.chat.ask(outcome.paper.id, outcome.analysis.title, "What is the capital of France?")
    assert "couldn't find" in "".join(unknown.tokens)


def test_similarity_works_with_builtin_embeddings(app, outcome):
    assert app.similarity.search("intrusion detection for IoT devices")[0].paper_id == outcome.paper.id


def test_nlp_toolkit_basics():
    assert split_into_sentences("We follow Smith et al. in Fig. 2 for this. Then we train.") == [
        "We follow Smith et al. in Fig. 2 for this.", "Then we train."]
    assert hashed_embedding("intrusion detection") == hashed_embedding("intrusion detection")  # deterministic
    assert explicit_keywords("Index Terms—Deep learning, IoT, security.\nI. INTRODUCTION") == [
        "Deep learning", "IoT", "security"]
    assert detect_domain("intrusion attack security threat anomaly malware attack").name == "Cybersecurity"
