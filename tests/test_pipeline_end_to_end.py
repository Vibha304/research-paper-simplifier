"""Full pipeline with real PyMuPDF, ChromaDB and SQLite, and a fake LLM."""

from __future__ import annotations

import io
import zipfile

from backend.container import build_container
from tests.conftest import FakeLLM, make_paper_pdf


def test_full_analysis_flow(settings, fake_llm, paper_pdf):
    app = build_container(settings, fake_llm)
    updates: list[tuple[float, str]] = []

    outcome = app.pipeline.run(paper_pdf, "paper.pdf", lambda f, m: updates.append((f, m)))
    a = outcome.analysis

    # Every section was produced and coerced into the canonical schema.
    assert a.simple_summary and a.problem_statement and len(a.key_contributions) == 3
    assert a.implementation_steps == ["Download CIFAR-10", "Build the CNN in PyTorch", "Train for 50 epochs"]
    assert a.datasets[0] == "CIFAR-10: 60k small images"
    assert a.project_idea.title == "PocketVision"
    assert a.viva_questions[0].answer == "A sliding filter."
    assert a.assessment.difficulty_level in {"Beginner", "Intermediate", "Advanced"}
    assert a.roadmap[0].phase == "Setup"
    assert not a.warnings

    # Progress is monotonic and finishes at 100 %.
    fractions = [f for f, _ in updates]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0

    # Persisted to SQLite + ChromaDB.
    stored = app.library.load(outcome.paper.id)
    assert stored.analysis.title == a.title
    assert app.library.stats()["analyzed"] == 1

    # Re-uploading identical bytes reuses the stored analysis without LLM calls.
    calls_before = len(fake_llm.calls)
    again = app.pipeline.run(paper_pdf, "paper.pdf")
    assert again.reused and len(fake_llm.calls) == calls_before

    # Forced re-analysis keeps the same paper id and adds a new analysis row.
    forced = app.pipeline.run(paper_pdf, "paper.pdf", force=True)
    assert forced.paper.id == outcome.paper.id and forced.analysis_id != outcome.analysis_id

    # Chat streams, cites sources and persists both turns.
    answer = app.chat.ask(outcome.paper.id, a.title, "What method is proposed?")
    assert "".join(answer.tokens) == "The paper proposes a CNN (p. 1)."
    assert answer.sources and answer.sources[0]["pages"].startswith("p")
    assert [m.role for m in app.chat.history(outcome.paper.id)] == ["user", "assistant"]

    # Exports.
    markdown = app.exporter.to_markdown(a, filename="paper.pdf", page_count=3)
    assert "## 3. Key Contributions" in markdown and "PocketVision" in markdown
    pdf_bytes = app.exporter.to_pdf(a, filename="paper.pdf", page_count=3)
    assert pdf_bytes.startswith(b"%PDF") and len(pdf_bytes) > 2000

    # One-click project kit: valid zip, valid Python module names.
    name, data = app.project_kit.build_zip(a)
    files = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert name == "pocket_vision_starter_kit.zip"
    assert "pocket_vision/src/data_loader.py" in files
    assert "pocket_vision/src/module_class.py" in files  # 'class' keyword sanitised
    main_py = zipfile.ZipFile(io.BytesIO(data)).read("pocket_vision/src/main.py").decode()
    compile(main_py, "main.py", "exec")

    # Delete cascades to analyses/chat and removes vectors.
    app.library.delete(outcome.paper.id)
    assert app.library.stats()["papers"] == 0
    assert app.chat.history(outcome.paper.id) == []


def test_similarity_search_between_papers(settings, fake_llm):
    app = build_container(settings, fake_llm)
    first = app.pipeline.run(make_paper_pdf(pages=3), "a.pdf")
    second = app.pipeline.run(make_paper_pdf(pages=2), "b.pdf")
    similar = app.similarity.similar_to_paper(first.paper.id)
    assert [s.paper_id for s in similar] == [second.paper.id]
    assert 0 <= similar[0].percent <= 100
    assert app.similarity.search("lightweight CNN image recognition")


def test_pipeline_survives_a_failing_task(settings, paper_pdf):
    llm = FakeLLM(broken_tasks=["viva_questions"])
    app = build_container(settings, llm)
    outcome = app.pipeline.run(paper_pdf, "paper.pdf")
    assert outcome.analysis.viva_questions == []
    assert any("viva" in w for w in outcome.analysis.warnings)
    assert outcome.analysis.simple_summary  # other sections unaffected
    assert llm.calls.count("viva_questions") == 2  # original attempt + one JSON retry
