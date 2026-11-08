"""Export an analysis to Markdown and to a styled PDF (fpdf2, offline)."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from fpdf import FPDF

from backend.models import NOT_FOUND, PaperAnalysis, QAItem
from utils.exceptions import ExportError
from utils.logger import get_logger
from utils.text import to_latin1

logger = get_logger(__name__)

# Unicode TTF fonts tried in order; falls back to Latin-1 core fonts.
_FONT_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    ("/Library/Fonts/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
)
_PROJECT_FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"


# ============================================================== Markdown
def _bullets(items: list[str], empty: str = NOT_FOUND) -> str:
    return "\n".join(f"- {item}" for item in items) if items else f"_{empty}_"


def _numbered(items: list[str], empty: str = NOT_FOUND) -> str:
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1)) if items else f"_{empty}_"


def _qa_block(items: list[QAItem]) -> str:
    if not items:
        return f"_{NOT_FOUND}_"
    return "\n\n".join(
        f"**Q{i}. {qa.question}**\n\n{qa.answer or '_Answer not generated._'}" for i, qa in enumerate(items, 1)
    )


def analysis_to_markdown(analysis: PaperAnalysis, *, filename: str = "", page_count: int | None = None) -> str:
    a = analysis
    s = a.assessment
    meta = [f"*Generated {datetime.now():%Y-%m-%d %H:%M} by Research Paper Simplifier (offline)*"]
    if filename:
        meta.append(f"*Source file: {filename}" + (f" - {page_count} pages*" if page_count else "*"))

    roadmap = "\n\n".join(
        f"**{phase.phase}**" + (f" ({phase.duration})" if phase.duration else "") + "\n" + _bullets(phase.tasks, "-")
        for phase in a.roadmap
    ) or f"_{NOT_FOUND}_"
    p = a.project_idea
    b = a.build_plan

    sections = [
        f"# {a.title or 'Research Paper Analysis'}",
        "\n\n".join(meta),
        "## At a Glance",
        "| Metric | Value |\n|---|---|\n"
        f"| Difficulty | {s.difficulty_level} ({s.difficulty_score}/10) |\n"
        f"| Resume impact | {s.resume_impact_score}/10 |\n"
        f"| Estimated implementation time | {s.estimated_time_label or f'{s.estimated_hours} hours'} |\n"
        f"| Keywords | {', '.join(a.keywords) or '-'} |",
        "## 1. Simple Summary", a.simple_summary or f"_{NOT_FOUND}_",
        "## 2. Problem Statement", a.problem_statement or f"_{NOT_FOUND}_",
        "## 3. Key Contributions", _bullets(a.key_contributions),
        "## 4. Methodology Explained Simply", a.methodology or f"_{NOT_FOUND}_",
        "## 5. Implementation Steps", _numbered(a.implementation_steps),
        "## 6. Datasets Used", _bullets(a.datasets),
        "## 7. Tools & Frameworks", _bullets(a.tools_frameworks),
        "## 8. Limitations", _bullets(a.limitations),
        "## 9. Future Work", _bullets(a.future_work),
        "## 10. Student Project Idea",
        f"### {p.title or 'Project idea'}\n\n{p.description or NOT_FOUND}\n\n"
        f"**Features**\n{_bullets(p.features, '-')}\n\n**Tech stack:** {', '.join(p.tech_stack) or '-'}\n\n"
        f"**Deliverables**\n{_bullets(p.deliverables, '-')}",
        "## 11. Build This Paper as a Project",
        f"**Project:** `{b.project_name or 'paper_project'}`\n\n**Objective:** {b.objective or '-'}\n\n"
        f"**Architecture:** {b.architecture or '-'}\n\n**Modules**\n{_bullets(b.modules, '-')}\n\n"
        f"**Setup steps**\n{_numbered(b.setup_steps, '-')}\n\n"
        f"**Evaluation metrics**\n{_bullets(b.evaluation_metrics, '-')}\n\n"
        f"**Stretch goals**\n{_bullets(b.stretch_goals, '-')}",
        "## 12. Implementation Roadmap", roadmap,
        "## 13. Scores",
        f"**Difficulty:** {s.difficulty_level} - {s.difficulty_score}/10. {s.difficulty_rationale}\n\n"
        f"**Resume impact:** {s.resume_impact_score}/10. {s.resume_impact_rationale}\n\n"
        f"**Estimated time:** {s.estimated_time_label}\n\n**Skills you will gain**\n{_bullets(s.skills_gained, '-')}",
        "## 14. Viva Questions", _qa_block(a.viva_questions),
        "## 15. Interview Questions", _qa_block(a.interview_questions),
    ]
    if a.warnings:
        sections += ["## Notes", _bullets(a.warnings)]
    return "\n\n".join(sections).strip() + "\n"


# ============================================================== PDF
class _AnalysisPDF(FPDF):
    def __init__(self, title: str) -> None:
        super().__init__(format="A4")
        self.doc_title = title
        self.unicode = self._register_unicode_font()
        self.family = "DocFont" if self.unicode else "Helvetica"
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)

    def _register_unicode_font(self) -> bool:
        candidates = [(str(_PROJECT_FONT_DIR / "DejaVuSans.ttf"), str(_PROJECT_FONT_DIR / "DejaVuSans-Bold.ttf"))]
        candidates += list(_FONT_CANDIDATES)
        for regular, bold in candidates:
            if Path(regular).is_file() and Path(bold).is_file():
                try:
                    self.add_font("DocFont", "", regular)
                    self.add_font("DocFont", "B", bold)
                    self.add_font("DocFont", "I", regular)
                    self.add_font("DocFont", "BI", bold)
                    return True
                except Exception as exc:  # pragma: no cover - font file issues
                    logger.debug("Font %s unusable: %s", regular, exc)
        return False

    def txt(self, text: str) -> str:
        text = text.replace("--", "-").replace("__", "_")  # avoid fpdf2 markdown underline/italic toggles
        return text if self.unicode else to_latin1(text)

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font(self.family, "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, self.txt(self.doc_title[:95]), align="L", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font(self.family, "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, f"Research Paper Simplifier  |  Page {self.page_no()}/{{nb}}", align="C")


_INLINE_CODE = re.compile(r"`([^`]*)`")
_ITALIC_UNDERSCORE = re.compile(r"(?<!\w)_(.+?)_(?!\w)")


def markdown_to_pdf(markdown: str, title: str) -> bytes:
    """Render the subset of Markdown produced by ``analysis_to_markdown``."""
    try:
        pdf = _AnalysisPDF(title)
        pdf.alias_nb_pages()
        pdf.add_page()
        accent = (37, 99, 235)

        for raw_line in markdown.splitlines():
            line = _ITALIC_UNDERSCORE.sub(r"\1", _INLINE_CODE.sub(r"\1", raw_line.rstrip()))
            if not line.strip():
                pdf.ln(2)
                continue
            if line.startswith("|"):
                if set(line.replace("|", "").strip()) <= {"-"}:
                    continue
                cells = [c.strip() for c in line.strip("|").split("|")]
                if cells[0].lower() == "metric":  # table header row
                    continue
                pdf.set_font(pdf.family, "", 10)
                pdf.set_text_color(30, 30, 30)
                pdf.multi_cell(0, 6, pdf.txt(f"**{cells[0]}:** {' | '.join(cells[1:])}"),
                               markdown=True, new_x="LMARGIN", new_y="NEXT")
                continue
            if line.startswith("# "):
                pdf.set_font(pdf.family, "B", 18)
                pdf.set_text_color(*accent)
                pdf.multi_cell(0, 9, pdf.txt(line[2:]), new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)
                continue
            if line.startswith("## "):
                pdf.ln(3)
                pdf.set_font(pdf.family, "B", 14)
                pdf.set_text_color(*accent)
                pdf.multi_cell(0, 8, pdf.txt(line[3:]), new_x="LMARGIN", new_y="NEXT")
                pdf.set_draw_color(*accent)
                pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
                pdf.ln(2)
                continue
            if line.startswith("### "):
                pdf.set_font(pdf.family, "B", 12)
                pdf.set_text_color(40, 40, 40)
                pdf.multi_cell(0, 7, pdf.txt(line[4:]), new_x="LMARGIN", new_y="NEXT")
                continue

            pdf.set_font(pdf.family, "", 10.5)
            pdf.set_text_color(30, 30, 30)
            bullet = re.match(r"^(\s*)(?:[-*]|(\d+)\.)\s+(.*)$", line)
            if bullet:
                marker = f"{bullet.group(2)}." if bullet.group(2) else ("\u2022" if pdf.unicode else "-")
                indent = 4 + len(bullet.group(1)) * 2
                pdf.set_x(pdf.l_margin + indent)
                pdf.cell(6, 6, marker)
                pdf.multi_cell(pdf.epw - indent - 6, 6, pdf.txt(bullet.group(3)), markdown=True,
                               new_x="LMARGIN", new_y="NEXT")
            else:
                pdf.multi_cell(0, 6, pdf.txt(line.strip("*") if line.startswith("*Generated") else line),
                               markdown=True, new_x="LMARGIN", new_y="NEXT")
        return bytes(pdf.output())
    except Exception as exc:
        raise ExportError(f"PDF rendering failed: {exc}") from exc


class ExportService:
    """Facade used by the UI."""

    def to_markdown(self, analysis: PaperAnalysis, *, filename: str = "", page_count: int | None = None) -> str:
        return analysis_to_markdown(analysis, filename=filename, page_count=page_count)

    def to_pdf(self, analysis: PaperAnalysis, *, filename: str = "", page_count: int | None = None) -> bytes:
        markdown = self.to_markdown(analysis, filename=filename, page_count=page_count)
        return markdown_to_pdf(markdown, analysis.title or "Research Paper Analysis")
