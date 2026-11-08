"""Built-in analysis engine: no Ollama, no model downloads, no internet, no API keys.

It implements the same ``LLMClient`` interface as the Ollama client (so the rest
of the app is unchanged) plus two structured capabilities:

* ``generate_task``   - produces each analysis section directly from the paper
                        using extractive summarisation (centroid + cue-phrase
                        sentence ranking), information extraction (datasets,
                        tools, metrics, glossary terms) and domain-aware
                        templates (project kit, roadmap, viva/interview Qs);
* ``answer_question`` - chat answers composed of the most relevant sentences of
                        the retrieved excerpts, with page citations.

Everything it writes is either quoted/condensed from the paper itself or a
clearly templated suggestion, so it cannot hallucinate facts about the paper.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from backend.models import NOT_FOUND, PaperAnalysis
from services.chunking import Chunk
from services.domain_profiles import DomainProfile, detect_domain
from services.llm_service import HealthStatus, Message
from services.nlp_toolkit import (
    Sentence,
    cosine,
    cue_score,
    explicit_keywords,
    glossary_terms,
    hashed_embedding,
    is_prose,
    keyphrases,
    rank_sentences,
    simplify_sentence,
    split_into_sentences,
    tokenize,
)
from services.rag_service import RetrievedChunk
from services.scoring_service import KNOWN_DATASETS, KNOWN_TOOLS, ScoringService, find_mentions
from utils.text import dedupe_preserve_order, slugify, truncate

EMBEDDING_DIM = 512

# ---------------------------------------------------------------- cue lexicons
PROBLEM_CUES = {"problem": 1.0, "challenge": 1.0, "however": 0.7, "difficult": 0.8, "limited": 0.6, "lack": 0.8,
                "suffer": 0.9, "existing": 0.5, "remain": 0.6, "issue": 0.7, "bottleneck": 0.9, "costly": 0.6,
                "inefficient": 0.8, "address": 0.6, "motivat": 0.6, "struggle": 0.8, "fail": 0.6, "expensive": 0.6}
CONTRIBUTION_CUES = {"we propose": 1.5, "we present": 1.3, "we introduce": 1.3, "we develop": 1.2, "we design": 1.2,
                     "our contribution": 1.5, "contributions": 1.0, "this paper": 0.7, "we show": 1.0,
                     "we demonstrate": 1.0, "novel": 0.8, "outperform": 0.8, "achiev": 0.5, "new ": 0.4,
                     "this work": 0.7, "we achieve": 1.0}
METHOD_CUES = {"propose": 0.4, "consist": 0.6, "first": 0.4, "then": 0.4, "finally": 0.4, "step": 0.6, "module": 0.5,
               "layer": 0.4, "algorithm": 0.5, "architecture": 0.6, "input": 0.4, "output": 0.4, "train": 0.4,
               "compute": 0.3, "stage": 0.5, "component": 0.5, "framework": 0.4, "based on": 0.3}
SEQUENCE_CUES = {"first": 1.0, "then": 0.8, "next": 0.8, "finally": 0.8, "step": 1.0, "stage": 0.8, "phase": 0.6,
                 "afterwards": 0.8, "subsequently": 0.8}
LIMITATION_CUES = {"limitation": 1.5, "drawback": 1.3, "however": 0.6, "do not": 0.6, "does not": 0.6, "cannot": 0.8,
                   "only": 0.4, "assum": 0.8, "restricted": 0.8, "fail": 0.8, "weakness": 1.2, "not able": 0.8,
                   "expensive": 0.6, "overhead": 0.6, "trade-off": 0.6, "unable": 0.8, "shortcoming": 1.3,
                   "may not": 0.6, "evaluated only": 0.8, "not generali": 0.8, "small dataset": 0.6}
FUTURE_CUES = {"future work": 2.0, "in the future": 1.5, "future research": 1.5, "we plan": 1.5, "we intend": 1.3,
               "could be extended": 1.3, "can be extended": 1.3, "further": 0.6, "extend": 0.8, "next step": 1.2,
               "open question": 1.0, "remains to be": 1.0, "future direction": 1.5, "would like to": 0.8}
RESULT_CUES = {"outperform": 1.2, "achiev": 1.0, "improv": 0.8, "accuracy": 0.6, "%": 0.6, "result": 0.5,
               "compared": 0.6, "better": 0.6, "state-of-the-art": 0.8, "reduc": 0.5, "faster": 0.6}
METRIC_TERMS = ("accuracy", "precision", "recall", "F1-score", "F1", "AUC", "ROC", "RMSE", "MAE", "MSE", "BLEU",
                "ROUGE", "perplexity", "PSNR", "SSIM", "mAP", "IoU", "top-1", "top-5", "throughput", "latency",
                "delay", "packet delivery ratio", "energy consumption", "BER", "SNR", "error rate", "FLOPs",
                "inference time", "training time", "detection rate", "false positive rate")

_DATASET_PATTERN = re.compile(
    r"\b((?:[A-Z][\w\-]*|\d+[\w\-]*)(?:[ \-](?:[A-Z0-9][\w\-]*)){0,3})\s+(?:dataset|data set|benchmark|corpus)s?\b"
)
_NON_NAMES = {"The", "This", "These", "Our", "Each", "Both", "All", "Large", "Public", "Real", "Synthetic", "Standard",
              "Two", "Three", "Several", "Different", "Popular", "New", "Original", "Training", "Test", "Validation",
              "In", "On", "For", "A", "An", "We", "Existing", "Benchmark", "Other", "Small", "Same", "Such"}

NOT_PIP_INSTALLABLE = {"Python", "C++", "Java", "R", "Julia", "MATLAB", "Simulink", "LabVIEW", "Raspberry Pi",
                       "Arduino", "FPGA", "Verilog", "VHDL", "NVIDIA", "CUDA", "cuDNN", "AWS", "Google Colab",
                       "Android", "Docker", "Kubernetes", "ANSYS", "COMSOL", "NS-2", "NS-3", "OMNeT++", "Cooja",
                       "Contiki", "Gazebo", "ROS", "GPT", "BERT", "ResNet", "LLaMA", "YOLO", "Stable Diffusion"}
_FRONT_MATTER_NOISE = re.compile(r"^(index terms|keywords|key words)\b", re.IGNORECASE)
_META_SENTENCE = re.compile(r"following contributions|as follows|organi[sz]ed as follows|rest of (this|the) paper|"
                            r"remainder of (this|the) paper", re.IGNORECASE)
_ABSTRACT_PREFIX = re.compile(r"^.{0,250}?\babstract\s*[^\w\s]?\s*(?=[A-Z])", re.IGNORECASE)
_WE_VERB = re.compile(r"^(we|this paper|this work)\s+(propose|proposes|introduce|introduces|present|presents|develop|design|demonstrate)s?\s+(that\s+)?(a |an |the )?", re.IGNORECASE)

SECTION_GROUPS = {
    "intro": ("abstract", "front_matter", "introduction"),
    "method": ("methodology", "implementation"),
    "results": ("experiments", "results"),
    "ending": ("conclusion", "limitations", "future_work", "results", "experiments"),
}


# ---------------------------------------------------------------- task input
@dataclass(frozen=True)
class TaskInput:
    """Everything the engine may use to produce one analysis section."""

    paper_id: str
    task_key: str
    title: str
    lead: str
    retrieved: list[RetrievedChunk]
    all_chunks: list[Chunk]
    page_count: int
    analysis: PaperAnalysis


@runtime_checkable
class StructuredGenerator(Protocol):
    """Optional capability: engines that build sections without free-form prompting."""

    def generate_task(self, task: TaskInput) -> dict[str, Any]: ...

    def answer_question(self, question: str, chunks: Sequence[RetrievedChunk], title: str) -> Iterator[str]: ...


# ---------------------------------------------------------------- paper digest
@dataclass
class PaperDigest:
    """Per-paper facts extracted once and shared by all eight tasks."""

    title: str
    sentences: list[Sentence]
    full_text: str
    page_count: int
    domain: DomainProfile
    keywords: list[str]
    glossary: list[tuple[str, str]]
    datasets: list[str]
    tools: list[str]
    metrics: list[str]
    by_section: dict[str, list[Sentence]] = field(default_factory=dict)

    def section(self, *groups: str) -> list[Sentence]:
        names = {name for group in groups for name in SECTION_GROUPS.get(group, (group,))}
        return [s for s in self.sentences if s.section in names]

    def page_range(self, sentences: Sequence[Sentence]) -> str:
        pages = sorted({s.page for s in sentences if s.page})
        if not pages:
            return ""
        return f"p. {pages[0]}" if pages[0] == pages[-1] else f"pp. {pages[0]}-{pages[-1]}"


def build_digest(title: str, chunks: Sequence[Chunk], page_count: int) -> PaperDigest:
    sentences: list[Sentence] = []
    seen: set[str] = set()
    for chunk in sorted(chunks, key=lambda c: c.index):
        for raw in split_into_sentences(chunk.text):
            raw = _ABSTRACT_PREFIX.sub("", raw)  # "Title ... Abstract— The rapid growth..." -> "The rapid growth..."
            if title and raw.startswith(title):
                raw = raw[len(title):].lstrip(" .:-")
            key = raw.lower()[:120]
            if key in seen or _is_noise(raw):
                continue
            seen.add(key)
            sentences.append(Sentence(raw, chunk.section, chunk.page_start, len(sentences)))
    full_text = "\n".join(c.text for c in sorted(chunks, key=lambda c: c.index))
    keywords = explicit_keywords(full_text) or keyphrases(full_text, top_n=8)
    return PaperDigest(
        title=title,
        sentences=sentences,
        full_text=full_text,
        page_count=page_count,
        domain=detect_domain(f"{title} {title} {full_text}"),
        keywords=keywords,
        glossary=glossary_terms(full_text, limit=6),
        datasets=_find_datasets(full_text),
        tools=find_mentions(full_text, KNOWN_TOOLS),
        metrics=_find_metrics(full_text),
    )


def _is_noise(sentence: str) -> bool:
    """Front-matter lines, 'the paper is organised as follows' sentences and non-prose."""
    return bool(_FRONT_MATTER_NOISE.match(sentence) or _META_SENTENCE.search(sentence)) or not is_prose(sentence)


def _lower_first(text: str) -> str:
    """Lower-case the first letter unless the first word is an acronym/name (e.g. 'LightIDS')."""
    first = text.split(" ", 1)[0]
    return text if any(c.isupper() for c in first[1:]) else text[:1].lower() + text[1:]


def _find_datasets(text: str) -> list[str]:
    names = find_mentions(text, KNOWN_DATASETS)
    for match in _DATASET_PATTERN.finditer(text):
        candidate = match.group(1).strip(" -")
        first = candidate.split()[0]
        if first in _NON_NAMES or len(candidate) < 3 or candidate.lower() in {n.lower() for n in names}:
            continue
        if any(ch.isupper() for ch in candidate[1:]) or any(ch.isdigit() for ch in candidate):
            names.append(candidate)  # e.g. "CIFAR-10", "UNSW-NB15", "KDD Cup"
    return dedupe_preserve_order(names)[:8]


def _find_metrics(text: str) -> list[str]:
    found = []
    for term in METRIC_TERMS:
        flags = 0 if term.isupper() or term in {"mAP", "IoU"} else re.IGNORECASE
        if re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", text, flags):
            found.append(term)
    if "F1-score" in found and "F1" in found:
        found.remove("F1")
    return found[:6]


def _context_sentence(digest: PaperDigest, name: str) -> str:
    for sentence in digest.sentences:
        if name.lower() in sentence.text.lower():
            return truncate(simplify_sentence(sentence.text), 150)
    return ""


def _bullets_from(sentences: Sequence[Sentence], max_chars: int = 240) -> list[str]:
    return [f"{simplify_sentence(s.text, max_chars)} {s.cite}".strip() for s in sentences]


# ---------------------------------------------------------------- the engine
class BuiltinEngine:
    """Zero-setup engine implementing ``LLMClient`` + ``StructuredGenerator``."""

    model = "built-in (no AI model needed)"
    embedding_model = f"builtin-hash-{EMBEDDING_DIM}"

    def __init__(self, cache_size: int = 8) -> None:
        self._digests: OrderedDict[str, PaperDigest] = OrderedDict()
        self._cache_size = cache_size
        self._scoring = ScoringService()

    # ----------------------------------------------------- LLMClient interface
    def health_check(self) -> HealthStatus:
        return HealthStatus(reachable=True, available_models=[self.model, self.embedding_model])

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [hashed_embedding(_strip_task_prefix(text), EMBEDDING_DIM) for text in texts]

    def chat(self, messages: list[Message], *, json_mode: bool = False, temperature: float | None = None) -> str:
        """Generic fallback (the app uses the structured methods instead)."""
        text = messages[-1]["content"] if messages else ""
        sentences = [Sentence(s, "body", 0, i) for i, s in enumerate(split_into_sentences(text)) if is_prose(s)]
        best = rank_sentences(sentences, top_n=3)
        return " ".join(s.text for s in best) or "I could not find enough text to answer."

    def stream_chat(self, messages: list[Message], *, temperature: float | None = None) -> Iterator[str]:
        yield from _stream_words(self.chat(messages))

    # ----------------------------------------------------- structured methods
    def generate_task(self, task: TaskInput) -> dict[str, Any]:
        digest = self._digest(task)
        handler = getattr(self, f"_task_{task.task_key}", None)
        if handler is None:
            raise ValueError(f"Built-in engine has no generator for task '{task.task_key}'")
        return handler(digest, task)

    def answer_question(self, question: str, chunks: Sequence[RetrievedChunk], title: str) -> Iterator[str]:
        candidates: list[Sentence] = []
        for chunk in chunks:
            for raw in split_into_sentences(chunk.text):
                raw = _ABSTRACT_PREFIX.sub("", raw)
                if not _is_noise(raw):
                    candidates.append(Sentence(raw, chunk.section, chunk.page_start, len(candidates)))
        question_vector = hashed_embedding(question)
        question_terms = set(tokenize(question))
        scored = []
        for sentence in candidates:
            overlap = len(question_terms & set(tokenize(sentence.text))) / max(1, len(question_terms))
            scored.append((cosine(hashed_embedding(sentence.text), question_vector) + 0.6 * overlap, sentence))
        scored.sort(key=lambda item: item[0], reverse=True)

        if not scored or scored[0][0] < 0.12:
            answer = (
                "I couldn't find a clear answer to that in the paper's text. Try rephrasing with words the "
                "paper is likely to use (for example a method name, dataset or metric), or check the "
                "Methodology and Experiments sections."
            )
            yield from _stream_words(answer)
            return

        best: list[Sentence] = []
        for score, sentence in scored:
            if score < scored[0][0] * 0.55 or len(best) >= 4:
                break
            if all(cosine(hashed_embedding(sentence.text), hashed_embedding(b.text)) < 0.8 for b in best):
                best.append(sentence)
        best.sort(key=lambda s: s.position)

        lines = ["Here is what the paper says about this:", ""]
        lines += [f"- {simplify_sentence(s.text, 300)} {s.cite}" for s in best]
        wants_explanation = re.search(r"\b(what is|what are|explain|meaning|define|beginner|simple)\b", question, re.I)
        terms = glossary_terms(question, limit=2)
        if wants_explanation and not terms:
            terms = glossary_terms(" ".join(s.text for s in best), limit=2)
        if terms:
            lines += ["", "**Jargon decoder:**"] + [f"- **{term}**: {meaning}" for term, meaning in terms]
        lines += ["", "_Answer extracted from the paper by the built-in engine (no AI model)._"]
        yield from _stream_words("\n".join(lines))

    # ----------------------------------------------------- digest cache
    def _digest(self, task: TaskInput) -> PaperDigest:
        key = f"{task.paper_id}:{len(task.all_chunks)}"
        if key not in self._digests:
            self._digests[key] = build_digest(task.title, task.all_chunks, task.page_count)
            while len(self._digests) > self._cache_size:
                self._digests.popitem(last=False)
        self._digests.move_to_end(key)
        return self._digests[key]

    # ----------------------------------------------------- section generators
    def _task_overview(self, d: PaperDigest, task: TaskInput) -> dict[str, Any]:
        intro = d.section("intro") or d.sentences[:25]
        abstract = [s for s in intro if s.section in ("abstract", "front_matter")] or intro[:10]
        summary_sentences = rank_sentences(abstract, reference=" ".join(s.text for s in abstract), top_n=4)
        topic = ", ".join(k if k[:2].isupper() else k.lower() for k in d.keywords[:3]) or d.domain.name.lower()
        summary_parts = [f"**In one line:** this paper is about {topic}."]
        summary_parts.append(" ".join(simplify_sentence(s.text, 280) for s in summary_sentences))
        summary_parts.append(f"**Analogy:** {d.domain.analogy}")
        if d.glossary:
            summary_parts.append("**Key terms:** " + "; ".join(f"*{t}* - {m}" for t, m in d.glossary[:2]) + ".")

        not_contributions = [s for s in intro if cue_score(s.text, CONTRIBUTION_CUES) < 1.0]
        problem = rank_sentences(not_contributions, cues=PROBLEM_CUES, min_cue=0.6, top_n=3)
        if not problem:
            problem = intro[:2]

        contributions = rank_sentences(d.section("intro"), cues=CONTRIBUTION_CUES, min_cue=0.7, top_n=5)
        if len(contributions) < 3:
            contributions += rank_sentences(d.section("conclusion"), cues=CONTRIBUTION_CUES, min_cue=0.7,
                                            top_n=3 - len(contributions))
        contributions = contributions or summary_sentences[:3]
        return {
            "simple_summary": "\n\n".join(p for p in summary_parts if p.strip()),
            "problem_statement": " ".join(simplify_sentence(s.text, 300) for s in problem) or NOT_FOUND,
            "key_contributions": _bullets_from(contributions),
            "keywords": d.keywords[:8],
        }

    def _task_methodology(self, d: PaperDigest, task: TaskInput) -> dict[str, Any]:
        method = d.section("method")
        if len(method) < 4:  # paper without a recognisable "Method" heading
            excluded = {"abstract", "front_matter", "introduction", "related_work", "conclusion", "results"}
            method = [s for s in d.sentences if s.section not in excluded] or d.sentences
        chosen = rank_sentences(method, cues=METHOD_CUES, top_n=7)
        method_text = " ".join(s.text for s in method)
        terms = glossary_terms(method_text, limit=4)
        pages = d.page_range(method)

        paragraphs = [
            f"**The big picture:** the authors' approach is described in the method section ({pages}). "
            "Here are its most important points, in reading order:" if pages else
            "**The big picture:** here are the most important points of the approach, in reading order:",
            "\n".join(f"{i}. {simplify_sentence(s.text, 300)} {s.cite}" for i, s in enumerate(chosen, 1)),
        ]
        if terms:
            paragraphs.append("**Jargon decoder:**\n" + "\n".join(f"- **{t}**: {m}" for t, m in terms))
        return {
            "methodology": "\n\n".join(paragraphs),
            "implementation_steps": self._implementation_steps(d, method, pages),
        }

    def _implementation_steps(self, d: PaperDigest, method: list[Sentence], pages: str) -> list[str]:
        installable = [t for t in d.tools if t not in NOT_PIP_INSTALLABLE]
        tools = installable[:5] or [t for t in d.domain.tech_stack if t not in NOT_PIP_INSTALLABLE][:4]
        steps = [
            f"Read the abstract, introduction and method section{f' ({pages})' if pages else ''} and sketch the "
            "system as a block diagram (inputs, components, outputs).",
            f"Set up a Python virtual environment and install: {', '.join(tools)}.",
            f"Get the data: {', '.join(d.datasets[:2])} (or a smaller public alternative to start)."
            if d.datasets else "Find or create a small dataset that matches the paper's setting (start small!).",
            "Pre-process the data exactly as the paper describes (cleaning, normalisation, train/test split).",
        ]
        components = [k for k in keyphrases(" ".join(s.text for s in method), top_n=6) if len(k) > 4][:3]
        for component in components:
            steps.append(f"Implement the '{component}' component and unit-test it on a tiny input.")
        ordered = rank_sentences(method, cues=SEQUENCE_CUES, min_cue=1.0, top_n=2)
        steps += [f"Follow the paper's procedure: {simplify_sentence(s.text, 160)} {s.cite}" for s in ordered]
        metrics = d.metrics[:3] or list(d.domain.metrics[:2])
        steps += [
            f"Run the experiments and record {', '.join(metrics)}.",
            "Compare against a simple baseline and try to reproduce one key table or figure from the paper.",
            "Summarise results with plots and a short report; note where your numbers differ and why.",
        ]
        return steps

    def _task_resources(self, d: PaperDigest, task: TaskInput) -> dict[str, Any]:
        return {"datasets": self._with_context(d, d.datasets), "tools_frameworks": self._with_context(d, d.tools)}

    @staticmethod
    def _with_context(d: PaperDigest, names: list[str]) -> list[str]:
        """'Name: sentence where it is mentioned' - each context sentence shown only once."""
        items, used = [], set()
        for name in names:
            context = _context_sentence(d, name)
            if context and context not in used:
                used.add(context)
                items.append(f"{name}: {context}")
            else:
                items.append(name)
        return items

    def _task_critique(self, d: PaperDigest, task: TaskInput) -> dict[str, Any]:
        excluded = {"introduction", "related_work", "abstract", "front_matter"}
        body = [s for s in d.sentences if s.section not in excluded]
        limitations = _bullets_from(rank_sentences(body, cues=LIMITATION_CUES, min_cue=0.8, top_n=4))
        implied = [
            f"Implied: evaluated on {len(d.datasets)} dataset(s) only - results may not generalise to other data."
            if 0 < len(d.datasets) <= 2 else "",
            "Implied: compute/hardware requirements may be significant for a student setup."
            if d.domain.base_difficulty >= 0.9 else "",
            "Implied: real-world deployment (robustness, cost, maintenance) is not fully validated.",
        ]
        limitations += [item for item in implied if item][: max(0, 4 - len(limitations))]

        future = _bullets_from(rank_sentences(d.sentences, cues=FUTURE_CUES, min_cue=0.8, top_n=4))
        if len(future) < 3:
            topic = d.keywords[0] if d.keywords else "the method"
            suggestions = [
                f"Suggested: evaluate {topic} on additional, larger or real-world datasets.",
                "Suggested: reduce computational cost so the approach runs on low-end or edge devices.",
                f"Suggested: combine the approach with recent advances in {d.domain.name.split('/')[0].strip().lower()}.",
                "Suggested: add explainability/visualisation of why the method makes its decisions.",
            ]
            future += suggestions[: 4 - len(future)]
        return {"limitations": limitations, "future_work": future}

    def _task_project(self, d: PaperDigest, task: TaskInput) -> dict[str, Any]:
        a = task.analysis
        core = d.keywords[0] if d.keywords else d.domain.name
        short_title = truncate(d.title, 70)
        tech = dedupe_preserve_order(["Python"] + d.tools[:4] + list(d.domain.tech_stack))[:6]
        name = slugify(f"{core} project", 40, "_")
        features = [_WE_VERB.sub("", re.sub(r"\s*\(p\. \d+\)$", "", c)) for c in a.key_contributions[:2]]
        features = [f"Implement {_lower_first(simplify_sentence(f, 150).rstrip('.'))}" for f in features if f] + [
            f"Experiments reporting {', '.join((d.metrics or list(d.domain.metrics))[:3])}",
            "Comparison with a simple baseline method",
            "Interactive demo (Streamlit) to try the model/system on new inputs",
            "Clear plots and a results table mirroring the paper",
        ]
        modules = list(d.domain.modules)
        description = (
            f"Build a simplified, working version of the approach from \"{short_title}\". "
            f"Focus on {core}: implement the core idea, reproduce a key result on "
            f"{d.datasets[0] if d.datasets else 'a small public dataset'}, and package it in an easy-to-use demo. "
            "The goal is to understand the method deeply rather than match every number in the paper."
        )
        architecture = (
            "Data flows through the pipeline in stages: "
            + " -> ".join(m.split(":")[0] for m in modules)
            + ". Each stage is a separate Python module with a run() function, so you can test and replace stages "
            "independently. Configuration (paths, hyper-parameters) lives in one config file."
        )
        return {
            "project_idea": {
                "title": f"{core.title() if core.islower() else core}: A Student Implementation",
                "description": description,
                "features": features[:6],
                "tech_stack": tech,
                "deliverables": ["Working code repository with README", "Jupyter notebook with experiments",
                                 "Streamlit demo app", "Project report (8-12 pages) and presentation slides"],
            },
            "build_plan": {
                "project_name": name,
                "objective": f"Re-implement and evaluate the core idea of \"{short_title}\" on a student budget.",
                "architecture": architecture,
                "modules": modules,
                "setup_steps": [
                    "Create a Git repository and a Python virtual environment.",
                    f"Install the tech stack: {', '.join(tech)}.",
                    f"Download the data ({', '.join(d.datasets[:2]) or 'choose a public dataset'}) into data/raw/.",
                    "Write a tiny end-to-end version first (few samples), then scale up.",
                ],
                "evaluation_metrics": d.metrics[:4] or list(d.domain.metrics),
                "stretch_goals": [f.replace("Suggested: ", "").replace("Implied: ", "") for f in a.future_work[:2]]
                + [d.domain.deployment],
            },
        }

    def _task_roadmap(self, d: PaperDigest, task: TaskInput) -> dict[str, Any]:
        factors = self._scoring.compute_factors(d.full_text, d.page_count, task.analysis)
        deep_learning = 1.0 if re.search(r"neural network|deep learning|transformer", d.full_text, re.I) else 0.0
        difficulty = 1 + 9 * min(1.0, 0.3 * factors.math_density + 0.25 * factors.advanced_terms
                                 + 0.1 * factors.length + 0.3 * (d.domain.base_difficulty / 1.5)
                                 + 0.2 * deep_learning)
        difficulty_score = int(round(min(10, max(1, difficulty))))
        level = "Beginner" if difficulty_score <= 3 else "Intermediate" if difficulty_score <= 6 else "Advanced"
        weeks = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}[level]
        resume = int(round(min(10, max(1, 2 + 4.5 * factors.trending + 2 * factors.practicality))))
        modules = [m.split(":")[0] for m in task.analysis.build_plan.modules] or [m.split(":")[0] for m in d.domain.modules]

        def window(start: int, length: int) -> str:
            end = start + length - 1
            return f"Week {start}" if start == end else f"Week {start}-{end}"

        phases, week = [], 1
        plan = [
            ("Understand the paper", 1, ["Read abstract, introduction and conclusion; write a 1-page summary",
                                         "Study the methodology and draw the system diagram",
                                         "List the equations/algorithms you must implement"]),
            ("Environment & data", weeks, ["Set up the repository, virtual environment and project structure",
                                           f"Download and explore {d.datasets[0] if d.datasets else 'the dataset'}",
                                           "Write preprocessing code and a train/test split"]),
            ("Core implementation", weeks + 1, [f"Implement the {m} module" for m in modules[:3]]
             + ["Unit-test each module on a tiny input"]),
            ("Experiments & evaluation", weeks, [f"Measure {', '.join((d.metrics or list(d.domain.metrics))[:3])}",
                                                 "Compare with a baseline and run one ablation",
                                                 "Create plots and a results table"]),
            ("Demo, report & viva prep", 1, ["Build the Streamlit demo", "Write the project report and README",
                                             "Practise the viva and interview questions"]),
        ]
        for name, length, tasks in plan:
            phases.append({"phase": name, "duration": window(week, length), "tasks": tasks})
            week += length

        hours = {"Beginner": 30, "Intermediate": 60, "Advanced": 100}[level] + 2 * d.page_count
        return {
            "roadmap": phases,
            "assessment": {
                "difficulty_level": level,
                "difficulty_score": difficulty_score,
                "difficulty_rationale": f"{d.domain.name} paper; rated from its mathematical content, "
                                        "technical terminology and length.",
                "resume_impact_score": resume,
                "resume_impact_rationale": f"{d.domain.name} skills"
                + (" in a currently trending area" if factors.trending >= 0.4 else "")
                + (", with a reproducible dataset" if d.datasets else "") + ".",
                "estimated_hours": hours,
                "skills_gained": dedupe_preserve_order(list(d.domain.skills) + d.tools[:2])[:6],
            },
        }

    def _task_viva(self, d: PaperDigest, task: TaskInput) -> dict[str, Any]:
        a = task.analysis
        results = rank_sentences(d.section("results", "conclusion") or d.sentences, cues=RESULT_CUES,
                                 min_cue=0.6, top_n=2)
        method_first = re.sub(r"\*\*.*?\*\*|^\d+\.\s*", "", a.methodology.split("\n\n")[1] if "\n\n" in a.methodology
                              else a.methodology, flags=re.MULTILINE)
        qa = [
            ("What problem does this paper try to solve, and why does it matter?", a.problem_statement),
            ("What are the main contributions of the paper?", " ".join(a.key_contributions[:3])),
            ("Explain the proposed methodology in simple terms.", truncate(" ".join(method_first.split()), 600)),
            ("Which datasets were used for evaluation?", "; ".join(a.datasets[:3]) or NOT_FOUND),
            ("How did the authors evaluate their approach, and what were the key results?",
             " ".join(simplify_sentence(s.text) + f" {s.cite}" for s in results) or
             f"They report {', '.join(d.metrics[:3]) or 'quantitative metrics'} compared with baselines."),
            ("Which tools or frameworks would you use to implement it?", ", ".join(a.tools_frameworks[:5])
             or ", ".join(d.domain.tech_stack[:4])),
            ("What are the limitations of this work?", " ".join(a.limitations[:3])),
            ("How could this work be extended in the future?", " ".join(a.future_work[:3])),
        ]
        for term, meaning in d.glossary[:2]:
            qa.append((f"What is meant by '{term}' in this paper?", f"In simple terms, {term} is {meaning}."))
        return {"viva_questions": [{"question": q, "answer": ans or NOT_FOUND} for q, ans in qa]}

    def _task_interview(self, d: PaperDigest, task: TaskInput) -> dict[str, Any]:
        a = task.analysis
        b = a.build_plan
        tool = (a.project_idea.tech_stack[1:2] or ["Python"])[0]
        metrics = ", ".join(b.evaluation_metrics[:3]) or "standard metrics"
        qa = [
            ("Walk me through the architecture of your project.", b.architecture),
            (f"Why did you choose {tool} for this project?",
             f"{tool} is widely used in {d.domain.name.lower()}, has strong documentation and community support, "
             "and matches the tools mentioned in the paper, which made reproducing results easier."),
            ("How did you evaluate whether your implementation works?",
             f"I tracked {metrics}, compared against a simple baseline, and tried to reproduce a key result from "
             "the paper. I also unit-tested each module on tiny inputs before running full experiments."),
            ("What was the hardest part of implementing this paper, and how did you handle it?",
             "Papers often omit implementation details (hyper-parameters, preprocessing). I started with a tiny "
             "end-to-end version, read the method section carefully, and made each assumption explicit and "
             "configurable so I could test alternatives."),
            ("How would you deploy this in production or scale it up?", d.domain.deployment),
            ("What are the trade-offs or limitations of this approach?", " ".join(a.limitations[:2])),
            ("If you had two more months, what would you improve?", " ".join(a.future_work[:2])),
        ]
        return {"interview_questions": [{"question": q, "answer": ans or NOT_FOUND} for q, ans in qa]}


def _strip_task_prefix(text: str) -> str:
    return re.sub(r"^(search_document|search_query):\s*", "", text)


def _stream_words(text: str, delay: float = 0.004) -> Iterator[str]:
    """Yield text in small pieces so the chat UI shows a typing effect."""
    for piece in re.findall(r"\S+\s*|\n", text):
        yield piece
        if delay:
            time.sleep(delay)
