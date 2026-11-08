"""Dependency-free NLP building blocks for the built-in (no-LLM) engine.

* sentence splitting that survives academic abbreviations ("et al.", "Fig. 3")
* tokenisation with stop-words and light stemming
* stable hashed embeddings (unigrams + bigrams, signed hashing trick)
* cue-phrase + centroid sentence ranking (extractive summarisation)
* RAKE-style key-phrase extraction and explicit "Index Terms" detection
* a student glossary that decodes common engineering jargon
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had has
    have having he her here hers him his how i if in into is it its itself just me more most my no nor not
    of off on once only or other our ours out over own same she should so some such than that the their
    them then there these they this those through to too under until up very was we were what when where
    which while who whom why will with would you your et al fig figure table eq equation section paper
    using used use based thus therefore hence may might must one two three via etc respectively without
    within among across per well many much often several since given various e g i ie eg vs""".split()
)

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*")
_ABBREVIATIONS = re.compile(r"\b(et al|e\.g|i\.e|Fig|Figs|Eq|Eqs|Sec|vs|cf|approx|resp|No|Ref|Refs|Dr|Prof)\.", re.IGNORECASE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
_PLACEHOLDER = "⁣"  # invisible separator used to protect abbreviation dots


# ------------------------------------------------------------------ tokenisation
def stem(word: str) -> str:
    """Very light stemmer: enough to match 'networks'/'network', 'trained'/'training'."""
    for suffix in ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in (m.group().lower() for m in _WORD.finditer(text)) if len(w) > 2 and w not in STOPWORDS]


def split_into_sentences(text: str) -> list[str]:
    protected = _ABBREVIATIONS.sub(lambda m: m.group(1) + _PLACEHOLDER, text)
    parts = _SENTENCE_BOUNDARY.split(protected)
    return [p.replace(_PLACEHOLDER, ".").strip() for p in parts if p.strip()]


def is_prose(sentence: str) -> bool:
    """Filter out equations, table rows, captions and broken fragments."""
    words = sentence.split()
    if not (6 <= len(words) <= 80) or len(sentence) < 40:
        return False
    letters = sum(c.isalpha() for c in sentence)
    if letters / max(1, len(sentence)) < 0.65:
        return False
    if re.match(r"^(table|fig(ure)?|algorithm)\s*\d", sentence, re.IGNORECASE):
        return False
    return sentence[0].isupper() or sentence[0] in "([\"'"


# -------------------------------------------------------------------- embeddings
def _bucket(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


def hashed_embedding(text: str, dim: int = 512) -> list[float]:
    """Deterministic bag-of-words embedding (hashing trick, unigrams + bigrams, log-TF).

    Stable across processes (blake2b, not Python's randomised ``hash``), so vectors
    persisted in ChromaDB stay valid after a restart.
    """
    tokens = tokenize(text)
    features = Counter(tokens)
    features.update(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))
    vector = [0.0] * dim
    for feature, count in features.items():
        index, sign = _bucket(feature, dim)
        weight = 1.0 + math.log(count)
        vector[index] += sign * (weight * (0.6 if "_" in feature else 1.0))
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        vector[0] = 1.0  # avoid zero vectors (cosine undefined)
        return vector
    return [v / norm for v in vector]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ------------------------------------------------------------------- ranking
@dataclass(frozen=True)
class Sentence:
    text: str
    section: str
    page: int
    position: int  # global order in the paper

    @property
    def cite(self) -> str:
        return f"(p. {self.page})" if self.page else ""


def cue_score(text: str, cues: dict[str, float]) -> float:
    lowered = text.lower()
    return sum(weight for cue, weight in cues.items() if cue in lowered)


def rank_sentences(
    sentences: Sequence[Sentence],
    *,
    cues: dict[str, float] | None = None,
    reference: str | None = None,
    top_n: int = 5,
    min_cue: float = 0.0,
    keep_order: bool = True,
    max_chars: int = 320,
) -> list[Sentence]:
    """Rank by (centroid similarity + cue phrases), return the best ``top_n``."""
    pool = [s for s in sentences if len(s.text) <= max_chars]
    if cues and min_cue > 0:
        pool = [s for s in pool if cue_score(s.text, cues) >= min_cue]
    if not pool:
        return []
    centroid = hashed_embedding(reference or " ".join(s.text for s in pool))
    scored = []
    for sentence in pool:
        score = cosine(hashed_embedding(sentence.text), centroid)
        if cues:
            score += 0.35 * cue_score(sentence.text, cues)
        scored.append((score, sentence))
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = _drop_near_duplicates([s for _, s in scored], top_n)
    return sorted(chosen, key=lambda s: s.position) if keep_order else chosen


def _drop_near_duplicates(candidates: Iterable[Sentence], limit: int, threshold: float = 0.8) -> list[Sentence]:
    chosen: list[Sentence] = []
    vectors: list[list[float]] = []
    for sentence in candidates:
        vector = hashed_embedding(sentence.text)
        if all(cosine(vector, other) < threshold for other in vectors):
            chosen.append(sentence)
            vectors.append(vector)
        if len(chosen) >= limit:
            break
    return chosen


# ------------------------------------------------------------- key phrases
_EXPLICIT_KEYWORDS = re.compile(
    r"(?:index terms|keywords|key words)\s*[\-—–:.]?\s*(.{10,400}?)(?:\n|(?<=[a-z])\.\s|$)", re.IGNORECASE
)
_PHRASE_SPLIT = re.compile(r"[^A-Za-z0-9\- ]+")
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,6}s?\b")
_BAD_ACRONYMS = {"IEEE", "ACM", "PDF", "URL", "II", "III", "IV", "VI", "VII", "VIII", "IX", "XI", "XII", "AND", "THE", "FOR"}


def explicit_keywords(text: str) -> list[str]:
    match = _EXPLICIT_KEYWORDS.search(text[:8000])
    if not match:
        return []
    items = [re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9)]+$", "", k) for k in re.split(r"[;,•·]", match.group(1))]
    return [k for k in items if 2 < len(k) < 60][:10]


def keyphrases(text: str, top_n: int = 8) -> list[str]:
    """RAKE-style phrases (1-4 words) ranked by word degree, plus frequent acronyms."""
    surface: dict[str, Counter[str]] = {}
    phrase_counts: Counter[str] = Counter()
    for fragment in _PHRASE_SPLIT.split(text):
        current: list[str] = []
        for word in fragment.split() + ["."]:
            lowered = word.lower()
            if word == "." or lowered in STOPWORDS or len(word) < 3 or word[0].isdigit():
                if 1 <= len(current) <= 4:
                    key = " ".join(stem(w.lower()) for w in current)
                    phrase_counts[key] += 1
                    surface.setdefault(key, Counter())[" ".join(current)] += 1
                current = []
            else:
                current.append(word)

    word_freq: Counter[str] = Counter()
    word_degree: Counter[str] = Counter()
    for phrase, count in phrase_counts.items():
        words = phrase.split()
        for word in words:
            word_freq[word] += count
            word_degree[word] += count * len(words)

    def score(phrase: str) -> float:
        words = phrase.split()
        base = sum(word_degree[w] / word_freq[w] for w in words)
        return base * math.log(1 + phrase_counts[phrase]) * (1.3 if len(words) > 1 else 1.0)

    ranked = sorted((p for p, c in phrase_counts.items() if c >= 2 and len(p) > 3), key=score, reverse=True)
    results: list[str] = []
    for key in ranked:
        best = surface[key].most_common(1)[0][0]
        if not any(best.lower() in r.lower() or r.lower() in best.lower() for r in results):
            results.append(best if best.isupper() else best.lower())
        if len(results) >= top_n:
            break

    acronyms = Counter(a.rstrip("s") for a in _ACRONYM.findall(text) if a.rstrip("s") not in _BAD_ACRONYMS)
    for acronym, count in acronyms.most_common(4):
        if count >= 3 and acronym.lower() not in " ".join(results).lower():
            results.append(acronym)
    return results[: top_n + 2]


# ------------------------------------------------------------------ glossary
GLOSSARY: dict[str, str] = {
    "neural network": "a model made of layers of simple math units that learn patterns from examples",
    "convolutional": "a neural network layer that slides small filters over an image to detect edges, shapes and textures",
    "cnn": "Convolutional Neural Network - a network specialised for images",
    "recurrent neural network": "a network that processes sequences one step at a time, keeping a memory of earlier steps",
    "lstm": "Long Short-Term Memory - a recurrent network that can remember information over long sequences",
    "transformer": "a model that processes all parts of the input at once and uses attention to decide what matters",
    "attention": "a mechanism that lets the model focus on the most relevant parts of the input",
    "embedding": "a list of numbers that represents a word, image or item so that similar things get similar numbers",
    "encoder": "the part of a model that compresses the input into a compact internal representation",
    "decoder": "the part of a model that turns the internal representation back into an output",
    "gradient descent": "an optimisation method that repeatedly nudges model parameters to reduce the error",
    "backpropagation": "the algorithm that computes how much each weight contributed to the error so it can be corrected",
    "loss function": "a formula that measures how wrong the model's predictions are",
    "overfitting": "when a model memorises the training data and performs poorly on new data",
    "regularization": "techniques that discourage overly complex models to prevent overfitting",
    "dropout": "randomly switching off neurons during training so the network does not rely on any single one",
    "hyperparameter": "a setting chosen before training (e.g. learning rate) rather than learned from data",
    "epoch": "one full pass of the training algorithm over the whole dataset",
    "fine-tuning": "taking a pre-trained model and training it a little more on your specific task",
    "transfer learning": "reusing knowledge a model learned on one task to help with another task",
    "pre-trained": "already trained on a large general dataset before being adapted to a specific task",
    "baseline": "a simpler existing method used as a reference point for comparison",
    "benchmark": "a standard dataset or test used to compare methods fairly",
    "state-of-the-art": "the best published result so far",
    "ablation": "an experiment that removes one component at a time to see how much it matters",
    "end-to-end": "trained or run as one complete system from raw input to final output",
    "precision": "of all items the model flagged as positive, the fraction that were actually positive",
    "recall": "of all truly positive items, the fraction the model managed to find",
    "f1": "the harmonic mean of precision and recall - one number that balances both",
    "accuracy": "the percentage of predictions that were correct",
    "cross-validation": "repeatedly splitting data into train/test parts to get a more reliable performance estimate",
    "classification": "predicting which category an input belongs to",
    "regression": "predicting a continuous number (e.g. a price or temperature)",
    "clustering": "grouping similar items together without using labels",
    "segmentation": "labelling every pixel (or region) of an image with a class",
    "object detection": "finding objects in an image and drawing a box around each one",
    "feature extraction": "turning raw data into informative numbers a model can learn from",
    "support vector machine": "a classic classifier that finds the boundary with the widest margin between classes",
    "random forest": "an ensemble of many decision trees that vote on the answer",
    "reinforcement learning": "learning by trial and error, guided by rewards and penalties",
    "generative adversarial": "two networks competing: one generates fake data, the other tries to spot fakes",
    "federated learning": "training a shared model across many devices without collecting their raw data centrally",
    "internet of things": "everyday devices connected to the internet that sense and share data",
    "iot": "Internet of Things - connected sensors and devices",
    "edge computing": "processing data near where it is produced (on the device) instead of in a distant cloud",
    "latency": "the delay between a request and its response",
    "throughput": "how much data or how many tasks a system handles per unit of time",
    "blockchain": "a tamper-resistant shared ledger where records are linked together cryptographically",
    "encryption": "scrambling data so only someone with the right key can read it",
    "intrusion detection": "monitoring a network or system to spot malicious activity",
    "software-defined networking": "controlling network switches from central software instead of configuring each device",
    "sdn": "Software-Defined Networking - centrally programmable networks",
    "ofdm": "a way of sending data over many closely spaced radio frequencies at once (used in Wi-Fi and 4G/5G)",
    "mimo": "using multiple antennas at sender and receiver to send more data reliably",
    "kalman filter": "an algorithm that estimates a changing quantity from noisy measurements",
    "pid controller": "a feedback controller that corrects errors using proportional, integral and derivative terms",
    "optimization": "finding the best solution according to some objective",
    "heuristic": "a practical rule of thumb that finds a good (not always perfect) solution quickly",
    "inference": "using a trained model to make predictions on new data",
    "bleu": "a score that compares machine-generated text with human reference translations",
    "retrieval-augmented": "looking up relevant documents first and giving them to a language model to ground its answer",
    "large language model": "a very large transformer trained on huge amounts of text to understand and generate language",
    "natural language processing": "teaching computers to understand and generate human language",
    "residual": "a shortcut connection that adds a layer's input to its output, making very deep networks trainable",
    "batch normalization": "re-scaling layer outputs during training to make learning faster and more stable",
}


def glossary_terms(text: str, limit: int = 4) -> list[tuple[str, str]]:
    """Glossary entries that occur in ``text``, most frequent first."""
    lowered = text.lower()
    hits = []
    for term, explanation in GLOSSARY.items():
        count = len(re.findall(rf"(?<![a-z]){re.escape(term)}", lowered))
        if count:
            hits.append((count, term, explanation))
    hits.sort(key=lambda h: (-h[0], h[1]))
    chosen: list[tuple[str, str]] = []
    for _, term, explanation in hits:
        if not any(term in other or other in term for other, _ in chosen):
            chosen.append((term, explanation))
        if len(chosen) >= limit:
            break
    return chosen


def simplify_sentence(sentence: str, max_chars: int = 260) -> str:
    """Strip academic filler at the start of a sentence and trim its length."""
    text = re.sub(
        r"^(in this (paper|work|article|study),?\s*|specifically,?\s*|moreover,?\s*|furthermore,?\s*|"
        r"in addition,?\s*|additionally,?\s*|to this end,?\s*|in particular,?\s*|finally,?\s*)",
        "", sentence.strip(), flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*\[[\d,\s\-–]+\]", "", text)  # citation markers like [12, 14]
    text = text[0].upper() + text[1:] if text else text
    if len(text) > max_chars:
        cut = text[:max_chars].rsplit(" ", 1)[0]
        text = cut.rstrip(",;:") + "..."
    return text
