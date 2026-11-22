"""
metrics.py — Targeted transfer metrics for the VLM-transfer evaluation.
======================================================================
Scope (maintainer decision 2026-11-21): **targeted metrics only**.  The
CLIP-BFA attack is a *targeted* backdoor — the interesting question is
whether generated answers/captions are pushed toward the attacker's target
semantics, not whether the VLM's task score merely degrades.

Two metrics, both text-side (BLIP.md §4: generative readouts break the
cosine-argmax ASR definition, so the transfer metric lives on the text):

  * ``target_mention_rate`` — fraction of generations that mention the
    target concept.  Directly interpretable as "the backdoor fired".
  * ``bleu4`` — BLEU-4 between the generation and the target text
    description, i.e. X-Transfer's targeted-ASR proxy (Appendix C.7: "we
    report the BLEU-4 score ... to measure the similarity between the
    generated caption and the adversary's target text description as the
    targeted ASR").

X-Transfer's *non-targeted* degradation ASR = (s_clean − s_adv)/s_clean is
deliberately NOT computed here (maintainer choice).  CIDEr / VQA-accuracy
are therefore not required and ``pycocoevalcap`` is not a dependency.

Self-contained: no external metric packages.
"""

from __future__ import annotations

import math
import re
from collections import Counter

__all__ = [
    "bleu4",
    "corpus_bleu4",
    "target_mention_rate",
    "target_terms_from_desc",
    "tokenize",
    "summarise_generations",
]

# ─────────────────────────────────────────────────────────────────────
#  Tokenisation (lower-cased word-level, punctuation stripped)
# ─────────────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Very small stopword list — only used to derive mention terms from a free
# text target description.  Never used in the BLEU computation.
_STOPWORDS = frozenset(
    "a an the of in on at to for and or with without is are was were be been "
    "being this that these those it its as by from into over under about "
    "image photo picture photograph showing shows show depicts depiction "
    "there here he she they his her their my your our".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens with punctuation removed."""
    return _TOKEN_RE.findall(text.lower())


# ─────────────────────────────────────────────────────────────────────
#  BLEU-4  (Papineni et al., 2002) — sentence & corpus level
# ─────────────────────────────────────────────────────────────────────
def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _modified_precision(hyp: list[str], refs: list[list[str]], n: int):
    """Clipped n-gram precision counts: (clipped, total)."""
    hyp_ngrams = _ngrams(hyp, n)
    if not hyp_ngrams:
        return 0, 0
    # max reference count per n-gram across all references
    max_ref: Counter = Counter()
    for ref in refs:
        for gram, cnt in _ngrams(ref, n).items():
            if cnt > max_ref[gram]:
                max_ref[gram] = cnt
    clipped = sum(min(cnt, max_ref[gram]) for gram, cnt in hyp_ngrams.items())
    return clipped, sum(hyp_ngrams.values())


def bleu4(hypothesis: str, references: list[str], smooth: bool = True) -> float:
    """Sentence-level BLEU-4 in [0, 1] against one or more references.

    ``smooth`` applies the standard add-epsilon fix so a single missing
    4-gram does not zero the whole score (generations are short, so this
    matters a lot here).
    """
    hyp = tokenize(hypothesis)
    refs = [tokenize(r) for r in references] or [[]]
    if not hyp:
        return 0.0

    log_precisions = []
    for n in (1, 2, 3, 4):
        clipped, total = _modified_precision(hyp, refs, n)
        if total == 0:
            # hyp shorter than n — no n-grams at all
            return 0.0
        if clipped == 0:
            if n == 1 or not smooth:
                # zero unigram overlap = the texts share no words at all;
                # smoothing must not turn a disjoint pair into a small
                # non-zero score.
                return 0.0
            clipped, total = 1.0, total + 1.0  # Lin & Och epsilon smoothing
        log_precisions.append(math.log(clipped / total))

    # brevity penalty
    ref_lens = [len(r) for r in refs]
    closest = min(ref_lens, key=lambda rl: (abs(rl - len(hyp)), rl))
    if len(hyp) > closest:
        bp = 1.0
    else:
        bp = math.exp(1.0 - closest / len(hyp)) if len(hyp) else 0.0

    return float(bp * math.exp(sum(log_precisions) / 4.0))


def corpus_bleu4(hypotheses: list[str],
                 references: list[list[str]],
                 smooth: bool = True) -> float:
    """Corpus-level BLEU-4 in [0, 1].  ``references[i]`` are the reference
    strings for ``hypotheses[i]`` (any number ≥ 1)."""
    if not hypotheses:
        return 0.0
    assert len(hypotheses) == len(references), \
        f"{len(hypotheses)} hypotheses vs {len(references)} reference lists"

    hyp_toks = [tokenize(h) for h in hypotheses]
    ref_toks = [[tokenize(r) for r in (refs or [""])] for refs in references]

    log_precisions = []
    for n in (1, 2, 3, 4):
        clipped_total = 0
        total_total = 0
        for hyp, refs in zip(hyp_toks, ref_toks):
            c, t = _modified_precision(hyp, refs, n)
            clipped_total += c
            total_total += t
        if total_total == 0:
            return 0.0
        if clipped_total == 0:
            if n == 1 or not smooth:
                # no unigram overlap anywhere in the corpus
                return 0.0
            clipped_total, total_total = 1.0, total_total + 1.0
        log_precisions.append(math.log(clipped_total / total_total))

    hyp_len = sum(len(h) for h in hyp_toks)
    ref_len = 0
    for hyp, refs in zip(hyp_toks, ref_toks):
        closest = min((len(r) for r in refs),
                      key=lambda rl: (abs(rl - len(hyp)), rl))
        ref_len += closest
    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / hyp_len)
    return float(bp * math.exp(sum(log_precisions) / 4.0))


# ─────────────────────────────────────────────────────────────────────
#  Target-mention rate
# ─────────────────────────────────────────────────────────────────────
def target_terms_from_desc(desc: str) -> list[str]:
    """Derive mention terms from a free-text target description.

    "a photo of a lion" -> ["lion"].  Multi-word concepts keep their bigram
    ("a photo of a tabby cat" -> ["tabby cat", "tabby", "cat"]).
    """
    toks = [t for t in tokenize(desc) if t not in _STOPWORDS]
    if not toks:
        toks = tokenize(desc)
    terms = list(toks)
    terms += [f"{a} {b}" for a, b in zip(toks, toks[1:])]
    # longest first so a bigram match is found before its unigrams
    return sorted(set(terms), key=len, reverse=True) or [""]


def target_mention_rate(texts: list[str], terms: list[str]) -> float:
    """Fraction of generations mentioning at least one target term.

    Matching is case-insensitive whole-word; a multi-word term matches as a
    phrase.  Empty input -> 0.0.
    """
    if not texts:
        return 0.0
    terms = [t.strip().lower() for t in terms if t and t.strip()]
    if not terms:
        raise ValueError("target_mention_rate needs at least one target term")

    # word-boundary regex per term (phrase terms keep their spaces)
    pats = [re.compile(r"\b" + re.escape(t) + r"\b") for t in terms]
    hits = sum(1 for text in texts
               if any(p.search(text.lower()) for p in pats))
    return hits / len(texts)


# ─────────────────────────────────────────────────────────────────────
#  Reporting helper
# ─────────────────────────────────────────────────────────────────────
def summarize_generations(texts: list[str],
                          terms: list[str],
                          target_desc: str) -> dict:
    """Metrics for one condition (e.g. attacked tower × triggered input)."""
    refs = [target_desc] * len(texts)
    return {
        "n": len(texts),
        "target_mention_rate": round(target_mention_rate(texts, terms), 4),
        "target_mentions": sum(
            1 for t in texts
            if any(re.search(r"\b" + re.escape(term.strip().lower()) + r"\b",
                             t.lower())
                   for term in terms if term.strip())),
        "bleu4_vs_target_desc": round(corpus_bleu4(texts, refs), 4),
        "mean_len_chars": round(
            (sum(len(t) for t in texts) / len(texts)) if texts else 0.0, 2),
    }
