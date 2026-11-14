"""
Leakage detector for gold-eval split (D19).
Computes overlap metrics between gold annotations and evaluation crawl text
to prevent train/test leakage in ontology evaluation.

Usage:
    from prototype.eval.leakage_audit import LeakageAuditor
    auditor = LeakageAuditor()
    report = auditor.audit(gold_entities, eval_texts)
    assert report['jaccard'] < 0.2, "Leakage too high"
"""
import re
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Set, Tuple


@dataclass
class LeakageReport:
    """Result of leakage audit between gold annotations and eval crawl."""
    gold_size: int
    eval_size: int
    jaccard: float          # Jaccard similarity (0..1)
    rouge_l_p: float        # ROUGE-L precision
    rouge_l_r: float        # ROUGE-L recall
    rouge_l_f1: float       # ROUGE-L F1
    tfidf_cosine: float     # TF-IDF cosine similarity
    overlap_terms: List[str] = field(default_factory=list)

    def is_safe(
        self,
        jaccard_threshold: float = 0.2,
        rouge_threshold: float = 0.3,
    ) -> bool:
        """True if leakage is below safe thresholds."""
        return self.jaccard < jaccard_threshold and self.rouge_l_f1 < rouge_threshold


class LeakageAuditor:
    """Detect train/test leakage between gold annotations and eval crawl."""

    STOP_WORDS: frozenset = frozenset({
        'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'for', 'to', 'is', 'are',
    })

    def __init__(self, normalize: bool = True, drop_stopwords: bool = True) -> None:
        self.normalize = normalize
        self.drop_stopwords = drop_stopwords

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase, split on whitespace/punctuation, optionally drop stopwords."""
        if self.normalize:
            text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        if self.drop_stopwords:
            tokens = [t for t in tokens if t not in self.STOP_WORDS]
        return tokens

    def _jaccard(self, set_a: Set[str], set_b: Set[str]) -> float:
        """Jaccard similarity between two token sets."""
        if not set_a and not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union else 0.0

    def _rouge_l(
        self, ref_tokens: List[str], hyp_tokens: List[str]
    ) -> Tuple[float, float, float]:
        """ROUGE-L precision, recall, F1 via longest common subsequence (DP)."""
        m, n = len(ref_tokens), len(hyp_tokens)
        if m == 0 or n == 0:
            return 0.0, 0.0, 0.0

        # Standard LCS DP — O(m*n) time and space
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs = dp[m][n]
        p = lcs / n
        r = lcs / m
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    def _tfidf_cosine(
        self, doc_a_tokens: List[str], doc_b_tokens: List[str]
    ) -> float:
        """TF-IDF cosine similarity using both docs as the 2-document corpus."""
        vocab = set(doc_a_tokens) | set(doc_b_tokens)
        if not vocab:
            return 0.0

        tf_a = Counter(doc_a_tokens)
        tf_b = Counter(doc_b_tokens)

        # IDF over 2-document corpus
        idf: dict = {}
        for term in vocab:
            df = (1 if term in tf_a else 0) + (1 if term in tf_b else 0)
            idf[term] = math.log(2 / df) if df else 0.0

        vec_a = {t: tf_a[t] * idf[t] for t in vocab}
        vec_b = {t: tf_b[t] * idf[t] for t in vocab}

        dot = sum(vec_a[t] * vec_b[t] for t in vocab)
        norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def audit(
        self,
        gold_entities: List[str],
        eval_texts: List[str],
    ) -> LeakageReport:
        """Run Jaccard, ROUGE-L, and TF-IDF cosine overlap checks.

        Args:
            gold_entities: List of gold-annotated entity strings.
            eval_texts: List of evaluation crawl text strings.

        Returns:
            LeakageReport with all three overlap scores and top overlap terms.
        """
        gold_text = ' '.join(gold_entities)
        eval_text = ' '.join(eval_texts)

        gold_tokens = self._tokenize(gold_text)
        eval_tokens = self._tokenize(eval_text)

        gold_set = set(gold_tokens)
        eval_set = set(eval_tokens)

        jaccard = self._jaccard(gold_set, eval_set)
        p, r, f1 = self._rouge_l(gold_tokens, eval_tokens)
        cosine = self._tfidf_cosine(gold_tokens, eval_tokens)

        overlap = sorted(gold_set & eval_set)

        return LeakageReport(
            gold_size=len(gold_set),
            eval_size=len(eval_set),
            jaccard=jaccard,
            rouge_l_p=p,
            rouge_l_r=r,
            rouge_l_f1=f1,
            tfidf_cosine=cosine,
            overlap_terms=overlap[:50],  # cap to keep report readable
        )
