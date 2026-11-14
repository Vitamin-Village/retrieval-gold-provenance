# -*- coding: utf-8 -*-
"""Auxiliary evidence-sentence audit for GraphRAG-Bench (medical).

The census assigns the GraphRAG-Bench subgraph axis to bin 3 (C-hat = 0.75),
but the released fields audited here are sentence-level textual realizations,
not gold subgraphs. This script therefore does not estimate per-unit
C_subgraph. It measures whether each evidence sentence surface-realizes a
source KG relation and how often that realization is a near-verbatim copy.
The result is a surface-form diagnostic used alongside the independently
audited ontology-to-subgraph construction record.

Two per-unit quantities:

  (1) Relation-realization fraction. An evidence sentence is `derived=1` iff a MAJORITY
      of the content (non-stopword) tokens of some KG relation string appear in
      the normalized evidence sentence. Deterministic and conservative: only
      surface token overlap is credited (no aliases, paraphrase expansion, or
      semantic matching), so this is a LOWER BOUND on the true derivation rate.

  (2) Verbatim-copy fraction. Among derived units, the fraction whose evidence
      sentence is nearly identical to the matched relation string, defined as
      token Jaccard >= 0.9 OR one normalized string being a substring of the
      other. This describes sentence surfaces only and is not used to assign a
      provenance-circularity bin to the unreleased gold subgraphs.

Normalization: lowercase, strip punctuation to spaces, collapse whitespace.

Full recomputation writes ``census/perunit_graphragbench.recomputed.json`` and
``census/perunit_graphragbench_summary.recomputed.md``.  In the submission archive the
upstream GraphRAG-Bench source file is not redistributed; running this script
there instead verifies the shipped per-question records and aggregates.  Pass
``--source`` to recompute from an independently obtained source file.
"""

import argparse
import io
import json
import os
import re

CENSUS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(CENSUS_DIR, ".."))
SRC = os.path.join(BASE, "external", "graphragbench", "raw", "medical_questions.json")
OUT_JSON = os.path.join(CENSUS_DIR, "perunit_graphragbench.recomputed.json")
OUT_MD = os.path.join(CENSUS_DIR, "perunit_graphragbench_summary.recomputed.md")
ARCHIVED_JSON = os.path.join(CENSUS_DIR, "perunit_graphragbench.json")

# derivation threshold: majority of relation content tokens present in sentence
DERIV_FRAC = 0.5
# verbatim-copy threshold: token Jaccard >= this (or substring) among derived
VERBATIM_JACCARD = 0.9

_PUNCT = re.compile(r"[^a-z0-9\s]+")
_WS = re.compile(r"\s+")

# small, fixed English stopword set (deterministic; conservative content-token set)
STOP = frozenset("""
a an the of for to in on at by with from into over under and or but nor so as
is are was were be been being am do does did have has had having will would
shall should can could may might must this that these those it its their his
her them they we you i he she who whom which what when where why how than then
not no yes such each any all both more most some few many much very also only
just about between among within without across per via vs versus include
includes including increase increases increased risk factor factors
""".split())


def norm(s):
    if s is None:
        return ""
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def tokens(s):
    return [t for t in norm(s).split() if t]


def content_tokens(s):
    return [t for t in tokens(s) if t not in STOP and len(t) > 1]


def as_relation_list(er):
    if isinstance(er, list):
        return [r for r in er if isinstance(r, str) and r.strip()]
    if isinstance(er, str) and er.strip():
        return [er]
    return []


def as_evidence_list(ev):
    if isinstance(ev, list):
        return [e for e in ev if isinstance(e, str) and e.strip()]
    if isinstance(ev, str) and ev.strip():
        return [ev]
    return []


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def judge_sentence(sent, relations):
    """Return (derived, verbatim, best_relation_index).

    derived=1 if some relation has a MAJORITY of its content tokens present in
    the sentence. Among the relations that qualify, pick the one with highest
    coverage; verbatim is judged against that best relation.
    """
    sent_norm = norm(sent)
    sent_tok = tokens(sent)
    sent_set = set(sent_tok)

    best = None  # (coverage, ridx)
    for ridx, rel in enumerate(relations):
        rct = content_tokens(rel)
        if not rct:
            continue
        present = sum(1 for t in set(rct) if t in sent_set)
        coverage = present / float(len(set(rct)))
        if coverage >= DERIV_FRAC:
            if best is None or coverage > best[0]:
                best = (coverage, ridx)

    if best is None:
        return 0, 0, -1

    ridx = best[1]
    rel = relations[ridx]
    rel_norm = norm(rel)
    # verbatim: substring either way, or high full-token Jaccard
    sub = (rel_norm and (rel_norm in sent_norm or sent_norm in rel_norm))
    jac = jaccard(sent_tok, tokens(rel))
    verbatim = 1 if (sub or jac >= VERBATIM_JACCARD) else 0
    return 1, verbatim, ridx


def aggregate(items):
    q_deriv_fracs = []          # per-question derivation fraction
    n_sent = 0
    n_derived = 0
    n_verbatim = 0              # verbatim among derived
    n_no_rel = 0
    n_no_ev = 0
    by_type = {}                # type -> [n_sent, n_derived, n_verbatim, n_q, sum_q_deriv]
    per_q = []

    for it in items:
        relations = as_relation_list(it.get("evidence_relations"))
        evid = as_evidence_list(it.get("evidence"))
        ty = it.get("question_type") or "unknown"
        if not relations:
            n_no_rel += 1
        if not evid:
            n_no_ev += 1

        k = len(evid)
        d = 0
        vb = 0
        for s in evid:
            der, ver, _ = judge_sentence(s, relations)
            d += der
            vb += ver
        n_sent += k
        n_derived += d
        n_verbatim += vb

        qfrac = (d / k) if k > 0 else None
        if qfrac is not None:
            q_deriv_fracs.append(qfrac)

        ta = by_type.setdefault(ty, [0, 0, 0, 0, 0.0])
        ta[0] += k
        ta[1] += d
        ta[2] += vb
        ta[3] += 1
        if qfrac is not None:
            ta[4] += qfrac

        per_q.append({"id": it.get("id"), "type": ty, "n_evidence": k,
                      "n_derived": d, "n_verbatim": vb})

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    by_type_out = {}
    for t, v in sorted(by_type.items()):
        nse, nd, nv, nq, sumq = v
        by_type_out[t] = {
            "n_questions": nq,
            "n_evidence_sentences": nse,
            "derivation_question_mean": round(sumq / nq, 4) if nq else 0.0,
            "derivation_sentence_pooled": round(nd / nse, 4) if nse else 0.0,
            "n_derived": nd,
            "verbatim_copy_fraction_among_derived": round(nv / nd, 4) if nd else 0.0,
            "n_verbatim": nv,
        }

    return {
        "n_questions": len(items),
        "n_questions_without_relations": n_no_rel,
        "n_questions_without_evidence": n_no_ev,
        "n_evidence_sentences": n_sent,
        "derivation": {
            "question_mean": round(mean(q_deriv_fracs), 4),
            "sentence_pooled": round(n_derived / n_sent, 4) if n_sent else 0.0,
            "n_derived": n_derived,
        },
        "verbatim_copy_among_derived": {
            "fraction": round(n_verbatim / n_derived, 4) if n_derived else 0.0,
            "n_verbatim": n_verbatim,
            "n_derived": n_derived,
        },
        "by_question_type": by_type_out,
        "_per_question": per_q,
    }


def build_md(agg):
    dqm = agg["derivation"]["question_mean"]
    dsp = agg["derivation"]["sentence_pooled"]
    vb = agg["verbatim_copy_among_derived"]["fraction"]
    L = []
    a = L.append
    a("# GraphRAG-Bench Evidence-Sentence Surface Diagnostic")
    a("")
    a("Generated by `census/perunit_graphragbench.py` (deterministic, offline).")
    a("")
    a("## Setup")
    a("")
    a("The census assigns the GraphRAG-Bench subgraph axis to bin 3 (C-hat = 0.75),")
    a("but the released units analyzed here are evidence sentences, not gold subgraphs.")
    a("This is therefore an auxiliary surface diagnostic, not a per-unit estimate of")
    a("C_subgraph. GraphRAG-Bench (medical split, %d questions) publishes,"
      % agg["n_questions"])
    a("per question, the KG relation(s) it was generated from (`evidence_relations`) and")
    a("sentence-level supporting evidence (`evidence`). We judge whether each evidence")
    a("sentence is a surface realization of a KG relation (derivation), and whether that")
    a("realization is a near-verbatim copy of the relation string.")
    a("")
    a("## Judgment rules (deterministic, conservative)")
    a("")
    a("Normalization: lowercase, punctuation -> space, whitespace collapsed.")
    a("")
    a("1. **Relation realization.** An evidence sentence is `derived = 1` iff a MAJORITY")
    a("   (>= %.2f) of the content (non-stopword, length > 1) tokens of some" % DERIV_FRAC)
    a("   `evidence_relations` string appear in the normalized sentence. Only surface")
    a("   token overlap is credited (no aliases, paraphrase, or semantic matching), so")
    a("   the derivation fraction is a conservative LOWER BOUND.")
    a("2. **Verbatim-copy.** Among matched units, a unit is verbatim iff")
    a("   the normalized sentence and its matched relation string satisfy token")
    a("   Jaccard >= %.2f, or one is a substring of the other. This describes how" % VERBATIM_JACCARD)
    a("   source relations are realized in text; it does not classify a subgraph unit.")
    a("")
    a("## Results")
    a("")
    a("Overall (%d questions, %d evidence sentences):" % (
        agg["n_questions"], agg["n_evidence_sentences"]))
    a("")
    a("| Quantity | Question-mean | Sentence-pooled |")
    a("|---|---|---|")
    a("| Relation-realization fraction | %.4f | %.4f |" % (
        agg["derivation"]["question_mean"], agg["derivation"]["sentence_pooled"]))
    a("")
    a("Verbatim-copy fraction among derived units: %.4f (%d / %d derived)." % (
        vb, agg["verbatim_copy_among_derived"]["n_verbatim"],
        agg["verbatim_copy_among_derived"]["n_derived"]))
    a("")
    a("Data integrity: %d questions lacked a relations list, %d lacked evidence." % (
        agg["n_questions_without_relations"], agg["n_questions_without_evidence"]))
    a("")
    a("### By question type")
    a("")
    a("| Type | Questions | Sentences | Derivation (q-mean) | Derivation (pooled) | Verbatim-copy (among derived) |")
    a("|---|---|---|---|---|---|")
    for t, tv in agg["by_question_type"].items():
        a("| %s | %d | %d | %.4f | %.4f | %.4f |" % (
            t, tv["n_questions"], tv["n_evidence_sentences"],
            tv["derivation_question_mean"], tv["derivation_sentence_pooled"],
            tv["verbatim_copy_fraction_among_derived"]))
    a("")
    a("## Interpretation and scope")
    a("")
    # per-type spread, computed (not hardcoded)
    bt = agg["by_question_type"]
    kg_native = [t for t in bt if t != "Creative Generation"]
    nat_lo = min(bt[t]["derivation_question_mean"] for t in kg_native) if kg_native else 0.0
    nat_hi = max(bt[t]["derivation_question_mean"] for t in kg_native) if kg_native else 0.0
    vb_max = max(tv["verbatim_copy_fraction_among_derived"] for tv in bt.values())
    cg = bt.get("Creative Generation")
    a("The measured relation-realization fraction is %.3f (question-mean) / %.3f (sentence-pooled)," % (dqm, dsp))
    a("a conservative lower bound because only surface token overlap is credited (a sentence")
    a("that expresses a relation through synonyms, paraphrase, or recontextualization scores 0).")
    a("Across the three KG-native question types (%s) the per-question derivation runs %.2f-%.2f;"
      % (", ".join(kg_native), nat_lo, nat_hi))
    if cg is not None:
        a("Creative Generation is the outlier at %.2f, because its evidence sentences rewrite the"
          % cg["derivation_question_mean"])
        a("source relations into role-play / discharge-summary scenarios, diverging most in surface")
        a("form from the KG triples. This rewording is what pulls the pooled mean below the")
        a("question mean (Creative Generation contributes the most sentences).")
    a("")
    a("The verbatim-copy fraction is low everywhere")
    a("(overall %.3f; per-type max %.3f), so the realized evidence is reworded rather than copied" % (vb, vb_max))
    a("verbatim from the KG.")
    a("")
    a("These figures describe sentence surfaces only and do not establish, reject, or rank")
    a("a provenance bin for the gold subgraphs. The subgraph bin remains a methodology-level")
    a("judgment grounded in the documented ontology-to-subgraph construction and human")
    a("confirmation step, with cross-model and human dissent reported separately.")
    a("")
    return "\n".join(L) + "\n"


def verify_frozen(path):
    """Validate archived per-question records against every headline total."""
    with io.open(path, encoding="utf-8") as f:
        payload = json.load(f)
    overall = payload["overall"]
    per_question = payload.get("_per_question", [])
    if len(per_question) != overall["n_questions"]:
        raise ValueError("per-question record count disagrees with the aggregate")

    n_sent = sum(row["n_evidence"] for row in per_question)
    n_derived = sum(row["n_derived"] for row in per_question)
    n_verbatim = sum(row["n_verbatim"] for row in per_question)
    if n_sent != overall["n_evidence_sentences"]:
        raise ValueError("evidence-sentence total disagrees with the aggregate")
    if n_derived != overall["derivation"]["n_derived"]:
        raise ValueError("derived-unit total disagrees with the aggregate")
    if n_verbatim != overall["verbatim_copy_among_derived"]["n_verbatim"]:
        raise ValueError("verbatim-unit total disagrees with the aggregate")

    q_fracs = [row["n_derived"] / row["n_evidence"]
               for row in per_question if row["n_evidence"]]
    q_mean = round(sum(q_fracs) / len(q_fracs), 4)
    pooled = round(n_derived / n_sent, 4)
    verbatim = round(n_verbatim / n_derived, 4)
    if q_mean != overall["derivation"]["question_mean"]:
        raise ValueError("question-mean derivation disagrees with archived records")
    if pooled != overall["derivation"]["sentence_pooled"]:
        raise ValueError("pooled derivation disagrees with archived records")
    if verbatim != overall["verbatim_copy_among_derived"]["fraction"]:
        raise ValueError("verbatim fraction disagrees with archived records")

    for question_type, expected in overall["by_question_type"].items():
        rows = [row for row in per_question if row["type"] == question_type]
        if len(rows) != expected["n_questions"]:
            raise ValueError("question-type count disagrees for " + question_type)
        if sum(row["n_evidence"] for row in rows) != expected["n_evidence_sentences"]:
            raise ValueError("question-type sentence total disagrees for " + question_type)
        if sum(row["n_derived"] for row in rows) != expected["n_derived"]:
            raise ValueError("question-type derived total disagrees for " + question_type)
        if sum(row["n_verbatim"] for row in rows) != expected["n_verbatim"]:
            raise ValueError("question-type verbatim total disagrees for " + question_type)

    print("verified frozen per-question audit:", path)
    print("questions =", len(per_question), "| evidence sentences =", n_sent)
    print("derivation: question_mean =", q_mean, "| sentence_pooled =", pooled)
    print("verbatim-copy among derived =", verbatim,
          "(", n_verbatim, "/", n_derived, ")")
    print("question-type totals: verified")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Recompute or verify the GraphRAG-Bench per-unit audit.")
    parser.add_argument("--source", default=None,
                        help="Path to an independently obtained medical_questions.json.")
    parser.add_argument("--frozen-json", default=None,
                        help="Archived result to verify when the source is unavailable.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    source = args.source or SRC
    if not os.path.exists(source):
        frozen = args.frozen_json or ARCHIVED_JSON
        if not os.path.exists(frozen):
            frozen = OUT_JSON
        verify_frozen(frozen)
        return

    with io.open(source, encoding="utf-8") as f:
        data = json.load(f)
    print("loaded medical_questions:", len(data))

    agg = aggregate(data)

    print("--- overall ---")
    print("  questions:", agg["n_questions"],
          "| evidence sentences:", agg["n_evidence_sentences"])
    print("  no_relations:", agg["n_questions_without_relations"],
          "| no_evidence:", agg["n_questions_without_evidence"])
    print("  derivation: question_mean =", agg["derivation"]["question_mean"],
          "| sentence_pooled =", agg["derivation"]["sentence_pooled"])
    print("  verbatim-copy among derived:",
          agg["verbatim_copy_among_derived"]["fraction"],
          "(", agg["verbatim_copy_among_derived"]["n_verbatim"], "/",
          agg["verbatim_copy_among_derived"]["n_derived"], ")")
    print("--- by type ---")
    for t, tv in agg["by_question_type"].items():
        print("  %-22s deriv(q=%.4f,pool=%.4f) verbatim=%.4f n_q=%d n_sent=%d" % (
            t, tv["derivation_question_mean"], tv["derivation_sentence_pooled"],
            tv["verbatim_copy_fraction_among_derived"],
            tv["n_questions"], tv["n_evidence_sentences"]))

    out = {
        "purpose": "Auxiliary GraphRAG-Bench evidence-sentence surface diagnostic; not a per-unit estimate of the subgraph axis",
        "source_file": "external/graphragbench/raw/medical_questions.json",
        "scope_note": "The analyzed units are evidence sentences paired with source relations, not gold subgraphs. The numeric result does not estimate C_subgraph.",
        "rules": {
            "derivation": "evidence sentence derived=1 iff majority (>=%.2f) of content tokens of some evidence_relations string appear in normalized sentence; surface overlap only; conservative lower bound" % DERIV_FRAC,
            "verbatim_copy": "among derived, verbatim=1 iff token Jaccard>=%.2f OR substring either way between sentence and matched relation" % VERBATIM_JACCARD,
            "normalization": "lowercase + punctuation->space + whitespace collapse",
            "stopwords": "fixed small English content-token filter (deterministic)",
        },
        "census_reference": {"benchmark": "GraphRAG-Bench", "axis": "subgraph",
                             "bin": 3, "c_hat": 0.75,
                             "rubric": "auto-generated from KG triples or templates"},
        "overall": {k: v2 for k, v2 in agg.items() if k != "_per_question"},
        "interpretation": "Descriptive sentence-surface statistics only; not a verdict on the gold-subgraph provenance bin.",
        "_per_question": agg["_per_question"],
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote", OUT_JSON)

    md = build_md(agg)
    with io.open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print("wrote", OUT_MD)


if __name__ == "__main__":
    main()
