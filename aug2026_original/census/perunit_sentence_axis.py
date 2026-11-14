# -*- coding: utf-8 -*-
"""Surface-realization diagnostic for the 2WikiMultiHop sentence axis.

2Wiki publishes, per question, both (a) Wikidata
evidence triples and (b) sentence-level supporting facts.  We therefore judge,
for every supporting-fact sentence, whether it is a surface realization of
some evidence triple, and compute the per-unit average.  This is an auxiliary
lexical diagnostic: it does not recover unit-level construction provenance and
therefore does not independently validate the ordinal census bin.

Judgment rule (deterministic, conservative, no aliases, no network):
  A supporting-fact sentence (document `title`, sentence index `sent_idx`)
  counts as realization=1 iff there exists an evidence triple (s, r, o) with
    (i)  norm(title) == norm(s)   [primary]  -- norm = lowercase + whitespace
         collapse; a secondary variant additionally strips one trailing
         parenthetical disambiguator from the title, e.g. "X (film)" -> "X";
    (ii) norm(sentence text) contains norm(o) as a substring.
  Otherwise realization=0.  Because aliases / paraphrases of the object and
  subject are NOT expanded, this is a conservative LOWER BOUND on the true
  realization rate.

Full recomputation writes ``census/perunit_sentence_axis.recomputed.json`` and
``census/perunit_sentence_axis_summary.recomputed.md`` (or the same names next
to this script in the development tree).  In the submission archive the
upstream 2Wiki raw ZIP is not redistributed; running this script there instead
verifies the shipped per-question records and aggregates.  Pass ``--raw-zip``
to recompute from an independently obtained 2Wiki release.
"""

import argparse
import io
import json
import os
import re
import zipfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
RAW_ZIP = os.path.join(BASE, "external", "2wiki", "raw", "data.zip")


def first_existing(*paths):
    """Use the archive layout when present, otherwise the development layout."""
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


SUB_QA = first_existing(
    os.path.join(BASE, "swap", "data", "sub200", "qa.jsonl"),
    os.path.join(BASE, "external", "2wiki", "sub200", "qa.jsonl"),
)
HOLD_QA = first_existing(
    os.path.join(BASE, "swap", "data", "holdout200", "qa.jsonl"),
    os.path.join(BASE, "external", "2wiki", "holdout200", "qa.jsonl"),
)
# Recompute outputs land next to the archived copy, under new names, so a
# --raw-zip recomputation never silently diverges from (or clobbers) the
# frozen census/perunit_sentence_axis.json that verify mode reads.
OUT_JSON = os.path.join(SCRIPT_DIR, "perunit_sentence_axis.recomputed.json")
OUT_MD = os.path.join(SCRIPT_DIR, "perunit_sentence_axis_summary.recomputed.md")
ARCHIVED_JSON = os.path.join(SCRIPT_DIR, "perunit_sentence_axis.json")

_WS = re.compile(r"\s+")
_TRAIL_PAREN = re.compile(r"\s*\([^()]*\)\s*$")


def norm(s):
    return _WS.sub(" ", s.strip().lower())


def strip_paren(title):
    return _TRAIL_PAREN.sub("", title).strip()


def load_jsonl(path):
    out = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def judge_question(item):
    """Return list of per-sentence records for one dev item."""
    ctx = {c[0]: c[1] for c in item["context"]}
    triples = item.get("evidences") or []
    recs = []
    for title, sent_idx in item["supporting_facts"]:
        sents = ctx.get(title)
        if sents is None or not (0 <= sent_idx < len(sents)):
            recs.append({"title": title, "sent_idx": sent_idx,
                         "status": "missing_sentence",
                         "realized_primary": 0, "realized_paren": 0})
            continue
        sent_n = norm(sents[sent_idx])
        t_n = norm(title)
        t_np = norm(strip_paren(title))
        r_primary = 0
        r_paren = 0
        for tr in triples:
            if len(tr) != 3:
                continue
            s_n, o_n = norm(tr[0]), norm(tr[2])
            if not o_n:
                continue
            obj_in = o_n in sent_n
            if obj_in and s_n == t_n:
                r_primary = 1
            if obj_in and (s_n == t_n or s_n == t_np):
                r_paren = 1
        recs.append({"title": title, "sent_idx": sent_idx, "status": "ok",
                     "realized_primary": r_primary, "realized_paren": r_paren})
    return recs


def aggregate(items, tag):
    """Compute question-mean and sentence-pooled realization rates."""
    q_fracs_primary, q_fracs_paren = [], []
    n_sent = 0
    n_real_primary = 0
    n_real_paren = 0
    n_missing = 0
    n_q_no_ev = 0
    per_q = []
    by_type = {}
    for it in items:
        recs = judge_question(it)
        k = len(recs)
        n_sent += k
        miss = sum(1 for r in recs if r["status"] == "missing_sentence")
        n_missing += miss
        if not (it.get("evidences") or []):
            n_q_no_ev += 1
        rp = sum(r["realized_primary"] for r in recs)
        rq = sum(r["realized_paren"] for r in recs)
        n_real_primary += rp
        n_real_paren += rq
        if k > 0:
            q_fracs_primary.append(rp / k)
            q_fracs_paren.append(rq / k)
        per_q.append({"_id": it["_id"], "type": it.get("type"),
                      "n_sf": k, "n_realized_primary": rp,
                      "n_realized_paren": rq})
        ty = it.get("type") or "unknown"
        ta = by_type.setdefault(ty, [0, 0, 0])
        ta[0] += k
        ta[1] += rp
        ta[2] += rq
    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0
    # distribution of per-question fractions (paren variant, 5 bins)
    bins = [0, 0, 0, 0, 0]  # [0,.25) [.25,.5) [.5,.75) [.75,1) [1]
    for f in q_fracs_paren:
        if f >= 1.0:
            bins[4] += 1
        else:
            bins[min(int(f / 0.25), 3)] += 1
    return {
        "subset": tag,
        "n_questions": len(items),
        "n_questions_without_evidences": n_q_no_ev,
        "n_supporting_sentences": n_sent,
        "n_missing_sentences": n_missing,
        "primary_rule": {
            "question_mean": round(mean(q_fracs_primary), 4),
            "sentence_pooled": round(n_real_primary / n_sent, 4) if n_sent else 0.0,
            "n_realized": n_real_primary,
        },
        "paren_stripped_variant": {
            "question_mean": round(mean(q_fracs_paren), 4),
            "sentence_pooled": round(n_real_paren / n_sent, 4) if n_sent else 0.0,
            "n_realized": n_real_paren,
        },
        "per_question_fraction_hist_paren": {
            "[0.00,0.25)": bins[0], "[0.25,0.50)": bins[1],
            "[0.50,0.75)": bins[2], "[0.75,1.00)": bins[3], "[1.00]": bins[4],
        },
        "by_type_sentence_pooled": {
            t: {"n_sentences": v[0],
                "primary": round(v[1] / v[0], 4) if v[0] else 0.0,
                "paren_stripped": round(v[2] / v[0], 4) if v[0] else 0.0}
            for t, v in sorted(by_type.items())
        },
        "_per_question": per_q,
    }


def _find_subset(payload, name):
    for subset in payload.get("subsets", []):
        if subset.get("subset") == name:
            return subset
    raise ValueError("missing subset: " + name)


def verify_frozen(path):
    """Validate the archived per-question records and reported aggregates."""
    with io.open(path, encoding="utf-8") as f:
        payload = json.load(f)
    audit = _find_subset(payload, "sub200+holdout200 (400)")
    full = _find_subset(payload, "dev_full (12576)")
    per_question = payload.get("per_question_400", [])

    if len(per_question) != 400:
        raise ValueError("expected 400 archived per-question records")
    n_sent = sum(row["n_sf"] for row in per_question)
    n_primary = sum(row["n_realized_primary"] for row in per_question)
    n_paren = sum(row["n_realized_paren"] for row in per_question)
    if (n_sent, n_primary, n_paren) != (800, 607, 650):
        raise ValueError("archived 400-item totals do not match the frozen audit")
    if audit["n_supporting_sentences"] != n_sent:
        raise ValueError("400-item sentence total disagrees with the aggregate")
    if audit["primary_rule"]["n_realized"] != n_primary:
        raise ValueError("400-item primary total disagrees with the aggregate")
    if audit["paren_stripped_variant"]["n_realized"] != n_paren:
        raise ValueError("400-item paren-stripped total disagrees with the aggregate")

    question_mean = sum(row["n_realized_paren"] / row["n_sf"]
                        for row in per_question if row["n_sf"]) / len(per_question)
    pooled = n_paren / n_sent
    if abs(question_mean - audit["paren_stripped_variant"]["question_mean"]) > 5e-5:
        raise ValueError("400-item question mean disagrees with per-question records")
    if abs(pooled - audit["paren_stripped_variant"]["sentence_pooled"]) > 5e-5:
        raise ValueError("400-item pooled rate disagrees with per-question records")

    for subset in (audit, full):
        hist_total = sum(subset["per_question_fraction_hist_paren"].values())
        if hist_total != subset["n_questions"]:
            raise ValueError("histogram total disagrees for " + subset["subset"])
        realized = subset["paren_stripped_variant"]["n_realized"]
        sentences = subset["n_supporting_sentences"]
        expected = round(realized / sentences, 4)
        if expected != subset["paren_stripped_variant"]["sentence_pooled"]:
            raise ValueError("pooled rate disagrees for " + subset["subset"])

    print("verified frozen per-question audit:", path)
    print("400 items: supporting sentences =", n_sent,
          "| primary =", round(n_primary / n_sent, 4),
          "| paren-stripped =", round(pooled, 4))
    print("full-dev aggregate arithmetic and histograms: verified")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Recompute or verify the 2Wiki sentence-axis audit.")
    parser.add_argument("--raw-zip", default=None,
                        help="Path to an independently obtained 2Wiki data ZIP.")
    parser.add_argument("--sub-qa", default=SUB_QA,
                        help="Path to the audited sub200 qa.jsonl.")
    parser.add_argument("--hold-qa", default=HOLD_QA,
                        help="Path to the audited holdout200 qa.jsonl.")
    parser.add_argument("--frozen-json", default=None,
                        help="Archived result to verify when the raw ZIP is unavailable.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    raw_zip = args.raw_zip or RAW_ZIP
    if not os.path.exists(raw_zip):
        frozen = args.frozen_json or ARCHIVED_JSON
        if not os.path.exists(frozen):
            frozen = OUT_JSON
        verify_frozen(frozen)
        return

    with zipfile.ZipFile(raw_zip) as zf:
        with zf.open("data/dev.json") as f:
            dev = json.load(f)
    by_id = {d["_id"]: d for d in dev}
    sub = load_jsonl(args.sub_qa)
    hold = load_jsonl(args.hold_qa)
    used_ids = [q["qa_id"] for q in sub] + [q["qa_id"] for q in hold]
    if len(used_ids) != 400:
        raise ValueError("expected 400 audited QA ids, found %d" % len(used_ids))
    if len(set(used_ids)) != len(used_ids):
        raise ValueError("audited sub200/holdout200 QA ids are not unique")
    missing_ids = [qa_id for qa_id in used_ids if qa_id not in by_id]
    if missing_ids:
        raise ValueError(
            "%d audited QA ids are absent from data/dev.json (first: %s)"
            % (len(missing_ids), missing_ids[0])
        )
    matched = [by_id[qa_id] for qa_id in used_ids]
    print("dev size:", len(dev))
    print("used 400 ids matched in dev:", len(matched), "/", len(used_ids))

    agg400 = aggregate(matched, "sub200+holdout200 (400)")
    aggdev = aggregate(dev, "dev_full (12576)")

    for a in (agg400, aggdev):
        print("---", a["subset"], "---")
        print("  sentences:", a["n_supporting_sentences"],
              "(missing:", str(a["n_missing_sentences"]) + ",",
              "q_without_evidences:", str(a["n_questions_without_evidences"]) + ")")
        print("  primary  rule  : question_mean =", a["primary_rule"]["question_mean"],
              "| sentence_pooled =", a["primary_rule"]["sentence_pooled"])
        print("  paren-stripped : question_mean =", a["paren_stripped_variant"]["question_mean"],
              "| sentence_pooled =", a["paren_stripped_variant"]["sentence_pooled"])
        print("  per-question fraction hist (paren):",
              json.dumps(a["per_question_fraction_hist_paren"]))

    # strip bulky per-question detail from full-dev block in the JSON output
    out = {
        "purpose": "Auxiliary 2Wiki sentence-level surface-realization diagnostic; not an independent estimate of construction-time provenance circularity",
        "rule": {
            "primary": "realized=1 iff exists triple (s,r,o): norm(title)==norm(s) AND norm(object) substring of norm(sentence); norm=lowercase+whitespace-collapse; no aliases",
            "paren_stripped_variant": "same, but title additionally matched after stripping one trailing parenthetical disambiguator",
            "bias": "conservative lower bound (object aliases/paraphrases not expanded)",
        },
        "census_reference": {"axis": "2wiki sentence", "bin": 3, "c_hat": 0.75},
        "subsets": [
            {k: v for k, v in agg400.items() if k != "_per_question"},
            {k: v for k, v in aggdev.items() if k != "_per_question"},
        ],
        "per_question_400": agg400["_per_question"],
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote", OUT_JSON)

    md = build_md(agg400, aggdev)
    with io.open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print("wrote", OUT_MD)


def build_md(agg400, aggdev):
    qm = agg400["paren_stripped_variant"]["question_mean"]
    sp = agg400["paren_stripped_variant"]["sentence_pooled"]
    lines = []
    a = lines.append
    a("# 2Wiki Sentence-Axis Surface-Realization Diagnostic")
    a("")
    a("Generated by `census/perunit_sentence_axis.py` (deterministic, offline).")
    a("")
    a("## Setup")
    a("")
    a("2WikiMultiHop publishes, per question, Wikidata evidence triples and sentence-level")
    a("supporting facts. We judge every supporting-fact sentence for whether it is a surface")
    a("realization of some evidence triple, then average. The census places the 2Wiki sentence")
    a("axis in bin 3 (C-hat = 0.75, KG derivation). The measured average is an auxiliary")
    a("surface statistic, not a direct reconstruction of each unit's construction lineage.")
    a("")
    a("## Judgment rule (deterministic, conservative)")
    a("")
    a("A sentence at (title, sent_idx) is realized (=1) iff some triple (s, r, o) satisfies:")
    a("")
    a("1. norm(title) == norm(s), where norm = lowercase + whitespace collapse.")
    a("   The paren-stripped variant also accepts a match after removing one trailing")
    a("   parenthetical disambiguator from the title (e.g. \"X (film)\" -> \"X\").")
    a("2. norm(o) appears as a substring of the normalized sentence text.")
    a("")
    a("No alias expansion is performed for subjects or objects, so both estimates are")
    a("conservative lower bounds: a sentence expressing the triple through an alias,")
    a("pronoun, or paraphrase of the object is scored 0.")
    a("")
    a("## Results")
    a("")
    a("| Subset | Questions | Sentences | Rule | Question-mean | Sentence-pooled |")
    a("|---|---|---|---|---|---|")
    for ag in (agg400, aggdev):
        for key, label in (("primary_rule", "primary"), ("paren_stripped_variant", "paren-stripped")):
            a("| %s | %d | %d | %s | %.4f | %.4f |" % (
                ag["subset"], ag["n_questions"], ag["n_supporting_sentences"],
                label, ag[key]["question_mean"], ag[key]["sentence_pooled"]))
    a("")
    a("Per-question realization-fraction distribution (400 items, paren-stripped variant):")
    a("")
    h = agg400["per_question_fraction_hist_paren"]
    a("| [0.00,0.25) | [0.25,0.50) | [0.50,0.75) | [0.75,1.00) | [1.00] |")
    a("|---|---|---|---|---|")
    a("| %d | %d | %d | %d | %d |" % (h["[0.00,0.25)"], h["[0.25,0.50)"],
                                       h["[0.50,0.75)"], h["[0.75,1.00)"], h["[1.00]"]))
    a("")
    hd = aggdev["per_question_fraction_hist_paren"]
    a("Full dev (12,576 items): | %d | %d | %d | %d | %d | in the same bins." % (
        hd["[0.00,0.25)"], hd["[0.25,0.50)"], hd["[0.50,0.75)"], hd["[0.75,1.00)"], hd["[1.00]"]))
    a("")
    a("Sentence-pooled realization by question type (full dev, paren-stripped):")
    a("")
    for ty, tv in aggdev["by_type_sentence_pooled"].items():
        a("- %s: %.4f (n = %d sentences)" % (ty, tv["paren_stripped"], tv["n_sentences"]))
    a("")
    a("All 400 evaluation questions are of type `inference`, which has the lowest lexical")
    a("realization rate among the four 2Wiki question types; the full-dev average is higher")
    a("because comparison-style questions almost always satisfy this subject/object")
    a("surface-realization rule. The rule does not measure full-sentence verbatim copying.")
    a("")
    a("Data integrity: %d/%d supporting-fact sentences resolved in context on the 400 set" % (
        agg400["n_supporting_sentences"] - agg400["n_missing_sentences"],
        agg400["n_supporting_sentences"]))
    a("(missing: %d); %d questions lacked an evidences list." % (
        agg400["n_missing_sentences"], agg400["n_questions_without_evidences"]))
    a("")
    a("## Interpretation relative to census bin 3 (C-hat = 0.75)")
    a("")
    a("On the 400 evaluation questions the measured per-unit average is %.3f (question-mean)" % qm)
    a("and %.3f (sentence-pooled) under the paren-stripped rule, a conservative lower bound" % sp)
    a("because object aliases and paraphrases are not credited. Its numerical proximity to")
    a("0.75 is descriptive only: surface realization is not the construction-lineage quantity")
    a("defined by C_a. The bin-3 assignment therefore continues to rest on the documented")
    a("Wikidata-to-supporting-sentence construction, not on this statistic alone.")
    a("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
