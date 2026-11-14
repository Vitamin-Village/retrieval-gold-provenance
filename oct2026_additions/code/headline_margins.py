"""Headline figure and verdict-flip analysis from frozen results (no retrieval or reader runs).

For every benchmark x split x graph retriever, the graph-over-BM25 hit@5 margin is computed under
the gold-aligned KG and under the independent OpenIE KG from the stored per-question hit@5 vectors
(external/<bench>/<split>/_kgswap_*.json), with an exact two-sided sign test against BM25 on the
same questions. The script then
  (a) counts "graph beats BM25" verdicts (margin > 0 and sign-test p < 0.05) per KG arm and lists
      the cells whose verdict flips between the arms (Holm over the cells of an arm as sensitivity);
  (b) computes Kendall tau-b between the retriever rankings under the two KGs per benchmark x
      split, with BM25 and the BGE-M3 dense-only ranking ([CLS] pooling) as KG-independent entries;
  (c) repeats (a) and (b) at the answer level (alias-aware EM; Mistral-Small-24B judge accuracy on
      GraphRAG-Bench) for the Qwen2.5-14B and Llama-3.1-8B readers, from per_question.jsonl.
Every number is cross-checked against the frozen summaries, tables/tab_e2e_merged_v3.tex
(manuscript Table 3, confirmed through the .aux), the manuscript's HippoRAG 2 table, and the
de-circularized prose ranges; the script aborts before writing anything if a check fails. A check
whose manuscript or table file is absent is skipped with a printed note.

Outputs: analysis/headline_margins_summary.{json,md}, figures/fig_headline_margins.{pdf,png}
Usage (from paper/): python analysis/headline_margins.py
  With SOURCE_DATE_EPOCH set (seconds since 1970, UTC), that time is written as the generation time
  instead of the current time, so a rerun can reproduce a stored summary byte for byte.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.e2e_qa_swap import sign_test  # noqa: E402  (exact two-sided, same as every paper p-value)
from eval.statistics import holm_bonferroni  # noqa: E402

EXT, RES, TAB, FIG, OUT = ROOT / "external", ROOT / "results", ROOT / "tables", ROOT / "figures", ROOT / "analysis"
ALPHA = 0.05

BENCHES = {"2wiki": "2WikiMultiHop", "metaqa": "MetaQA", "stark": "STaRK-Prime", "graphragbench": "GraphRAG-Bench"}
SPLITS = {"sub200": "first", "holdout200": "held-out"}
# key -> (label, per-question file pattern, results key, e2e system name, frozen summary file or None)
RETRIEVERS = {
    "hybrid": ("Hybrid", "_kgswap_kg_{kg}.json", "proposed_v3", "audited", "kg_swap_{b}.json"),
    "ppr": ("PPR", "_kgswap_ppr_kg_{kg}.json", "ppr", "ppr", "kg_swap_{b}_ppr.json"),
    "hybrid_ppr": ("Hybrid-PPR", "_kgswap_proposed_v5_graphcentric_kg_{kg}.json", "proposed_v5_graphcentric",
                   "graph_centric", "kg_swap_{b}_proposed_v5_graphcentric.json"),
    "hipporag2": ("HippoRAG 2", "_kgswap_hipporag2_kg_{kg}.json", "hipporag2", "hipporag2", None),
}
HIPPO_BENCHES = ("2wiki", "metaqa", "graphragbench")
ARMS = {"aligned": "silver", "independent": "openie"}
READERS = {"qwen2_5_14b": "Qwen2.5-14B", "llama3_1_8b": "Llama-3.1-8B"}
JUDGE = "mistral-small:24b"
TABLE_ROWS = {"Hybrid": "hybrid", "PPR": "ppr", "Hybrid-PPR": "hybrid_ppr"}

# Figure: Okabe-Ito vermillion / bluish green, the circular / independent colors of Figure 4.
# Validated with the dataviz validator (light surface): CVD dE 11.0, normal-vision dE 25.8, contrast >= 3:1.
C_ALIGNED, C_INDEP = "#D55E00", "#009E73"
C_REAL = "#0072B2"  # Okabe-Ito blue: structures not induced from the evaluated questions
# Realistic structures (Appendix C.3): a KG fetched from live Wikidata for the 2WikiMultiHop corpus entities and the
# full MetaQA KB, neither induced from the evaluated questions; per-question vectors under the same file scheme.
REALISTIC = {"2wiki": ("wikidata", "live Wikidata KG"), "metaqa": ("fullkb", "full MetaQA KB")}
REAL_PREFIX = {"hybrid": "proposed_v3", "ppr": "ppr", "hybrid_ppr": "proposed_v5_graphcentric", "hipporag2": "hipporag2"}
# Table 4 of the paper: realistic structure minus independent KG, first split (cross-check)
TABLE4_REAL = {("2wiki", "hybrid"): -0.045, ("2wiki", "ppr"): 0.075, ("2wiki", "hybrid_ppr"): -0.015, ("2wiki", "hipporag2"): -0.085,
               ("metaqa", "hybrid"): -0.080, ("metaqa", "ppr"): 0.060, ("metaqa", "hybrid_ppr"): -0.020, ("metaqa", "hipporag2"): -0.060}
C_TEXT, C_MUTED, C_GRID, C_LINK, C_ZERO, C_SURFACE = "#1a1a1a", "#555555", "#e4e4e4", "#b3b3b3", "#333333", "#ffffff"

CHECKS: list[dict] = []


def generated_utc() -> str:
    """Current UTC time, or SOURCE_DATE_EPOCH when that variable is set (reproducible reruns)."""
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    t = datetime.fromtimestamp(int(epoch), timezone.utc) if epoch else datetime.now(timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def check(what: str, expected: float, got: float, tol: float = 1e-9) -> None:
    CHECKS.append({"check": what, "expected": float(expected), "got": float(got), "ok": bool(abs(float(expected) - float(got)) <= tol)})


def cells():
    for b in BENCHES:
        for sp in SPLITS:
            for r in RETRIEVERS:
                if r != "hipporag2" or b in HIPPO_BENCHES:
                    yield b, sp, r


def paired(x: np.ndarray, y: np.ndarray) -> dict:
    """x minus y on the same questions: mean difference, discordant counts, exact sign test."""
    d = x - y
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    return {"margin": float(d.mean()), "discordant_pos": pos, "discordant_neg": neg, "sign_test_p": sign_test(pos, neg)}


def verdict(margin: float, p: float) -> str:
    return "beats" if margin > 0 and p < ALPHA else "below" if margin < 0 and p < ALPHA else "n.s."


def add_holm(records: list[dict], arm_keys: tuple[str, ...]) -> None:
    """Holm over the cells of one arm; adds p_holm and verdict_holm to each arm entry."""
    for arm in arm_keys:
        adj = holm_bonferroni({str(i): rec[arm]["sign_test_p"] for i, rec in enumerate(records)}, alpha=ALPHA)
        for i, rec in enumerate(records):
            rec[arm]["p_holm"] = adj[str(i)][0]
            rec[arm]["verdict_holm"] = verdict(rec[arm]["margin"], adj[str(i)][0])


def ranking(names: list[str], scores: list[float]) -> str:
    order = sorted(zip(scores, names), key=lambda t: -t[0])
    out, prev = "", None
    for s, n in order:
        out += ("" if prev is None else (" = " if abs(s - prev) < 1e-12 else " > ")) + n
        prev = s
    return out


def tau_entry(names: list[str], aligned: list[float], indep: list[float], n_kg_free: int) -> dict:
    tau_all = kendalltau(aligned, indep)[0]
    tau_graph = kendalltau([aligned[0]] + aligned[n_kg_free:], [indep[0]] + indep[n_kg_free:])[0]
    return {"items": names, "aligned": aligned, "independent": indep, "tau_b": float(tau_all),
            "tau_b_without_dense_only": float(tau_graph),
            "ranking_aligned": ranking(names, aligned), "ranking_independent": ranking(names, indep)}


def flip_counts(records: list[dict], vkey: str) -> dict:
    a = [r["aligned"][vkey] for r in records]
    i = [r["independent"][vkey] for r in records]
    flips = [r["cell"] for r in records if r["aligned"][vkey] == "beats" and r["independent"][vkey] != "beats"]
    return {"n_cells": len(records), "beats_aligned": a.count("beats"), "beats_independent": i.count("beats"),
            "below_aligned": a.count("below"), "below_independent": i.count("below"),
            "flips_beats_to_not": len(flips),
            "flips_beats_to_below": sum(1 for r in records if r["aligned"][vkey] == "beats" and r["independent"][vkey] == "below"),
            "flips_not_to_beats": sum(1 for r in records if r["aligned"][vkey] != "beats" and r["independent"][vkey] == "beats"),
            "flipped_cells": flips}


# ----------------------------------------------------------------------------- retrieval level
def load_hits(path: Path, key: str) -> dict[str, float]:
    d = json.load(open(path, encoding="utf-8"))
    pq = d["results"][key]["summary"]["per_qa"]
    assert set(pq["hit_at_k"]) <= {0.0, 1.0}, f"non-binary hit@5 in {path}"
    return dict(zip(pq["qa_id"], pq["hit_at_k"]))


def retrieval():
    bm25, recs = {}, []
    for b in BENCHES:
        for sp in SPLITS:
            ref = load_hits(EXT / b / sp / "_kgswap_kg_silver.json", "bm25")
            for r in ("hybrid", "ppr", "hybrid_ppr"):
                for kg in ARMS.values():
                    other = load_hits(EXT / b / sp / RETRIEVERS[r][1].format(kg=kg), "bm25")
                    check(f"BM25 vector identical: {b}/{sp}/{r}/{kg}", 1, float(other == ref))
            bm25[(b, sp)] = ref
    for b, sp, r in cells():
        label, pat, key, _, _ = RETRIEVERS[r]
        qids = list(bm25[(b, sp)])
        y = np.array([bm25[(b, sp)][q] for q in qids])
        vec = {}
        for arm, kg in ARMS.items():
            h = load_hits(EXT / b / sp / pat.format(kg=kg), key)
            assert set(h) == set(qids), f"question sets differ: {b}/{sp}/{r}/{kg}"
            vec[arm] = np.array([h[q] for q in qids])
        rec = {"cell": f"{BENCHES[b]} / {SPLITS[sp]} / {label}", "bench": b, "split": sp, "retriever": r, "n": len(qids),
               "bm25_hit5": float(y.mean())}
        for arm in ARMS:
            rec[arm] = {"hit5": float(vec[arm].mean()), **paired(vec[arm], y)}
            rec[arm]["verdict"] = verdict(rec[arm]["margin"], rec[arm]["sign_test_p"])
        rec["swap"] = paired(vec["aligned"], vec["independent"])
        recs.append(rec)
    add_holm(recs, tuple(ARMS))
    for rec in recs:
        rec["flip"] = rec["aligned"]["verdict"] == "beats" and rec["independent"]["verdict"] != "beats"
        rec["flip_holm"] = rec["aligned"]["verdict_holm"] == "beats" and rec["independent"]["verdict_holm"] != "beats"
    return recs


def check_retrieval(recs: list[dict]) -> None:
    by = {(r["bench"], r["split"], r["retriever"]): r for r in recs}
    # frozen main-text summaries (results/kg_swap_*.json)
    for (b, sp, r), rec in by.items():
        fname = RETRIEVERS[r][4]
        if fname is None:
            continue
        rows = json.load(open(RES / fname.format(b=b), encoding="utf-8"))
        row = next(x for x in rows if x["name"].endswith(sp))
        tag = f"results/{fname.format(b=b)} {sp}"
        check(f"{tag} aligned hit5", row["silver"]["graph"], rec["aligned"]["hit5"])
        check(f"{tag} independent hit5", row["openie"]["graph"], rec["independent"]["hit5"])
        check(f"{tag} BM25 hit5", row["silver"]["bm25"], rec["bm25_hit5"])
        check(f"{tag} aligned margin", row["silver"]["delta"], rec["aligned"]["margin"], 1e-6)
        check(f"{tag} independent margin", row["openie"]["delta"], rec["independent"]["margin"], 1e-6)
        check(f"{tag} swap", row["swap_delta"], rec["swap"]["margin"], 1e-6)
        if "sign_test_p" in row:
            check(f"{tag} swap sign-test p (relative)", 1, rec["swap"]["sign_test_p"] / row["sign_test_p"], 1e-9)
    # HippoRAG 2 summary
    hs = json.load(open(RES / "hipporag2_swap" / "summary.json", encoding="utf-8"))
    for s in hs["splits"]:
        rec = by[("2wiki", s["split"], "hipporag2")]
        check(f"hipporag2_swap/summary 2wiki {s['split']} aligned", s["hit5_aligned"], rec["aligned"]["hit5"])
        check(f"hipporag2_swap/summary 2wiki {s['split']} independent", s["hit5_independent"], rec["independent"]["hit5"])
        check(f"hipporag2_swap/summary 2wiki {s['split']} BM25", s["hit5_bm25"], rec["bm25_hit5"])
    for b in ("metaqa", "graphragbench"):
        for s in hs["benchmarks"][b]["splits"]:
            rec = by[(b, s["split"], "hipporag2")]
            check(f"hipporag2_swap/summary {b} {s['split']} aligned", s["hit5"]["kg_silver"], rec["aligned"]["hit5"])
            check(f"hipporag2_swap/summary {b} {s['split']} independent", s["hit5"]["kg_openie"], rec["independent"]["hit5"])
            check(f"hipporag2_swap/summary {b} {s['split']} BM25", s["hit5_bm25"], rec["bm25_hit5"])
            p = s["paired"]["kg_silver - kg_openie"]["sign_test_p"]
            check(f"hipporag2_swap/summary {b} {s['split']} swap p (relative)", 1, rec["swap"]["sign_test_p"] / p, 1e-9)


def realistic(ret: list[dict]) -> list[dict]:
    """Graph-over-BM25 verdicts with the realistic structures, on the 16 cells where they were run."""
    R = {(r["bench"], r["split"], r["retriever"]): r for r in ret}
    recs = []
    for b, (kg, lab) in REALISTIC.items():
        for sp in SPLITS:
            bm = load_hits(EXT / b / sp / "_kgswap_kg_silver.json", "bm25")
            qids = list(bm)
            y = np.array([bm[q] for q in qids])
            for r in RETRIEVERS:
                path = EXT / b / sp / f"_kgswap_{REAL_PREFIX[r]}_kg_{kg}.json"
                h = load_hits(path, RETRIEVERS[r][2])
                assert set(h) == set(qids), f"question sets differ: {path}"
                if "bm25" in json.load(open(path, encoding="utf-8"))["results"]:
                    check(f"BM25 vector identical: {b}/{sp}/{r}/{kg}", 1, float(load_hits(path, "bm25") == bm))
                ind = load_hits(EXT / b / sp / RETRIEVERS[r][1].format(kg=ARMS["independent"]), RETRIEVERS[r][2])
                x, xi = np.array([h[q] for q in qids]), np.array([ind[q] for q in qids])
                rec = {"cell": R[(b, sp, r)]["cell"], "bench": b, "split": sp, "retriever": r, "structure": lab,
                       "realistic": {"hit5": float(x.mean()), **paired(x, y)}, "realistic_minus_independent": paired(x, xi),
                       "aligned_verdict": R[(b, sp, r)]["aligned"]["verdict"], "independent_verdict": R[(b, sp, r)]["independent"]["verdict"]}
                rec["realistic"]["verdict"] = verdict(rec["realistic"]["margin"], rec["realistic"]["sign_test_p"])
                if sp == "sub200":
                    check(f"Table 4 realistic minus independent: {b}/{r}", TABLE4_REAL[(b, r)], rec["realistic_minus_independent"]["margin"], 1e-6)
                recs.append(rec)
    add_holm(recs, ("realistic",))
    return recs


def realistic_counts(real: list[dict]) -> dict:
    v = [r["realistic"]["verdict"] for r in real]
    return {"n_cells": len(real), "beats_realistic": v.count("beats"), "below_realistic": v.count("below"),
            "beats_realistic_holm": sum(r["realistic"]["verdict_holm"] == "beats" for r in real),
            "beats_aligned_same_cells": sum(r["aligned_verdict"] == "beats" for r in real),
            "beats_independent_same_cells": sum(r["independent_verdict"] == "beats" for r in real)}


def dense_only_retrieval(recs: list[dict]) -> dict:
    d = json.load(open(RES / "kg_swap_dense_bge_m3_cls.json", encoding="utf-8"))
    assert d["pooling"] == "cls"
    bm25 = {(r["bench"], r["split"]): r["bm25_hit5"] for r in recs}
    out = {}
    for row in d["rows"]:
        k = (row["bench"], row["split"])
        assert out.get(k, row["dense_only"]) == row["dense_only"]
        out[k] = row["dense_only"]
        check(f"kg_swap_dense_bge_m3_cls {k[0]}/{k[1]}/{row['system']} BM25 = frozen BM25", row["bm25"], bm25[k])
    return out


def retrieval_tau(recs: list[dict], dense: dict) -> list[dict]:
    by = {(r["bench"], r["split"], r["retriever"]): r for r in recs}
    out = []
    for b in BENCHES:
        for sp in SPLITS:
            rs = [r for r in RETRIEVERS if (b, sp, r) in by]
            names = ["BM25", "Dense-only"] + [RETRIEVERS[r][0] for r in rs]
            bm = by[(b, sp, rs[0])]["bm25_hit5"]
            al = [bm, dense[(b, sp)]] + [by[(b, sp, r)]["aligned"]["hit5"] for r in rs]
            ind = [bm, dense[(b, sp)]] + [by[(b, sp, r)]["independent"]["hit5"] for r in rs]
            out.append({"bench": b, "split": sp, **tau_entry(names, al, ind, 2)})
    return out


# ----------------------------------------------------------------------------- answer level
def load_judge_cache() -> dict[str, str]:
    cache = {}
    for line in open(RES / "e2e_qa" / "judge_cache_mistral_small_24b.jsonl", encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        cache[r["key"]] = r["verdict"]
    return cache


def answer_scores(tag: str, cache: dict[str, str]) -> tuple[dict, int]:
    """(bench, split, system, condition) -> {qa_id: correct}. EM, or judge verdict on GraphRAG-Bench
    reconstructed exactly as eval/e2e_qa_judge.py keys it (empty prediction = incorrect)."""
    out, missing = {}, 0
    for line in open(RES / "e2e_qa" / tag / "per_question.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["bench"] == "graphragbench":
            gold = r["golds"][0] if r["golds"] else ""
            key = hashlib.sha1("\x00".join([JUDGE, r["bench"], r["qa_id"], gold, r["pred"]]).encode("utf-8")).hexdigest()
            if r["pred"].strip() == "":
                v = 0.0
            else:
                missing += key not in cache
                v = 1.0 if cache.get(key) == "CORRECT" else 0.0
        else:
            v = float(r["em"])
        out.setdefault((r["bench"], r["split"], r["system"], r["condition"]), {})[r["qa_id"]] = v
    return out, missing


def answer(tag: str, cache: dict[str, str], ret_hit5: dict):
    sc, missing = answer_scores(tag, cache)
    summ = json.load(open(RES / "e2e_qa" / tag / "summary.json", encoding="utf-8"))
    judge = json.load(open(RES / "e2e_qa" / tag / "judge_mistral_small_24b.json", encoding="utf-8"))
    ref_em = {(c["bench"], c["split"], c["system"], c["condition"]): c for c in summ["cells"]}
    ref_j = {(c["bench"], c["split"], c["system"], c["condition"]): c["judge_acc"] for c in judge["cells"]}
    recs = []
    for b, sp, r in cells():
        label, sysname = RETRIEVERS[r][0], RETRIEVERS[r][3]
        base = sc[(b, sp, "bm25", "none")]
        qids = list(base)
        y = np.array([base[q] for q in qids])
        rec = {"cell": f"{BENCHES[b]} / {SPLITS[sp]} / {label}", "bench": b, "split": sp, "retriever": r, "n": len(qids),
               "metric": "judge_acc" if b == "graphragbench" else "em", "bm25_acc": float(y.mean())}
        vec = {}
        for arm, kg in ARMS.items():
            s = sc[(b, sp, sysname, f"kg_{kg}")]
            assert set(s) == set(qids), f"question sets differ: {tag} {b}/{sp}/{sysname}/{kg}"
            vec[arm] = np.array([s[q] for q in qids])
            rec[arm] = {"acc": float(vec[arm].mean()), **paired(vec[arm], y)}
            rec[arm]["verdict"] = verdict(rec[arm]["margin"], rec[arm]["sign_test_p"])
        rec["swap"] = paired(vec["aligned"], vec["independent"])
        recs.append(rec)
        # cross-checks against the reader's stored summaries
        for sysn, cond, val in ((sysname, "kg_silver", rec["aligned"]["acc"]), (sysname, "kg_openie", rec["independent"]["acc"]),
                                ("bm25", "none", rec["bm25_acc"])):
            k = (b, sp, sysn, cond)
            refv = ref_j[k] if b == "graphragbench" else ref_em[k]["em"]
            check(f"e2e_qa/{tag} {'/'.join(k)} answer acc", refv, val)
        ret_hit = {"aligned": (sysname, "kg_silver"), "independent": (sysname, "kg_openie")}
        for arm, (sysn, cond) in ret_hit.items():
            check(f"e2e_qa/{tag} {b}/{sp}/{sysn}/{cond} hit5 = retrieval vector", ref_em[(b, sp, sysn, cond)]["hit5"], ret_hit5[(b, sp, r, arm)])
    add_holm(recs, tuple(ARMS))
    for rec in recs:
        rec["flip"] = rec["aligned"]["verdict"] == "beats" and rec["independent"]["verdict"] != "beats"
        rec["flip_holm"] = rec["aligned"]["verdict_holm"] == "beats" and rec["independent"]["verdict_holm"] != "beats"
    return recs, missing


def dense_only_answer(tag: str) -> dict:
    s = json.load(open(RES / "e2e_qa_cls" / tag / "summary.json", encoding="utf-8"))
    j = json.load(open(RES / "e2e_qa_cls" / tag / "judge_mistral_small_24b.json", encoding="utf-8"))
    out = {}
    for c in s["cells"]:
        if c["system"] == "dense" and c["bench"] != "graphragbench":
            out[(c["bench"], c["split"])] = c["em"]
    for c in j["cells"]:
        if c["system"] == "dense" and c["bench"] == "graphragbench":
            out[(c["bench"], c["split"])] = c["judge_acc"]
    return out


def answer_tau(recs: list[dict], dense: dict) -> list[dict]:
    by = {(r["bench"], r["split"], r["retriever"]): r for r in recs}
    out = []
    for b in BENCHES:
        for sp in SPLITS:
            rs = [r for r in RETRIEVERS if (b, sp, r) in by]
            names = ["BM25", "Dense-only"] + [RETRIEVERS[r][0] for r in rs]
            bm = by[(b, sp, rs[0])]["bm25_acc"]
            al = [bm, dense[(b, sp)]] + [by[(b, sp, r)]["aligned"]["acc"] for r in rs]
            ind = [bm, dense[(b, sp)]] + [by[(b, sp, r)]["independent"]["acc"] for r in rs]
            out.append({"bench": b, "split": sp, **tau_entry(names, al, ind, 2)})
    return out


# ----------------------------------------------------------------------------- manuscript cross-checks
def lead_float(cell: str) -> float:
    return float(re.match(r"\s*([+-]?\d+\.\d+)", cell).group(1))


def stars(p: float) -> int:
    return 2 if p < 0.01 else 1 if p < 0.05 else 0


def check_table3(ret: list[dict], ans: dict[str, list[dict]]) -> None:
    """Checks against the manuscript (.aux, .tex) and Table 3 (tables/tab_e2e_merged_v3.tex). A check group whose
    file is absent is skipped with a printed note; the prose checks at the end need no file."""
    aux_p, tex_p, tab_p = ROOT / "manuscript" / "eacl2027_camera_ready.aux", ROOT / "manuscript" / "eacl2027_camera_ready.tex", TAB / "tab_e2e_merged_v3.tex"
    for p in (aux_p, tex_p, tab_p):
        if not p.exists():
            print(f"NOTE: {p.relative_to(ROOT).as_posix()} not found; the cross-checks that read it are skipped")
    if aux_p.exists():
        aux = aux_p.read_text(encoding="utf-8", errors="ignore")
        check("manuscript: tab:e2e is Table 3 (.aux)", 1, float("\\newlabel{tab:e2e}{{3}" in aux))
    tex = tex_p.read_text(encoding="utf-8") if tex_p.exists() else None
    if tex is not None:
        marks = [tex.find(s) for s in ("\\input{tables/tab_e2e_merged_v3.tex}", "\\input{../tables/tab_e2e_merged_v3.tex}")]
        block = tex[min(i for i in marks if i >= 0):]
        check("manuscript: Table 3 inputs tables/tab_e2e_merged_v3.tex", 1, float(block.index("\\label{tab:e2e}") < 2000))
    R = {(r["bench"], r["split"], r["retriever"]): r for r in ret}
    A = {tag: {(r["bench"], r["split"], r["retriever"]): r for r in recs} for tag, recs in ans.items()}
    inv = {v: k for k, v in BENCHES.items()}
    bench, n_rows = None, 0
    for line in (tab_p.read_text(encoding="utf-8").split("\n") if tab_p.exists() else []):
        c = line.split(" & ")
        if len(c) != 13 or c[1] not in TABLE_ROWS:
            continue
        bench = inv.get(c[0], bench)
        r = TABLE_ROWS[c[1]]
        n_rows += 1
        f, h = R[(bench, "sub200", r)], R[(bench, "holdout200", r)]
        t = f"Table 3 {c[0] or BENCHES[bench]}/{c[1]}"
        check(f"{t} hit@5 aligned", lead_float(c[2]), f["aligned"]["hit5"], 5e-4 + 1e-9)
        check(f"{t} hit@5 indep.", lead_float(c[3]), f["independent"]["hit5"], 5e-4 + 1e-9)
        check(f"{t} hit@5 BM25", lead_float(c[4]), f["bm25_hit5"], 5e-4 + 1e-9)
        check(f"{t} hit@5 swap held-out", lead_float(c[5]), h["swap"]["margin"], 5e-4 + 1e-9)
        q = A["qwen2_5_14b"][(bench, "sub200", r)]
        check(f"{t} QA Qwen aligned", lead_float(c[6]), q["aligned"]["acc"], 5e-4 + 1e-9)
        check(f"{t} QA Qwen indep.", lead_float(c[7]), q["independent"]["acc"], 5e-4 + 1e-9)
        check(f"{t} QA Qwen BM25", lead_float(c[8]), q["bm25_acc"], 5e-4 + 1e-9)
        for idx, tag, sp in ((9, "llama3_1_8b", "sub200"), (10, "llama3_1_8b", "holdout200"),
                             (11, "qwen2_5_14b", "sub200"), (12, "qwen2_5_14b", "holdout200")):
            sw = A[tag][(bench, sp, r)]["swap"]
            check(f"{t} QA swap {tag} {sp}", lead_float(c[idx]), sw["margin"], 5e-4 + 1e-9)
            check(f"{t} QA swap {tag} {sp} significance stars", c[idx].count("*"), stars(sw["sign_test_p"]))
    if tab_p.exists():
        check("Table 3 retriever rows parsed", 12, n_rows)
    # HippoRAG 2 appendix table (tab:hipporag2more)
    hb = ""
    if tex is not None:
        end = tex.index("\\label{tab:hipporag2more}")
        hb = tex[tex.rindex("\\begin{tabular}", 0, end):end]
    n_h = 0
    for line in hb.split("\n"):
        c = [x.strip() for x in line.rstrip("\\").split("&")]
        if len(c) != 8 or c[0] not in inv:
            continue
        n_h += 1
        rec = R[(inv[c[0]], "sub200" if c[1] == "first" else "holdout200", "hipporag2")]
        t = f"tab:hipporag2more {c[0]}/{c[1]}"
        check(f"{t} aligned", float(c[2]), rec["aligned"]["hit5"], 5e-4 + 1e-9)
        check(f"{t} indep.", float(c[3]), rec["independent"]["hit5"], 5e-4 + 1e-9)
        check(f"{t} BM25", float(c[6]), rec["bm25_hit5"], 5e-4 + 1e-9)
        check(f"{t} swap", lead_float(c[7].strip("$")), rec["swap"]["margin"], 5e-4 + 1e-9)
    if tex is not None:
        check("tab:hipporag2more rows parsed", 6, n_h)
    # prose, 'De-circularized evaluation' paragraph: margin ranges on 2WikiMultiHop and HippoRAG 2 on GraphRAG-Bench
    ours = ("hybrid", "ppr", "hybrid_ppr")
    for sp, arm, lo, hi, tol in (("sub200", "aligned", 0.17, 0.39, 0.005), ("sub200", "independent", 0.01, 0.06, 0.005),
                                 ("holdout200", "aligned", 0.125, 0.385, 5e-4), ("holdout200", "independent", 0.010, 0.075, 5e-4)):
        m = [R[("2wiki", sp, r)][arm]["margin"] for r in ours]
        check(f"prose 2Wiki {sp} {arm} margin min", lo, min(m), tol + 1e-9)
        check(f"prose 2Wiki {sp} {arm} margin max", hi, max(m), tol + 1e-9)
    for sp, v in (("sub200", 0.020), ("holdout200", 0.070)):
        check(f"prose HippoRAG 2 GraphRAG-Bench {sp} independent margin", v, R[("graphragbench", sp, "hipporag2")]["independent"]["margin"], 5e-4 + 1e-9)
    check("prose: none of ours beats BM25 on GraphRAG-Bench first split, independent KG", 0,
          sum(R[("graphragbench", "sub200", r)]["independent"]["margin"] > 0 for r in ours))
    check("prose: best of ours on GraphRAG-Bench held-out, independent KG", 0.015,
          max(R[("graphragbench", "holdout200", r)]["independent"]["margin"] for r in ours), 5e-4 + 1e-9)
    # prose, same paragraph: answer-level margins over BM25
    for tag, arm, lo, hi in (("qwen2_5_14b", "aligned", 0.07, 0.36), ("llama3_1_8b", "aligned", 0.08, 0.38)):
        m = [A[tag][("2wiki", "sub200", r)][arm]["margin"] for r in ours]
        check(f"prose 2Wiki first EM margin min ({tag})", lo, min(m), 0.005 + 1e-9)
        check(f"prose 2Wiki first EM margin max ({tag})", hi, max(m), 0.005 + 1e-9)
    for tag, hi in (("qwen2_5_14b", 0.02), ("llama3_1_8b", 0.045)):
        check(f"prose 2Wiki first EM margin, independent KG, at most ({tag})", hi,
              max(A[tag][("2wiki", "sub200", r)]["independent"]["margin"] for r in ours), 5e-4 + 1e-9)
    grb = [A[t][("graphragbench", sp, r)] for t in READERS for sp in SPLITS for r in ours]
    check("prose GraphRAG-Bench answer margin of ours, aligned KG, up to", 0.09, max(x["aligned"]["margin"] for x in grb), 5e-4 + 1e-9)
    check("prose GraphRAG-Bench answer margin of ours, independent KG, at most", 0.02, max(x["independent"]["margin"] for x in grb), 5e-4 + 1e-9)
    hr = [A[t][("graphragbench", sp, "hipporag2")][arm]["margin"] for t in READERS for sp in SPLITS for arm in ARMS]
    check("prose GraphRAG-Bench HippoRAG 2 answer margin min", 0.015, min(hr), 5e-4 + 1e-9)
    check("prose GraphRAG-Bench HippoRAG 2 answer margin max", 0.115, max(hr), 5e-4 + 1e-9)


# ----------------------------------------------------------------------------- figure
# Verdict encoding: filled = graph significantly above BM25, hollow = not significant, and a hollow
# marker crossed by an x = graph significantly below BM25 (uncorrected sign test, p < 0.05).
BELOW_LABEL = "crossed: sig. below BM25"


def marker_kw(verdict_: str, col: str) -> dict:
    if verdict_ == "beats":
        return {"mfc": col, "mec": C_SURFACE, "mew": 0.45}
    return {"mfc": C_SURFACE, "mec": col, "mew": 0.8}


def draw_marker(ax, x: float, y: float, mk: str, ms: float, col: str, verdict_: str, z: float) -> None:
    ax.plot([x], [y], marker=mk, ms=ms, ls="", zorder=z, **marker_kw(verdict_, col))
    if verdict_ == "below":
        ax.plot([x], [y], marker="x", ms=ms * 0.62, ls="", mec=col, mew=0.8 if ms > 3 else 0.6, zorder=z + 0.1)


def marker_handle(ms: float, verdict_: str, label: str) -> tuple:
    """(legend handle, label); a sig.-below handle is a (marker, x overlay) tuple drawn by HandlerTuple."""
    base = Line2D([], [], marker="o", ls="", ms=ms, **marker_kw(verdict_, C_MUTED))
    if verdict_ != "below":
        return base, label
    return (base, Line2D([], [], marker="x", ls="", ms=ms * 0.62, mec=C_MUTED, mew=0.8)), label


def figure(ret: list[dict], real: list[dict]) -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.size": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "xtick.major.size": 2.5})
    R = {(r["bench"], r["split"], r["retriever"]): r for r in ret}
    RR = {(r["bench"], r["split"], r["retriever"]): r for r in real}
    layout, y = [], 0.0  # (kind, y, payload)
    for b in BENCHES:
        layout.append(("header", y, BENCHES[b]))
        y += 0.95
        for r in RETRIEVERS:
            if (b, "sub200", r) in R:
                layout.append(("row", y, (b, r)))
                y += 1.0
        y += 0.1
    n_slots = y
    # Drawn at the final \columnwidth (219.09 pt = 3.043 in), so font sizes are final sizes (>= 7.5 pt).
    # The third legend row (sig. below BM25) is paid for by a slightly smaller row pitch: the rendered
    # height stays below the 2026-09-30 version (3.41 in at \columnwidth).
    # The fourth legend row (realistic structure) is paid for by a smaller row pitch: the rendered height stays at or
    # below the 2026-10-02 version (3.39 in at \columnwidth).
    slot_in, top_in, bottom_in, left_in, right_in, width_in = 0.121, 0.69, 0.37, 0.68, 0.12, 219.09 / 72
    height_in = n_slots * slot_in + top_in + bottom_in
    fig = plt.figure(figsize=(width_in, height_in))
    ax = fig.add_axes([left_in / width_in, bottom_in / height_in, 1 - (left_in + right_in) / width_in, n_slots * slot_in / height_in])
    ax.set_facecolor(C_SURFACE)
    xlo, xhi = -0.4, 0.6
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(n_slots - 0.45, -0.55)
    for x in (-0.2, 0.2, 0.4):
        ax.axvline(x, color=C_GRID, lw=0.5, zorder=0)
    ax.axvline(0, color=C_ZERO, lw=0.8, zorder=1)
    hdr = blended_transform_factory(fig.transFigure, ax.transData)
    first_header = True
    for kind, yy, payload in layout:
        if kind == "header":
            ax.text(0.012, yy, payload, transform=hdr, ha="left", va="center", fontsize=8, fontweight="bold", color=C_TEXT)
            if yy > 0:
                ax.axhline(yy - 0.55, color=C_GRID, lw=0.5, zorder=0)
            if first_header:
                ax.text(0.012, yy, "no advantage over BM25", ha="left", va="center", fontsize=7.5, fontstyle="italic", color=C_MUTED)
                first_header = False
            continue
        b, r = payload
        for sp, dy, ms, lw in (("sub200", 0.0, 4.6, 0.9), ("holdout200", 0.35, 2.6, 0.5)):
            rec = R[(b, sp, r)]
            xa, xi = rec["aligned"]["margin"], rec["independent"]["margin"]
            ax.plot([xi, xa], [yy + dy, yy + dy], color=C_LINK, lw=lw, zorder=2, solid_capstyle="butt")
            for arm, x, col, mk, z in (("independent", xi, C_INDEP, "s", 3), ("aligned", xa, C_ALIGNED, "o", 4)):
                v = rec[arm]["verdict"]
                draw_marker(ax, x, yy + dy, mk, ms * (0.92 if mk == "s" else 1.0), col, v, z + (0.5 if v != "beats" else 0))
            if (b, sp, r) in RR:
                rr = RR[(b, sp, r)]["realistic"]
                draw_marker(ax, rr["margin"], yy + dy, "D", ms * 0.82, C_REAL, rr["verdict"], 3.6 + (0.5 if rr["verdict"] != "beats" else 0))
    ax.set_yticks([yy for kind, yy, _ in layout if kind == "row"])
    ax.set_yticklabels([RETRIEVERS[p[1]][0] for kind, _, p in layout if kind == "row"], fontsize=8, color=C_TEXT)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", labelsize=7.5, colors=C_TEXT, pad=1.5)
    ax.set_xticks(np.round(np.arange(-0.4, 0.61, 0.2), 1))
    ax.set_xticklabels(["\u22120.4", "\u22120.2", "0", "+0.2", "+0.4", "+0.6"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(C_MUTED)
    ax.set_xlabel("hit@5 margin over BM25 (graph − BM25)", fontsize=8, color=C_TEXT, labelpad=2)
    entries = [  # two columns, filled column-major: KG arm and split | verdict against BM25
        (Line2D([], [], marker="o", ls="", ms=4.6, mfc=C_ALIGNED, mec=C_SURFACE, mew=0.45), "aligned KG"),
        (Line2D([], [], marker="s", ls="", ms=4.2, mfc=C_INDEP, mec=C_SURFACE, mew=0.45), "independent (OpenIE) KG"),
        (Line2D([], [], marker="D", ls="", ms=3.8, mfc=C_REAL, mec=C_SURFACE, mew=0.45), "realistic, not induced"),
        (Line2D([], [], marker="o", ls="", ms=2.6, mfc=C_MUTED, mec=C_SURFACE, mew=0.3), "small: held-out split"),
        marker_handle(4.6, "beats", "filled: sig. above BM25"),
        marker_handle(4.6, "n.s.", "hollow: not significant"),
        marker_handle(4.6, "below", BELOW_LABEL),
    ]
    fig.legend(handles=[h for h, _ in entries], labels=[lab for _, lab in entries], loc="upper left",
               bbox_to_anchor=(0.005, 0.998), ncol=2, frameon=False, fontsize=7.5, handletextpad=0.35,
               columnspacing=1.2, borderaxespad=0.1, labelspacing=0.3, handlelength=1.0,
               handler_map={tuple: HandlerTuple(ndivide=1, pad=0)})
    FIG.mkdir(exist_ok=True)
    meta = {"Creator": "analysis/headline_margins.py"}
    fig.savefig(FIG / "fig_headline_margins.pdf", metadata=meta)
    fig.savefig(FIG / "fig_headline_margins.png", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------- outputs
def f3(x: float) -> str:
    return f"{x:+.3f}"


def pfmt(p: float) -> str:
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


def markdown(summary: dict) -> str:
    rc, rt = summary["retrieval"]["counts"], summary["retrieval"]["kendall_tau"]
    L = ["# Headline margins: graph-over-BM25 advantage under the aligned vs the independent KG", "",
         f"Generated {summary['generated']} by `analysis/headline_margins.py` from frozen per-question vectors; no retrieval or reader run. "
         f"{summary['cross_checks']['n_ok']}/{summary['cross_checks']['n']} cross-checks pass (frozen summaries, HippoRAG 2 summary, "
         "e2e summaries and judge files, Table 3 = `tables/tab_e2e_merged_v3.tex`, `tab:hipporag2more`, prose ranges).", "",
         "Margin = graph hit@5 minus BM25 hit@5 on the same 200 questions. Verdict *beats* = margin > 0 and exact two-sided sign test "
         "on the per-question pairs p < 0.05 (uncorrected); *below* = margin < 0 at p < 0.05; *n.s.* otherwise. Holm column: Holm over the "
         f"{rc['uncorrected']['n_cells']} cells of one arm. Cells: 4 benchmarks x 2 splits x {{Hybrid, PPR, Hybrid-PPR}} + HippoRAG 2 on "
         "2WikiMultiHop, MetaQA, GraphRAG-Bench x 2 splits.", "",
         "## (a) Retrieval verdicts", "",
         "| | aligned KG | independent KG | flips beats -> not |", "|---|---:|---:|---:|",
         f"| beats BM25 (p<0.05) | {rc['uncorrected']['beats_aligned']}/{rc['uncorrected']['n_cells']} | {rc['uncorrected']['beats_independent']}/{rc['uncorrected']['n_cells']} | {rc['uncorrected']['flips_beats_to_not']} |",
         f"| significantly below BM25 | {rc['uncorrected']['below_aligned']} | {rc['uncorrected']['below_independent']} | beats -> below: {rc['uncorrected']['flips_beats_to_below']} |",
         f"| beats BM25 (Holm) | {rc['holm']['beats_aligned']}/{rc['holm']['n_cells']} | {rc['holm']['beats_independent']}/{rc['holm']['n_cells']} | {rc['holm']['flips_beats_to_not']} |",
         "", f"Reverse flips (not beats -> beats): {rc['uncorrected']['flips_not_to_beats']} uncorrected, {rc['holm']['flips_not_to_beats']} Holm.", "",
         "Flipped cells (uncorrected): " + "; ".join(rc["uncorrected"]["flipped_cells"]) + ".", "",
         "| cell | BM25 | aligned hit@5 | margin | p | verdict | indep. hit@5 | margin | p | verdict | flip |",
         "|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|"]
    for r in summary["retrieval"]["cells"]:
        a, i = r["aligned"], r["independent"]
        L.append(f"| {r['cell']} | {r['bm25_hit5']:.3f} | {a['hit5']:.3f} | {f3(a['margin'])} | {pfmt(a['sign_test_p'])} | {a['verdict']} | "
                 f"{i['hit5']:.3f} | {f3(i['margin'])} | {pfmt(i['sign_test_p'])} | {i['verdict']} | {'yes' if r['flip'] else ''} |")
    q = summary["realistic"]["counts"]
    L += ["", "## (a2) Realistic structures not induced from the evaluated questions (live Wikidata KG on 2WikiMultiHop, full MetaQA KB)", "",
          f"On these {q['n_cells']} cells graph retrieval beats BM25 in {q['beats_aligned_same_cells']} with the aligned KG, "
          f"{q['beats_realistic']} with the realistic structure ({q['beats_realistic_holm']} under Holm over these cells; "
          f"{q['below_realistic']} significantly below), and {q['beats_independent_same_cells']} with the independent KG.", "",
          "| cell | structure | hit@5 | margin over BM25 | p | verdict | minus independent | p |", "|---|---|---:|---:|---:|---|---:|---:|"]
    for r in summary["realistic"]["cells"]:
        x, d = r["realistic"], r["realistic_minus_independent"]
        L.append(f"| {r['cell']} | {r['structure']} | {x['hit5']:.3f} | {f3(x['margin'])} | {pfmt(x['sign_test_p'])} | {x['verdict']} | "
                 f"{f3(d['margin'])} | {pfmt(d['sign_test_p'])} |")
    L += ["", "## (b) Kendall tau-b between retriever rankings, aligned vs independent KG (retrieval hit@5)", "",
          "Items: BM25 and BGE-M3 dense-only ([CLS] pooling, `results/kg_swap_dense_bge_m3_cls.json`; KG-independent, identical in both arms) "
          "plus the graph retrievers run on that benchmark. tau-b handles ties. The second tau drops dense-only.", "",
          "| benchmark | split | tau-b | tau-b w/o dense | ranking, aligned KG | ranking, independent KG |", "|---|---|---:|---:|---|---|"]
    for t in rt:
        L.append(f"| {BENCHES[t['bench']]} | {SPLITS[t['split']]} | {t['tau_b']:+.2f} | {t['tau_b_without_dense_only']:+.2f} | {t['ranking_aligned']} | {t['ranking_independent']} |")
    L += ["", "## (c) Answer level (alias-aware EM; Mistral-Small-24B judge accuracy on GraphRAG-Bench)", ""]
    for tag, lab in READERS.items():
        a = summary["answer"][tag]
        c = a["counts"]
        L += [f"### {lab}", "",
              f"beats BM25 (p<0.05): aligned {c['uncorrected']['beats_aligned']}/{c['uncorrected']['n_cells']}, independent "
              f"{c['uncorrected']['beats_independent']}/{c['uncorrected']['n_cells']}; flips beats -> not: {c['uncorrected']['flips_beats_to_not']} "
              f"(reverse: {c['uncorrected']['flips_not_to_beats']}); significantly below BM25: aligned {c['uncorrected']['below_aligned']}, "
              f"independent {c['uncorrected']['below_independent']}. Holm: aligned {c['holm']['beats_aligned']}, independent "
              f"{c['holm']['beats_independent']}, flips {c['holm']['flips_beats_to_not']}.", "",
              "Flipped cells: " + ("; ".join(c["uncorrected"]["flipped_cells"]) or "none") + ".", "",
              "| cell | BM25 | aligned | margin | p | verdict | indep. | margin | p | verdict |", "|---|---:|---:|---:|---:|---|---:|---:|---:|---|"]
        for r in a["cells"]:
            x, y = r["aligned"], r["independent"]
            L.append(f"| {r['cell']} | {r['bm25_acc']:.3f} | {x['acc']:.3f} | {f3(x['margin'])} | {pfmt(x['sign_test_p'])} | {x['verdict']} | "
                     f"{y['acc']:.3f} | {f3(y['margin'])} | {pfmt(y['sign_test_p'])} | {y['verdict']} |")
        L += ["", "| benchmark | split | tau-b | tau-b w/o dense | ranking, aligned KG | ranking, independent KG |", "|---|---|---:|---:|---|---|"]
        for t in a["kendall_tau"]:
            L.append(f"| {BENCHES[t['bench']]} | {SPLITS[t['split']]} | {t['tau_b']:+.2f} | {t['tau_b_without_dense_only']:+.2f} | {t['ranking_aligned']} | {t['ranking_independent']} |")
        L.append("")
    L += ["## Caption draft", "", summary["caption_draft"], "",
          "Figure: `figures/fig_headline_margins.pdf` (vector, TrueType fonts) and `.png` preview; 3.04 in (\\columnwidth) wide.", ""]
    return "\n".join(L)


def main() -> None:
    ret = retrieval()
    ret_hit5 = {(r["bench"], r["split"], r["retriever"], arm): r[arm]["hit5"] for r in ret for arm in ARMS}
    check_retrieval(ret)
    dense_r = dense_only_retrieval(ret)
    cache = load_judge_cache()
    ans, missing = {}, {}
    for tag in READERS:
        ans[tag], missing[tag] = answer(tag, cache, ret_hit5)
        check(f"judge cache covers every non-empty GraphRAG-Bench prediction ({tag})", 0, missing[tag])
    check_table3(ret, ans)
    real = realistic(ret)
    bad = [c for c in CHECKS if not c["ok"]]
    if bad:
        for c in bad:
            print("CHECK FAILED:", c)
        sys.exit(f"{len(bad)} of {len(CHECKS)} cross-checks failed; nothing written")

    summary = {
        "generated": generated_utc(),
        "script": "analysis/headline_margins.py",
        "definitions": {
            "margin": "graph score minus BM25 score on the same questions (hit@5, or answer accuracy)",
            "test": "exact two-sided sign test on per-question pairs (eval/e2e_qa_swap.sign_test)",
            "verdict": "beats: margin>0 and p<0.05; below: margin<0 and p<0.05; n.s. otherwise",
            "holm": "Holm-Bonferroni over the cells of one arm (30 per arm per level/reader)",
            "flip": "aligned verdict 'beats' and independent verdict not 'beats'",
            "kendall_tau": "scipy.stats.kendalltau (tau-b) between the item scores under the aligned and the independent KG; BM25 and dense-only are identical in both arms",
            "answer_metric": "alias-aware EM; GraphRAG-Bench: Mistral-Small-24B judge accuracy reconstructed per question from results/e2e_qa/judge_cache_mistral_small_24b.jsonl",
        },
        "sources": {
            "per_question_retrieval": "external/<bench>/<split>/_kgswap_{kg,ppr_kg,proposed_v5_graphcentric_kg,hipporag2_kg}_{silver,openie}.json",
            "frozen_summaries": "results/kg_swap_<bench>{,_ppr,_proposed_v5_graphcentric}.json, results/hipporag2_swap/summary.json",
            "dense_only_retrieval": "results/kg_swap_dense_bge_m3_cls.json (BGE-M3, [CLS] pooling)",
            "per_question_answers": "results/e2e_qa/<reader>/per_question.jsonl",
            "dense_only_answers": "results/e2e_qa_cls/<reader>/{summary,judge_mistral_small_24b}.json",
            "table3": "tables/tab_e2e_merged_v3.tex (manuscript/eacl2027_camera_ready.tex Table 3)",
        },
        "retrieval": {
            "cells": ret,
            "counts": {"uncorrected": flip_counts(ret, "verdict"), "holm": flip_counts(ret, "verdict_holm")},
            "kendall_tau": retrieval_tau(ret, dense_r),
        },
        "realistic": {"cells": real, "counts": realistic_counts(real),
                      "structures": {b: lab for b, (_, lab) in REALISTIC.items()}},
        "answer": {},
        "figure": {"pdf": "figures/fig_headline_margins.pdf", "png": "figures/fig_headline_margins.png",
                   "encoding": "first split large markers, held-out split small markers below; circle = aligned KG (vermillion), "
                               "square = independent KG (bluish green), diamond = realistic structure not induced from the questions (blue; live Wikidata KG on 2WikiMultiHop, full KB on MetaQA); filled = graph significantly above BM25 (sign test p<0.05), "
                               "hollow crossed by an x = graph significantly below BM25 (p<0.05), hollow = neither"},
    }
    for tag in READERS:
        recs = ans[tag]
        summary["answer"][tag] = {"reader": READERS[tag], "cells": recs,
                                  "counts": {"uncorrected": flip_counts(recs, "verdict"), "holm": flip_counts(recs, "verdict_holm")},
                                  "kendall_tau": answer_tau(recs, dense_only_answer(tag))}
    rc = summary["retrieval"]["counts"]["uncorrected"]
    rh = summary["retrieval"]["counts"]["holm"]
    summary["caption_draft"] = (
        "Graph-over-BM25 hit@5 margin of each graph retriever with the gold-aligned KG (circles) and with an independent OpenIE KG "
        "(squares), on the first split (large) and the held-out split (small); filled markers beat BM25 and crossed markers fall "
        f"below it by an exact paired sign test ($p<0.05$). With the aligned KG {rc['beats_aligned']} of {rc['n_cells']} "
        f"benchmark--split--retriever cells beat BM25; with the independent KG {rc['beats_independent']} do, and "
        f"{rc['flips_beats_to_not']} verdicts flip (Holm over the {rh['n_cells']} cells of an arm: {rh['beats_aligned']}, "
        f"{rh['beats_independent']}, and {rh['flips_beats_to_not']}).")
    summary["cross_checks"] = {"n": len(CHECKS), "n_ok": sum(c["ok"] for c in CHECKS), "checks": CHECKS}
    figure(ret, real)
    (OUT / "headline_margins_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (OUT / "headline_margins_summary.md").write_text(markdown(summary), encoding="utf-8")
    print(f"cross-checks {summary['cross_checks']['n_ok']}/{summary['cross_checks']['n']} ok")
    print("retrieval:", json.dumps({k: v for k, v in rc.items() if k != "flipped_cells"}))
    for tag in READERS:
        c = summary["answer"][tag]["counts"]["uncorrected"]
        print(tag, json.dumps({k: v for k, v in c.items() if k != "flipped_cells"}))
    print("wrote analysis/headline_margins_summary.{json,md}, figures/fig_headline_margins.{pdf,png}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
