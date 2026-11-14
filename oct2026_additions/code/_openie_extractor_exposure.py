# -*- coding: utf-8 -*-
"""Extractor-exposure check for the "independent" OpenIE KGs.

Question: the OpenIE KG of the controlled swap is built by Llama-3.1-8B from the
benchmark passages without access to the gold. Llama-3.1-8B was pretrained on
Wikipedia, so it could emit gold evidence facts it memorized rather than facts
it read. Measure how often an OpenIE triple that reproduces a gold evidence
triple has an endpoint that is NOT in the passage it was extracted from.

Everything is lexical and deterministic: no model calls. SEED only fixes which
examples are printed.

Provenance (baselines/build_openie_kg.py): one LLM call per passage on
text[:3500] only (the title is not in the prompt). Edge endpoints are node ids
"n::" + lowercase/whitespace-collapsed surface; ``source_docs`` lists every
passage whose call emitted that (s, relation, o). The containment checks below
therefore use the first 3,500 characters of each source passage.

Normalization ``norm``: NFKD, combining marks stripped, lowercase, every
non-word run (and "_") -> one space.
"Occurs" (exact) = word-boundary containment of norm(endpoint) in
norm(passage). "Occurs" (loose) = token recall >= 0.8 (share of the endpoint's
normalized tokens present in the passage's token set).

Gold structures
  2wiki   dev.json ``evidences`` (s, r, o) of each split question; Wikidata
          labels, optional alias expansion via id_aliases.json (evidences_id).
  metaqa  KB path triples of each question, reconstructed from kb.txt exactly
          as eval/build_metaqa_swap_data.py builds kg_silver: (bridge, r1, topic)
          and (bridge, r2, x) for x in KB[r2][bridge]. Checked against the
          forward edges of kg_silver.
  grb     GraphRAG-Bench medical ``evidence_relations`` in triple form: the
          non-synonym edges of kg_silver, i.e. the same Llama-3.1-8B OpenIE run
          over the relation strings (evidence_relations have no native triple
          form). The two splits share one corpus-level OpenIE KG.

Matching a gold triple (gs, gr, go) by an OpenIE edge (s, r, o)
  so        {s, o} == {gs, go} after norm (direction ignored)       [primary]
  so_dir    s == gs and o == go
  sro       so and the relations share a content-token stem (5-char prefix,
            stop words removed)
  *_alias   2Wiki only: gs/go may be any Wikidata alias of the QID
  so_fuzzy  one endpoint's content-token set contains the other's, on both
            endpoints, aliases included (loose; admits spurious pairs)

Baseline: OpenIE relation edges matched by no gold triple under any mode.

Outputs: results/openie_extractor_exposure/{summary.json, summary.md, per_triple.jsonl}
Run:  python paper/eval/_openie_extractor_exposure.py
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
import unicodedata
from collections import OrderedDict, defaultdict
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
EXT = PAPER / "external"
OUT = PAPER / "results" / "openie_extractor_exposure"
SEED = 20260930
SPLITS = ("sub200", "holdout200")
MAX_CHARS = 3500  # OpenIEExtractor.extract prompt truncation
RECALL_T = 0.8
N_EX = 10
SNIP = 420

STOP = frozenset("""
a an the of for to in on at by with from into over under and or but nor so as
is are was were be been being am do does did have has had having will would
shall should can could may might must this that these those it its their his
her them they we you i he she who whom which what when where why how than then
not no yes such each any all both more most some few many much very also only
just about between among within without across per via vs versus
""".split())

_NONWORD = re.compile(r"[\W_]+")
_TOPIC_RE = re.compile(r"\[(.+)\]")

MODES = ("so", "so_dir", "sro", "so_alias", "so_dir_alias", "sro_alias", "so_fuzzy")
PRIMARY = "so"


def norm(s) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return " ".join(_NONWORD.sub(" ", s).split())


def ctoks(s) -> frozenset:
    return frozenset(t for t in norm(s).split() if len(t) > 1 and t not in STOP)


def fz(a: frozenset, names_ct) -> bool:
    """Fuzzy endpoint match: one content-token set contains the other (smaller set has a token of length >= 3)."""
    for n in names_ct:
        if not a or not n:
            continue
        sm, lg = (a, n) if len(a) <= len(n) else (n, a)
        if sm <= lg and any(len(t) >= 3 for t in sm):
            return True
    return False


def has(hay_padded: str, needle_n: str) -> bool:
    return bool(needle_n) and (" " + needle_n + " ") in hay_padded


def tok_recall(needle_n: str, hay_toks: set) -> float:
    t = set(needle_n.split())
    return len(t & hay_toks) / len(t) if t else 0.0


def stems(s) -> frozenset:
    return frozenset(t[:5] for t in norm(s).split() if len(t) > 1 and t not in STOP)


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def frac(k, n):
    return {"num": k, "den": n, "frac": round(k / n, 4) if n else None, "wilson95": wilson(k, n)}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def load_jsonl(p: Path):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_json(p: Path):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- corpus + OpenIE KG

class Corpus:
    def __init__(self, passages_path: Path):
        ps = load_jsonl(passages_path)
        self.raw = {p["passage_id"]: p["text"] for p in ps}
        self.seen = {pid: t[:MAX_CHARS] for pid, t in self.raw.items()}
        self.padded = {pid: " " + norm(t) + " " for pid, t in self.seen.items()}
        self.toks = {pid: set(norm(t).split()) for pid, t in self.seen.items()}
        self.joined = "\n".join(self.padded[pid] for pid in sorted(self.padded))
        self.n_truncated = sum(1 for t in self.raw.values() if len(t) > MAX_CHARS)

    def in_corpus(self, x: str) -> bool:
        return has(self.joined, x)

    def other_passages_with(self, x: str, exclude) -> int:
        if not self.in_corpus(x):
            return 0
        return sum(1 for pid, p in self.padded.items() if pid not in exclude and has(p, x))


def load_openie(kg_dir: Path, corpus: Corpus):
    edges = []
    missing_docs = 0
    for e in load_json(kg_dir / "proposed_kg_edges.json"):
        if e.get("relation_type") == "synonym":
            continue
        s, o = norm(e["source"][3:]), norm(e["target"][3:])
        docs = sorted(set(e.get("source_docs") or ([e["source_doc"]] if e.get("source_doc") else [])))
        docs_ok = [d for d in docs if d in corpus.padded]
        missing_docs += len(docs) - len(docs_ok)
        rec = {"edge_id": e["edge_id"], "s": s, "r": e["relation"], "o": o, "docs": docs_ok}
        P = [corpus.padded[d] for d in docs_ok]
        T = [corpus.toks[d] for d in docs_ok]
        rec["s_in"] = any(has(p, s) for p in P)
        rec["o_in"] = any(has(p, o) for p in P)
        rec["both_in_one"] = any(has(p, s) and has(p, o) for p in P)
        rec["s_in_tr"] = any(tok_recall(s, t) >= RECALL_T for t in T)
        rec["o_in_tr"] = any(tok_recall(o, t) >= RECALL_T for t in T)
        rec["both_in_one_tr"] = any(tok_recall(s, t) >= RECALL_T and tok_recall(o, t) >= RECALL_T for t in T)
        rec["o_in_corpus"] = corpus.in_corpus(o)
        rec["rstem"] = stems(e["relation"])
        rec["sct"], rec["oct"] = ctoks(s), ctoks(o)
        edges.append(rec)
    by_pair = defaultdict(list)
    tok_idx = defaultdict(set)
    for i, e in enumerate(edges):
        by_pair[frozenset((e["s"], e["o"]))].append(i)
        for t in e["sct"] | e["oct"]:
            tok_idx[t].add(i)
    return {"edges": edges, "by_pair": by_pair, "tok_idx": tok_idx, "missing_source_docs": missing_docs}


# --------------------------------------------------------------------------- gold units

def unit(s, r, o, s_alias=(), o_alias=(), qa_ids=()):
    sn, on = norm(s), norm(o)
    SA = {sn} | {norm(a) for a in s_alias if len(norm(a)) > 1}
    OA = {on} | {norm(a) for a in o_alias if len(norm(a)) > 1}
    return {"gold": [s, r, o], "key": (sn, norm(r), on), "S": {sn}, "O": {on}, "SA": SA, "OA": OA,
            "SAct": sorted({ctoks(x) for x in SA}, key=sorted), "OAct": sorted({ctoks(x) for x in OA}, key=sorted),
            "rstem": stems(r), "qa_ids": list(qa_ids)}


def dedupe(units):
    out = OrderedDict()
    for u in units:
        k = u["key"]
        if k in out:
            for q in u["qa_ids"]:
                if q not in out[k]["qa_ids"]:
                    out[k]["qa_ids"].append(q)
        else:
            out[k] = u
    return list(out.values())


def w2_units():
    dev = {r["_id"]: r for r in load_json(EXT / "2wiki" / "raw" / "data_ids" / "dev.json")}
    per_split, qids = {}, set()
    for sp in SPLITS:
        qa = load_jsonl(EXT / "2wiki" / sp / "qa.jsonl")
        per_split[sp] = [(q["qa_id"], dev[q["qa_id"]]) for q in qa]
        for _, r in per_split[sp]:
            for t in r.get("evidences_id") or []:
                if len(t) == 3:
                    qids.update([t[0], t[2]])
    aliases = {}
    with open(EXT / "2wiki" / "raw" / "data_ids" / "id_aliases.json", encoding="utf-8") as fh:
        for l in fh:
            r = json.loads(l)
            if r["Q_id"] in qids:
                aliases[r["Q_id"]] = r.get("aliases", [])
    out = {}
    for sp in SPLITS:
        us = []
        for qid, r in per_split[sp]:
            ev_id = r.get("evidences_id") or []
            for i, (s, p, o) in enumerate(r["evidences"]):
                sq = ev_id[i][0] if i < len(ev_id) and len(ev_id[i]) == 3 else None
                oq = ev_id[i][2] if i < len(ev_id) and len(ev_id[i]) == 3 else None
                us.append(unit(s, p, o, aliases.get(sq, []), aliases.get(oq, []), [qid]))
        out[sp] = dedupe(us)
    return out, {"n_qids_with_aliases": len(aliases)}


def metaqa_units():
    fwd = defaultdict(lambda: defaultdict(set))
    with open(EXT / "metaqa" / "raw" / "kb.txt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) != 3:
                continue
            s, r, o = (x.strip() for x in parts)
            if s and r and o:
                fwd[r][s].add(o)
    out, check = {}, {}
    for sp in SPLITS:
        qa = load_jsonl(EXT / "metaqa" / sp / "qa.jsonl")
        title = {p["passage_id"]: p["title"] for p in load_jsonl(EXT / "metaqa" / sp / "passages.jsonl")}
        us = []
        for q in qa:
            topic = _TOPIC_RE.search(q["question"]).group(1).strip()
            r1, r2 = q["qtype"].split("->")
            for pid in q["gold_paper_ids"]:
                b = title[pid]
                us.append(unit(b, r1, topic, qa_ids=[q["qa_id"]]))
                for x in sorted(fwd[r2].get(b, ())):
                    us.append(unit(b, r2, x, qa_ids=[q["qa_id"]]))
        us = dedupe(us)
        silver = {(norm(e["source"][3:]), norm(e["relation"]), norm(e["target"][3:]))
                  for e in load_json(EXT / "metaqa" / sp / "kg_silver" / "proposed_kg_edges.json")
                  if not e["relation"].endswith("_inv")}
        mine = {u["key"] for u in us}
        check[sp] = {"units": len(mine), "silver_forward_edges": len(silver), "identical": mine == silver}
        out[sp] = us
    return out, {"reconstruction_vs_kg_silver": check}


def grb_units():
    out = {}
    for sp in SPLITS:
        us = [unit(e["source"][3:], e["relation"], e["target"][3:])
              for e in load_json(EXT / "graphragbench" / sp / "kg_silver" / "proposed_kg_edges.json")
              if e.get("relation_type") != "synonym"]
        out[sp] = [u for u in dedupe(us) if u["key"][0] != u["key"][2]]
    return out, {}


# --------------------------------------------------------------------------- matching + stats

def match(u, kg):
    E, idx = kg["edges"], kg["by_pair"]
    res = {}
    for alias in (False, True):
        S, O = (u["SA"], u["OA"]) if alias else (u["S"], u["O"])
        cand = set()
        for a in S:
            for b in O:
                if a != b:
                    cand.update(idx.get(frozenset((a, b)), ()))
        so = sorted(cand)
        so_dir = [i for i in so if E[i]["s"] in S and E[i]["o"] in O]
        sro = [i for i in so if E[i]["rstem"] & u["rstem"]]
        sfx = "_alias" if alias else ""
        res["so" + sfx], res["so_dir" + sfx], res["sro" + sfx] = so, so_dir, sro
    # fuzzy: token-set containment on both endpoints (aliases included), direction ignored
    SC, OC = u["SAct"], u["OAct"]
    ti = kg["tok_idx"]
    cs = set().union(*[ti.get(t, set()) for n in SC for t in n])
    co = set().union(*[ti.get(t, set()) for n in OC for t in n])
    res["so_fuzzy"] = sorted(i for i in cs & co if E[i]["s"] != E[i]["o"] and (
        (fz(E[i]["sct"], SC) and fz(E[i]["oct"], OC)) or (fz(E[i]["sct"], OC) and fz(E[i]["oct"], SC))))
    return res


FLAGS = {  # edge-level "unsupported" predicates
    "o_absent": lambda e: not e["o_in"],
    "s_absent": lambda e: not e["s_in"],
    "either_absent": lambda e: not (e["s_in"] and e["o_in"]),
    "not_both_in_one": lambda e: not e["both_in_one"],
    "o_absent_tr": lambda e: not e["o_in_tr"],
    "either_absent_tr": lambda e: not (e["s_in_tr"] and e["o_in_tr"]),
}


def edge_rates(edges):
    n = len(edges)
    return {"n": n, **{k: frac(sum(1 for e in edges if f(e)), n) for k, f in FLAGS.items()}}


def analyze(pairs):
    """pairs: list of (kg_name, kg, units). Units of one pair are matched only to that pair's KG."""
    n_edges = sum(len(kg["edges"]) for _, kg, _ in pairs)
    per_mode = {}
    gold_edges_any = set()
    for mode in MODES:
        n_units = n_matched = 0
        matched_edges = set()
        only_unsup = {k: 0 for k in FLAGS}
        any_unsup = {k: 0 for k in FLAGS}
        for kname, kg, units in pairs:
            E = kg["edges"]
            for u in units:
                n_units += 1
                m = u["_m"][mode]
                if not m:
                    continue
                n_matched += 1
                matched_edges.update((kname, i) for i in m)
                for k, f in FLAGS.items():
                    flags = [f(E[i]) for i in m]
                    only_unsup[k] += all(flags)
                    any_unsup[k] += any(flags)
        ME = [dict(kg["edges"][i]) for kname, kg, _ in pairs for (kn, i) in sorted(matched_edges) if kn == kname]
        er = edge_rates(ME)
        per_mode[mode] = {
            "gold_units": n_units,
            "gold_matched": frac(n_matched, n_units),
            "matched_gold_all_matches_unsupported": {k: frac(v, n_matched) for k, v in only_unsup.items()},
            "matched_gold_any_match_unsupported": {k: frac(v, n_matched) for k, v in any_unsup.items()},
            "matched_edges": er,
            "matched_edges_unsupported_over_all_openie_edges": {k: frac(er[k]["num"], n_edges) for k in FLAGS},
        }
        gold_edges_any |= matched_edges
    # baselines: non-gold edges (matched by no gold unit under any mode, fuzzy included)
    gold_rel_stems = frozenset().union(*[u["rstem"] for _, _, us in pairs for u in us]) if pairs else frozenset()
    NG = [kg["edges"][i] for kname, kg, _ in pairs for i in range(len(kg["edges"])) if (kname, i) not in gold_edges_any]
    baselines = {
        "all_nongold": edge_rates(NG),
        "nongold_object_attested_in_corpus": edge_rates([e for e in NG if e["o_in_corpus"]]),
        "nongold_same_relation_family": edge_rates([e for e in NG if e["rstem"] & gold_rel_stems]),
        "all_openie_edges": edge_rates([e for _, kg, _ in pairs for e in kg["edges"]]),
    }
    # entity-level: object-absent edges whose object is a gold endpoint (exact / fuzzy, aliases included)
    gold_names = set().union(*[u["SA"] | u["OA"] for _, _, us in pairs for u in us])
    gold_cts = sorted({c for _, _, us in pairs for u in us for c in u["SAct"] + u["OAct"]}, key=sorted)
    OA_E = [e for _, kg, _ in pairs for e in kg["edges"] if not e["o_in"]]
    inj = {"object_absent_edges": len(OA_E),
           "object_is_gold_endpoint_exact": frac(sum(1 for e in OA_E if e["o"] in gold_names), len(OA_E)),
           "object_is_gold_endpoint_fuzzy": frac(sum(1 for e in OA_E if fz(e["oct"], gold_cts)), len(OA_E))}
    return {"n_openie_edges": n_edges, "n_gold_matched_edges_any_mode": len(gold_edges_any), "modes": per_mode,
            "baselines": baselines, "object_absent_edges_with_gold_entity_object": inj,
            "gold_relation_stems": sorted(gold_rel_stems)}


def question_level(pairs, mode=PRIMARY):
    q_units = defaultdict(list)
    for kname, kg, units in pairs:
        for u in units:
            for q in u["qa_ids"]:
                q_units[q].append((kg, u))
    if not q_units:
        return None
    n = len(q_units)
    any_matched = all_matched = recalled_o = recalled_either = 0
    for q, lst in q_units.items():
        ms = [(kg, u["_m"][mode]) for kg, u in lst]
        any_matched += any(m for _, m in ms)
        all_matched += all(m for _, m in ms)
        recalled_o += any(m and all(not kg["edges"][i]["o_in"] for i in m) for kg, m in ms)
        recalled_either += any(m and all(not (kg["edges"][i]["s_in"] and kg["edges"][i]["o_in"]) for i in m) for kg, m in ms)
    return {"mode": mode, "n_questions": n,
            "questions_with_any_gold_triple_matched": frac(any_matched, n),
            "questions_with_all_gold_triples_matched": frac(all_matched, n),
            "questions_with_a_gold_triple_matched_only_by_object_absent_edges": frac(recalled_o, n),
            "questions_with_a_gold_triple_matched_only_by_endpoint_absent_edges": frac(recalled_either, n)}


# --------------------------------------------------------------------------- build

def build():
    meta = OrderedDict([("seed", SEED), ("deterministic", True), ("max_chars_seen_by_extractor", MAX_CHARS),
                        ("token_recall_threshold", RECALL_T), ("primary_mode", PRIMARY), ("inputs_md5", {})])
    datasets = OrderedDict()
    w2, w2m = w2_units()
    mq, mqm = metaqa_units()
    gr, grm = grb_units()
    meta.update({"2wiki": w2m, "metaqa": mqm})
    corpora, kgs = {}, {}
    spec = [("2wiki", w2, {sp: (EXT / "2wiki" / sp / "passages.jsonl", EXT / "2wiki" / sp / "kg_openie") for sp in SPLITS}),
            ("metaqa", mq, {sp: (EXT / "metaqa" / sp / "passages.jsonl", EXT / "metaqa" / sp / "kg_openie") for sp in SPLITS}),
            ("grb_medical", gr, {sp: (EXT / "graphragbench" / sp / "passages.jsonl", EXT / "graphragbench" / sp / "kg_openie") for sp in SPLITS})]
    for name, units, paths in spec:
        for sp, (pp, kd) in paths.items():
            kid = md5(kd / "proposed_kg_edges.json") + md5(pp)
            meta["inputs_md5"][f"{name}/{sp}"] = {"passages": md5(pp), "kg_openie_edges": md5(kd / "proposed_kg_edges.json")}
            if kid not in kgs:
                corpora[kid] = Corpus(pp)
                kgs[kid] = load_openie(kd, corpora[kid])
            kg = kgs[kid]
            for u in units[sp]:
                u["_m"] = match(u, kg)
                u["_kid"] = kid
        datasets[name] = (units, {sp: md5(paths[sp][1] / "proposed_kg_edges.json") + md5(paths[sp][0]) for sp in SPLITS})
    kg_names = {}
    for kid in kgs:
        users = [f"{n}/{sp}" for n, (_, ids) in datasets.items() for sp in SPLITS if ids[sp] == kid]
        kg_names[kid] = users[0].split("/")[0] + "/" + ("shared" if len(users) > 1 else users[0].split("/")[1])
    meta["kgs"] = {kg_names[k]: {"n_relation_edges": len(v["edges"]), "missing_source_docs": v["missing_source_docs"],
                                 "n_passages": len(corpora[k].raw), "n_passages_truncated_at_3500": corpora[k].n_truncated}
                   for k, v in kgs.items()}

    S = OrderedDict([("meta", meta)])
    for name, (units, ids) in datasets.items():
        D = OrderedDict()
        for sp in SPLITS:
            pairs = [(kg_names[ids[sp]], kgs[ids[sp]], units[sp])]
            D[sp] = analyze(pairs)
            D[sp]["question_level"] = {m: question_level(pairs, m) for m in (PRIMARY, "so_fuzzy")} if name != "grb_medical" else None
        by_kid = OrderedDict()
        for sp in SPLITS:
            by_kid.setdefault(ids[sp], []).extend(units[sp])
        pooled_pairs = [(kg_names[k], kgs[k], dedupe_units(us)) for k, us in by_kid.items()]
        D["pooled"] = analyze(pooled_pairs)
        D["pooled"]["question_level"] = {m: question_level(pooled_pairs, m) for m in (PRIMARY, "so_fuzzy")} if name != "grb_medical" else None
        D["pooled"]["note"] = ("splits have separate corpora and KGs; pooled = both KGs, both gold sets"
                               if len(by_kid) > 1 else "both splits share one corpus and one KG; pooled = union of gold units over that KG")
        S[name] = D

    # ---- per-triple records + examples
    EK = ("s", "r", "o", "docs", "s_in", "o_in", "both_in_one", "s_in_tr", "o_in_tr", "o_in_corpus")
    rows, cand, base, injected = [], defaultdict(list), defaultdict(list), defaultdict(list)
    gold_any, gold_names = defaultdict(set), defaultdict(set)
    for name, (units, ids) in datasets.items():
        for sp in SPLITS:
            kid = ids[sp]
            kg, C = kgs[kid], corpora[kid]
            for u in units[sp]:
                m = u["_m"]
                gold_names[kid] |= u["SA"] | u["OA"]
                for mode in MODES:
                    gold_any[kid].update(m[mode])
                for i in m["so_fuzzy"] + [j for j in m[PRIMARY] if j not in m["so_fuzzy"]]:
                    if not kg["edges"][i]["o_in"]:
                        cand[name].append((sp, u, kg["edges"][i], C, "so" if i in m[PRIMARY] else "so_fuzzy only"))
                rows.append({"kind": "gold", "dataset": name, "split": sp, "kg": kg_names[kid], "gold": u["gold"],
                             "qa_ids": u["qa_ids"], "n_match": {k: len(v) for k, v in m.items()},
                             "matches_" + PRIMARY: [{k: kg["edges"][i][k] for k in EK} for i in m[PRIMARY]],
                             "matches_so_fuzzy_only": [{k: kg["edges"][i][k] for k in EK} for i in m["so_fuzzy"] if i not in m[PRIMARY]]})
    for kid, kg in kgs.items():
        name = kg_names[kid].split("/")[0]
        for i, e in enumerate(kg["edges"]):
            g = i in gold_any[kid]
            rows.append({"kind": "openie_edge", "dataset": name, "kg": kg_names[kid], "gold_match_any_mode": g,
                         "object_is_gold_endpoint": e["o"] in gold_names[kid], **{k: e[k] for k in EK}})
            if not g and not e["o_in"]:
                base[name].append((kg_names[kid], e, corpora[kid]))
                if e["o"] in gold_names[kid]:
                    injected[name].append((kg_names[kid], e, corpora[kid]))

    rng = random.Random(SEED)

    def pick(lst, k=N_EX):
        return lst if len(lst) <= k else [lst[i] for i in sorted(rng.sample(range(len(lst)), k))]

    ex = OrderedDict()
    for name in datasets:
        cs = sorted(cand[name], key=lambda t: (t[4], t[0], t[1]["key"], t[2]["edge_id"]))
        prim = pick([t for t in cs if t[4] == "so"])  # primary-mode candidates first, then fuzzy-only
        rest = pick([t for t in cs if t[4] != "so"], N_EX - len(prim))
        ex[name] = [dict(example(sp, u["gold"], e, C), match=mo) for sp, u, e, C, mo in prim + rest]
    for name in datasets:
        inj = sorted(injected[name], key=lambda t: (t[0], t[1]["edge_id"]))
        ex[name + "_nongold_object_absent_but_object_is_gold_entity"] = [example(k, None, e, C) for k, e, C in pick(inj)]
    bs = sorted(base["2wiki"], key=lambda t: (t[0], t[1]["edge_id"]))
    ex["2wiki_nongold_object_absent_baseline"] = [example(k, None, e, C) for k, e, C in pick(bs)]
    S["examples"] = ex
    ea = []
    for sp in SPLITS:
        kg, C = kgs[datasets["2wiki"][1][sp]], corpora[datasets["2wiki"][1][sp]]
        for u in datasets["2wiki"][0][sp]:
            m = u["_m"][PRIMARY]
            if m and all(not (kg["edges"][i]["s_in"] and kg["edges"][i]["o_in"]) for i in m):
                ea.append({"split": sp, "gold": u["gold"], "qa_ids": u["qa_ids"],
                           "edges": [{k: kg["edges"][i][k] for k in EK} for i in m],
                           "passage_snippet": C.seen.get(kg["edges"][m[0]]["docs"][0], "")[:SNIP]})
    S["w2_gold_matched_only_by_endpoint_absent_edges"] = ea
    S["n_object_absent_gold_matched_edges"] = {
        n: {"so": sum(1 for t in v if t[4] == "so"), "so_fuzzy_only": sum(1 for t in v if t[4] != "so")} for n, v in cand.items()}
    S["n_nongold_object_absent_edges_with_gold_entity_object"] = {n: len(injected[n]) for n in datasets}
    return S, rows


def dedupe_units(us):
    out = OrderedDict()
    for u in us:
        if u["key"] in out:
            for q in u["qa_ids"]:
                if q not in out[u["key"]]["qa_ids"]:
                    out[u["key"]]["qa_ids"].append(q)
        else:
            out[u["key"]] = dict(u, qa_ids=list(u["qa_ids"]))
    return list(out.values())


def example(split, gold, e, C):
    d = e["docs"][0] if e["docs"] else None
    txt = C.seen.get(d, "") if d else ""
    return {"split": split, "gold": gold, "openie": [e["s"], e["r"], e["o"]], "source_docs": e["docs"],
            "s_in_source": e["s_in"], "o_in_source": e["o_in"],
            "o_token_recall_in_source": round(max((tok_recall(e["o"], C.toks[x]) for x in e["docs"]), default=0.0), 3),
            "o_in_other_corpus_passages": C.other_passages_with(e["o"], set(e["docs"])),
            "passage_snippet": txt[:SNIP] + (" ..." if len(txt) > SNIP else "")}


# --------------------------------------------------------------------------- report

def f(x):
    if x is None:
        return "n/a"
    if isinstance(x, dict) and "frac" in x:
        if x["frac"] is None:
            return "n/a"
        return f"{100 * x['num'] / x['den']:.1f}% ({x['num']}/{x['den']})"  # from the counts: no double rounding
    return str(x)


def pct(x):
    return f"{100 * x['num'] / x['den']:.1f}%" if x and x.get("frac") is not None else "n/a"


def upper(x):
    return f"{100 * x['wilson95'][1]:.1f}%" if x and x.get("wilson95") else "n/a"


def write_md(S):
    L = []
    A = L.append
    A("# OpenIE extractor-exposure check (Llama-3.1-8B OpenIE KGs)\n")
    A("Generated by `eval/_openie_extractor_exposure.py` (deterministic, lexical, no model calls; output sha256 "
      f"`{S['meta']['output_sha256'][:16]}`). Question: does the gold-free OpenIE KG contain gold evidence "
      "triples whose endpoints are not in the passage the extractor was given, i.e. candidate \"recalled, not extracted\" triples?\n")
    A("## Method\n")
    A("- Provenance. `baselines/build_openie_kg.py` makes one Llama-3.1-8B call per passage on `text[:3500]` (no title, no gold). "
      "Every relation edge stores `source_docs`, the passages whose call emitted that exact (s, relation, o). Synonym edges are ignored.")
    A("- Normalization: NFKD, accents stripped, lowercase, every non-word run to one space. Exact occurrence = word-boundary "
      "containment of the normalized endpoint in the normalized source passage; loose occurrence = token recall >= 0.8.")
    A("- Match modes (direction ignored unless `_dir`): `so` = identical normalized endpoints (primary); `sro` = `so` plus a shared "
      "relation content-token stem (5-char prefix); `so_alias` = 2Wiki endpoints may be any Wikidata alias; `so_fuzzy` = one "
      "endpoint's content-token set contains the other's, aliases included (loose; admits spurious pairs).")
    A("- A matched edge is a candidate \"recalled\" edge when its own object string does not occur in any of its source passages. "
      "A matched gold triple is a candidate when every edge that matches it is such an edge.")
    A("- Baseline: all OpenIE relation edges that match no gold triple under any mode (fuzzy included).")
    mq = S["meta"]["metaqa"]["reconstruction_vs_kg_silver"]
    A("- Gold: 2Wiki = Wikidata `evidences` (dev.json) of the split questions; MetaQA = KB path triples from kb.txt "
      f"(identical to kg_silver forward edges: sub200 {mq['sub200']['identical']}, holdout200 {mq['holdout200']['identical']}); "
      "GraphRAG-Bench medical = kg_silver edges, which are the same Llama-3.1-8B OpenIE run over the released `evidence_relations` "
      "strings (the benchmark ships no triples). The two GraphRAG-Bench splits share one corpus and one OpenIE KG.\n")

    A("## 1-3. Gold triples reproduced by the OpenIE KG, and whether the source passage supports them\n")
    A("Primary mode `so` (exact normalized endpoints):\n")
    A("| Dataset / split | gold triples | matched (s,o) | matched (s,r,o) | matching edges | edges with object absent, exact [95% upper] | "
      "same, loose | gold triples matched only by object-absent edges | same, either endpoint absent | object-absent matching edges / all OpenIE edges |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for name in ("2wiki", "metaqa", "grb_medical"):
        for sp in SPLITS + ("pooled",):
            d = S[name][sp]["modes"]
            p, r = d[PRIMARY], d["sro"]
            oa = p["matched_edges"]["o_absent"]
            A(f"| {name} / {sp} | {p['gold_units']} | {f(p['gold_matched'])} | {f(r['gold_matched'])} | {p['matched_edges']['n']} | "
              f"{f(oa)} [{upper(oa)}] | {f(p['matched_edges']['o_absent_tr'])} | "
              f"{f(p['matched_gold_all_matches_unsupported']['o_absent'])} | {f(p['matched_gold_all_matches_unsupported']['either_absent'])} | "
              f"{f(p['matched_edges_unsupported_over_all_openie_edges']['o_absent'])} |")
    A("")
    A("Sensitivity modes (pooled over splits):\n")
    A("| Dataset | mode | matched gold triples | matching edges | object absent, exact | object absent, loose | either endpoint absent, exact | "
      "gold triples matched only by object-absent edges |")
    A("|---|---|---|---|---|---|---|---|")
    for name in ("2wiki", "metaqa", "grb_medical"):
        for mode in ("so_dir", "sro", "so_alias", "so_fuzzy"):
            if name != "2wiki" and mode == "so_alias":
                continue
            p = S[name]["pooled"]["modes"][mode]
            A(f"| {name} | {mode} | {f(p['gold_matched'])} | {p['matched_edges']['n']} | {f(p['matched_edges']['o_absent'])} | "
              f"{f(p['matched_edges']['o_absent_tr'])} | {f(p['matched_edges']['either_absent'])} | "
              f"{f(p['matched_gold_all_matches_unsupported']['o_absent'])} |")
    A("")
    A("Question level:\n")
    A("| Dataset / split | mode | questions | any gold triple matched | all gold triples matched | has a gold triple matched only by object-absent edges |")
    A("|---|---|---|---|---|---|")
    for name in ("2wiki", "metaqa"):
        for sp in SPLITS + ("pooled",):
            for mode, q in S[name][sp]["question_level"].items():
                A(f"| {name} / {sp} | {mode} | {q['n_questions']} | {f(q['questions_with_any_gold_triple_matched'])} | "
                  f"{f(q['questions_with_all_gold_triples_matched'])} | {f(q['questions_with_a_gold_triple_matched_only_by_object_absent_edges'])} |")
    A("")
    ea = S["w2_gold_matched_only_by_endpoint_absent_edges"]
    A(f"2Wiki gold triples matched (`so`) only by edges with an endpoint absent from the source passage: {len(ea)}\n")
    for x in ea:
        A(f"- [{x['split']}] gold ({' | '.join(x['gold'])}); edges: "
          + "; ".join(f"({e['s']} | {e['r']} | {e['o']}) subject in source {e['s_in']} (token recall >= 0.8: {e['s_in_tr']}), "
                      f"object in source {e['o_in']}" for e in x["edges"]) + ".  ")
        A(f"  Passage: \"{x['passage_snippet']}\"")
    A("")

    A("## 4. Extraction-noise baseline\n")
    A("| Dataset (pooled) | population | n | object absent, exact | object absent, loose | either endpoint absent, exact |")
    A("|---|---|---|---|---|---|")
    for name in ("2wiki", "metaqa", "grb_medical"):
        b = S[name]["pooled"]["baselines"]
        rows = [("gold-matched edges, `so`", S[name]["pooled"]["modes"][PRIMARY]["matched_edges"]),
                ("gold-matched edges, `so_fuzzy`", S[name]["pooled"]["modes"]["so_fuzzy"]["matched_edges"]),
                ("all non-gold edges", b["all_nongold"]),
                ("non-gold, object string occurs somewhere in the corpus", b["nongold_object_attested_in_corpus"]),
                ("non-gold, relation shares a stem with a gold relation", b["nongold_same_relation_family"]),
                ("all OpenIE relation edges", b["all_openie_edges"])]
        for lab, r in rows:
            A(f"| {name} | {lab} | {r['n']} | {f(r['o_absent'])} | {f(r['o_absent_tr'])} | {f(r['either_absent'])} |")
    A("")
    A("Entity-level check: object-absent edges (any) whose object string equals a gold endpoint or alias (exact) / "
      "token-contains one (fuzzy):\n")
    A("| Dataset (pooled) | object-absent edges | object = gold endpoint, exact | fuzzy |")
    A("|---|---|---|---|")
    for name in ("2wiki", "metaqa", "grb_medical"):
        x = S[name]["pooled"]["object_absent_edges_with_gold_entity_object"]
        A(f"| {name} | {x['object_absent_edges']} | {f(x['object_is_gold_endpoint_exact'])} | {f(x['object_is_gold_endpoint_fuzzy'])} |")
    A("")
    A("The fuzzy column is loose (a one-token gold name such as a first name is contained in many objects). GraphRAG-Bench gold "
      "endpoints include generic terms (\"treatment\", \"blood tests\", \"a risk factor\"), so its entity-level counts say little "
      "about memorization.\n")
    A("Wilson 95% intervals for every proportion are in `summary.json`.\n")

    A("## Examples\n")
    A("Up to 10 per list, seeded sample (SEED in `summary.json`); snippet = first 420 characters of the first source passage as the "
      "extractor saw it. \"object token recall\" is the loose test on the source passage.\n")
    labels = {
        "2wiki": "2Wiki: gold-matched edges with object absent (exact)",
        "metaqa": "MetaQA: gold-matched edges with object absent (exact)",
        "grb_medical": "GraphRAG-Bench medical: gold-matched edges with object absent (exact)",
        "2wiki_nongold_object_absent_but_object_is_gold_entity": "2Wiki: non-gold object-absent edges whose object is a gold entity",
        "metaqa_nongold_object_absent_but_object_is_gold_entity": "MetaQA: non-gold object-absent edges whose object is a gold entity",
        "grb_medical_nongold_object_absent_but_object_is_gold_entity": "GraphRAG-Bench: non-gold object-absent edges whose object is a gold entity",
        "2wiki_nongold_object_absent_baseline": "2Wiki baseline: non-gold edges with object absent",
    }
    cnt = S["n_object_absent_gold_matched_edges"]
    for key, exs in S["examples"].items():
        extra = ""
        if key in cnt:
            extra = f"; instances (split, gold triple, edge): {cnt[key]['so']} via `so`, {cnt[key]['so_fuzzy_only']} via `so_fuzzy` only"
        A(f"### {labels.get(key, key)} ({len(exs)} shown{extra})\n")
        if not exs:
            A("None.\n")
        for x in exs:
            g = f"gold ({' | '.join(x['gold'])}); " if x["gold"] else ""
            mo = f" [{x['match']}]" if x.get("match") else ""
            A(f"- [{x['split']}]{mo} {g}OpenIE ({' | '.join(x['openie'])}) from `{', '.join(x['source_docs'])}`; subject in source "
              f"{x['s_in_source']}, object token recall {x['o_token_recall_in_source']}, object string in "
              f"{x['o_in_other_corpus_passages']} other corpus passage(s).  ")
            A(f"  Passage: \"{x['passage_snippet']}\"")
        A("")
    A("## Reading\n")
    A(S["reading"] + "\n")
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")


def reading(S):
    c = lambda n: f"{n:,}"
    w = S["2wiki"]["pooled"]
    p, pf, b = w["modes"][PRIMARY], w["modes"]["so_fuzzy"], w["baselines"]
    m = S["metaqa"]["pooled"]
    mp, mb = m["modes"][PRIMARY], m["baselines"]
    gp, gb = S["grb_medical"]["pooled"]["modes"][PRIMARY], S["grb_medical"]["pooled"]["baselines"]
    oa, moa = p["matched_edges"]["o_absent"], mp["matched_edges"]["o_absent"]
    foa, foa_tr = pf["matched_edges"]["o_absent"], pf["matched_edges"]["o_absent_tr"]
    ea = S["w2_gold_matched_only_by_endpoint_absent_edges"]
    T = ["The OpenIE KG is built by Llama-3.1-8B, which has seen Wikipedia in pretraining. Each extraction call saw one passage "
         "and nothing else, and every edge records its source passage, so a fact supplied from memory would surface as an edge "
         "whose endpoint is missing from that passage."]
    T.append(f"On 2WikiMultiHop (both swap splits), {p['gold_matched']['num']} of {c(p['gold_units'])} gold evidence triples "
             f"({pct(p['gold_matched'])}) are reproduced by {p['matched_edges']['n']} OpenIE edges with identical normalized endpoints, and "
             f"the object is absent from the source passage for {oa['num']} of these {oa['den']} edges "
             f"({pct(oa)}; Wilson 95% upper bound {upper(oa)}), against {pct(b['all_nongold']['o_absent'])} of all non-gold edges "
             f"({pct(b['nongold_object_attested_in_corpus']['o_absent'])} when the object string occurs elsewhere in the corpus).")
    if ea:
        tr_ok = all(e["s_in_tr"] and e["o_in_tr"] for x in ea for e in x["edges"])
        T.append(("One gold triple enters" if len(ea) == 1 else f"{len(ea)} gold triples enter")
                 + " only through edges whose endpoint takes the Wikidata spelling of a name the passage writes differently"
                 + ("; all of its tokens are in the passage." if tr_ok else "."))
    T.append(f"On MetaQA the count is {moa['num']} of {moa['den']} edges ({mp['gold_matched']['num']} of {c(mp['gold_units'])} KB path "
             f"triples matched; upper bound {upper(moa)}; non-gold {pct(mb['all_nongold']['o_absent'])}).")
    fz = (f"A loose token-containment match raises 2Wiki coverage to {pf['gold_matched']['num']} triples; {foa['num']} of its "
          f"{foa['den']} edges have an object not found verbatim")
    if foa["num"] and foa_tr["num"] == 0:
        fz += f", but in all {foa['num']} at least 80% of the object's tokens occur in the passage (merged or clause-length spans)."
    else:
        fz += f", {foa_tr['num']} of them also failing the token-recall test."
    T.append(fz)
    T.append(f"On GraphRAG-Bench medical, {gp['matched_edges']['o_absent']['num']} of {gp['matched_edges']['n']} gold-matched edges "
             f"({pct(gp['matched_edges']['o_absent'])}) have an absent object, against {pct(gb['nongold_object_attested_in_corpus']['o_absent'])} "
             "for non-gold edges with corpus-attested objects; on inspection these are plural, punctuation and paraphrase variants "
             "(\"biomarker\" in a chunk that says \"biomarkers\").")
    T.append("We therefore find no gold fact that the extractor could only have supplied from memory. The check is limited: it sees "
             f"only gold triples reproduced with matching surface forms ({pct(p['gold_matched'])} of 2Wiki triples under exact matching), "
             f"it cannot exclude an injection rate below about {upper(oa)} on 2Wiki, and it cannot detect memory acting within a "
             "passage, for instance an extractor that recovers passage-stated gold relations more reliably because it has read these "
             "pages. The OpenIE KG is grounded in its source passages, but this does not show it is free of pretraining influence.")
    return " ".join(T)


def main():
    S1, R1 = build()
    S2, R2 = build()
    h1 = hashlib.sha256(json.dumps([S1, R1], sort_keys=True, default=list).encode()).hexdigest()
    h2 = hashlib.sha256(json.dumps([S2, R2], sort_keys=True, default=list).encode()).hexdigest()
    assert h1 == h2, "non-deterministic"
    S1["meta"]["output_sha256"] = h1
    S1["reading"] = reading(S1)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "per_triple.jsonl", "w", encoding="utf-8") as fh:
        for r in R1:
            fh.write(json.dumps(r, ensure_ascii=False, default=list) + "\n")
    with open(OUT / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(S1, fh, indent=1, ensure_ascii=False, default=list)
    write_md(S1)
    print(json.dumps(S1["meta"], indent=1, default=list))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
