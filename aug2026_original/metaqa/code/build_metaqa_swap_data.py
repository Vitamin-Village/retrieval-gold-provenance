# -*- coding: utf-8 -*-
"""MetaQA second-swap data builder (S1 subset + S2 silver KG).

Implements paper/eval/_metaqa_swap_design.md exactly:

S1  From qa_test_2hop.txt (dev appended only if the eligible pool is short),
    mechanically derive each question's KB path topic --r1--> bridge movies
    --r2--> answers. The relation pair per question TEMPLATE is inferred by
    exhaustive candidate scoring against the KB itself (no manual choice).
    Keep questions with EXACTLY 2 bridge movies, both with unambiguous
    wiki.txt docs, and an answer string present in at least one bridge doc.
    Stratified round-robin sample (seed 20260613) -> sub200 (+ disjoint
    holdout200 when available). Corpus = gold docs + deterministic
    distractors up to 1,250 passages per split. Output mirrors the 2Wiki
    sub200 schema byte-for-byte in structure.

S2  Silver KG per split: the KB subgraph induced by the selected questions'
    2-hop paths (path triples + reverses; movie node -> own doc, non-movie
    node -> KB-connected movie docs within the subgraph). Asserts gold-title
    coverage == 1.0 by construction.

Determinism: the whole build runs twice in memory and the script asserts the
two passes serialize identically before anything is written.

ASCII-only output. Run from repo root:
    python paper/eval/build_metaqa_swap_data.py
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "paper" / "external" / "metaqa" / "raw"
OUT_BASE = ROOT / "paper" / "external" / "metaqa"
PROGRESS = OUT_BASE / "_PROGRESS.log"

SEED = 20260613
N_PER_SPLIT = 200
CORPUS_SIZE = 1250
TEMPLATE_SAMPLE = 60          # instances scored per template for inference
TEMPLATE_ACCEPT_RATE = 0.9    # min exact answer-set match rate

_WS_RE = re.compile(r"\s+")
_LINE_NO_RE = re.compile(r"^\d+\s")
_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")
_TOPIC_RE = re.compile(r"\[(.+)\]")


def norm(s: str) -> str:
    return _WS_RE.sub(" ", s.strip()).lower()


def log_progress(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as fh:
        fh.write("%s %s\n" % (ts, msg))
    print("[progress] " + msg, flush=True)


# ---------------------------------------------------------------------------
# Raw parsers
# ---------------------------------------------------------------------------
def parse_kb(path: Path):
    """Return (fwd, inv, movies): fwd[rel][subj]=set(obj), inv[rel][obj]=set(subj)."""
    fwd: dict = defaultdict(lambda: defaultdict(set))
    inv: dict = defaultdict(lambda: defaultdict(set))
    movies: set = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                continue
            s, r, o = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if not s or not r or not o:
                continue
            fwd[r][s].add(o)
            inv[r][o].add(s)
            movies.add(s)
    return fwd, inv, movies


def parse_wiki(path: Path):
    """Return list of (title_raw, text). Docs separated by blank lines,
    line 1 = movie title, every line prefixed with its number."""
    docs = []
    title, body = None, []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                if title is not None:
                    docs.append((title, " ".join(body)))
                title, body = None, []
                continue
            stripped = _LINE_NO_RE.sub("", line, count=1)
            if title is None:
                title = stripped.strip()
            else:
                body.append(stripped.strip())
    if title is not None:
        docs.append((title, " ".join(body)))
    return docs


def parse_qa(path: Path, source: str):
    """Return list of dicts {question, topic, answers, template, source, line_no}."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                q, ans = line.split("\t", 1)
            except ValueError:
                continue
            m = _TOPIC_RE.search(q)
            if not m:
                continue
            topic = m.group(1).strip()
            answers = [a.strip() for a in ans.split("|") if a.strip()]
            if not topic or not answers:
                continue
            template = q[: m.start()] + "[TOPIC]" + q[m.end():]
            rows.append({"question": q, "topic": topic, "answers": answers,
                         "template": template, "source": source, "line_no": i})
    return rows


# ---------------------------------------------------------------------------
# Movie -> wiki doc mapping (unambiguous only)
# ---------------------------------------------------------------------------
def map_movies_to_docs(movies, docs):
    """Return {movie_surface: doc_index} for unambiguously mapped movies.

    Exact normalized-title match preferred; fallback strips the trailing
    parenthetical disambiguator (e.g. '(1971 film)'). 0 or >1 candidate
    docs -> unmapped. Distinct movies that case-fold to the same passage_id
    or to the same doc are dropped (passage_id collision)."""
    by_full = defaultdict(set)
    by_strip = defaultdict(set)
    for idx, (title_raw, _text) in enumerate(docs):
        by_full[norm(title_raw)].add(idx)
        by_strip[norm(_PAREN_RE.sub("", title_raw))].add(idx)
    mapping = {}
    for m in movies:
        nm = norm(m)
        cands = by_full.get(nm) or by_strip.get(nm) or set()
        if len(cands) == 1:
            mapping[m] = next(iter(cands))
    # drop normalized-name collisions (passage_id must be unique)
    by_pid = defaultdict(list)
    for m in mapping:
        by_pid[norm(m)].append(m)
    by_doc = defaultdict(list)
    for m, idx in mapping.items():
        by_doc[idx].append(m)
    dropped = set()
    for pid, ms in by_pid.items():
        if len(ms) > 1:
            dropped.update(ms)
    for idx, ms in by_doc.items():
        if len(ms) > 1:
            dropped.update(ms)
    for m in dropped:
        mapping.pop(m, None)
    return mapping


# ---------------------------------------------------------------------------
# Template -> relation-pair inference (mechanical, KB-scored)
# ---------------------------------------------------------------------------
def predict(ptype, r1, r2, topic, fwd, inv):
    """Predicted answer set for a 2-hop path of the given type."""
    out = set()
    if ptype == "inv_fwd":      # entity topic -> bridge movies -> entity answers
        for mv in inv[r1].get(topic, ()):
            out.update(fwd[r2].get(mv, ()))
    elif ptype == "fwd_inv":    # movie topic -> mid entities -> movie answers
        for mid in fwd[r1].get(topic, ()):
            out.update(inv[r2].get(mid, ()))
    out.discard(topic)
    return out


def infer_templates(qa_rows, fwd, inv):
    """Score every (ptype, r1, r2) candidate per template; accept the best
    when its exact answer-set match rate >= TEMPLATE_ACCEPT_RATE."""
    relations = sorted(fwd.keys())
    by_template = defaultdict(list)
    for row in qa_rows:
        by_template[row["template"]].append(row)
    accepted = {}
    report = []
    for template in sorted(by_template):
        instances = by_template[template][:TEMPLATE_SAMPLE]
        best_key, best_rate = None, -1.0
        for ptype in ("inv_fwd", "fwd_inv"):
            for r1 in relations:
                for r2 in relations:
                    n_ok = 0
                    n_use = 0
                    for row in instances:
                        pred = predict(ptype, r1, r2, row["topic"], fwd, inv)
                        if not pred:
                            continue
                        n_use += 1
                        if pred == set(row["answers"]):
                            n_ok += 1
                    if n_use == 0:
                        continue
                    rate = n_ok / len(instances)
                    key = (ptype, r1, r2)
                    if rate > best_rate or (rate == best_rate and key < best_key):
                        best_key, best_rate = key, rate
        ok = best_key is not None and best_rate >= TEMPLATE_ACCEPT_RATE
        report.append({"template": template, "n_questions": len(by_template[template]),
                       "best": list(best_key) if best_key else None,
                       "match_rate": round(best_rate, 4), "accepted": bool(ok)})
        if ok:
            accepted[template] = best_key
    return accepted, report


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------
def find_span(text: str, needle: str):
    i = text.lower().find(needle.lower())
    if i < 0:
        return None
    return [i, i + len(needle)]


def eligible_questions(qa_rows, accepted, fwd, inv, mapping, docs):
    """Apply the design's gates; return per-template ordered lists of QA dicts."""
    out = defaultdict(list)
    seen = set()
    stats = {"total": 0, "tpl_accepted": 0, "bridge2": 0, "docs_ok": 0,
             "answers_match": 0, "span_ok": 0}
    for row in qa_rows:
        stats["total"] += 1
        key = (row["template"], row["topic"])
        if key in seen:
            continue
        seen.add(key)
        relpair = accepted.get(row["template"])
        if relpair is None or relpair[0] != "inv_fwd":
            continue  # bridge must be a movie (design)
        stats["tpl_accepted"] += 1
        _ptype, r1, r2 = relpair
        bridges = sorted(inv[r1].get(row["topic"], ()))
        if len(bridges) != 2:
            continue
        stats["bridge2"] += 1
        if not all(b in mapping for b in bridges):
            continue
        stats["docs_ok"] += 1
        derived = set()
        for b in bridges:
            derived.update(fwd[r2].get(b, ()))
        derived.discard(row["topic"])
        if derived != set(row["answers"]):
            continue  # path must reproduce the benchmark's own answer set
        stats["answers_match"] += 1
        spans = []
        for b in bridges:
            text = docs[mapping[b]][1]
            for ans in row["answers"]:
                sp = find_span(text, ans)
                if sp is not None:
                    spans.append({"passage_id": norm(b), "span": sp})
                    break
        if not spans:
            continue  # answer string must be present in >=1 bridge doc
        stats["span_ok"] += 1
        answer = None
        covered = {s["passage_id"] for s in spans}
        for ans in row["answers"]:
            if any(find_span(docs[mapping[b]][1], ans) for b in bridges
                   if norm(b) in covered):
                answer = ans
                break
        if answer is None:
            answer = row["answers"][0]
        qa_id = "metaqa2h_" + hashlib.sha1(
            (row["question"] + "\t" + "|".join(row["answers"])).encode("utf-8")
        ).hexdigest()[:12]
        out[row["template"]].append({
            "qa_id": qa_id,
            "question": row["question"],
            "answer": answer,
            "answers": row["answers"],
            "gold_paper_ids": [norm(b) for b in bridges],
            "gold_movies": bridges,
            "topic": row["topic"],
            "qtype": "%s->%s" % (r1, r2),
            "template": row["template"],
            "hop_count": 2,
            "evidence_spans": spans,
            "source": row["source"],
        })
    return out, stats


def stratified_select(eligible_by_tpl, n_total):
    """Seeded shuffle within template, round-robin across sorted templates."""
    rng = random.Random(SEED)
    pools = OrderedDict()
    for tpl in sorted(eligible_by_tpl):
        lst = list(eligible_by_tpl[tpl])
        rng.shuffle(lst)
        pools[tpl] = lst
    selected = []
    while len(selected) < n_total and any(pools.values()):
        for tpl in pools:
            if pools[tpl]:
                selected.append(pools[tpl].pop(0))
                if len(selected) >= n_total:
                    break
    return selected


# ---------------------------------------------------------------------------
# Split artifacts
# ---------------------------------------------------------------------------
def build_split(split_qas, mapping, docs, fwd, inv):
    """Return dict of serializable artifacts for one split."""
    gold_ids = set()
    for qa in split_qas:
        gold_ids.update(qa["gold_paper_ids"])
    pid_to_doc = {norm(m): (m, idx) for m, idx in mapping.items()}
    candidates = sorted(pid for pid in pid_to_doc if pid not in gold_ids)
    n_distract = max(0, CORPUS_SIZE - len(gold_ids))
    rng = random.Random(SEED)
    distractors = rng.sample(candidates, min(n_distract, len(candidates)))
    corpus_ids = sorted(gold_ids | set(distractors))

    passages, manifest = [], []
    for pid in corpus_ids:
        m_surface, idx = pid_to_doc[pid]
        text = docs[idx][1]
        passages.append({"passage_id": pid, "paper_id": pid,
                         "text": text, "title": m_surface})
        manifest.append({"id": pid, "paper_id": pid, "title": m_surface,
                         "abstract": text, "partition": "eval"})

    qa_rows = []
    for qa in split_qas:
        row = {k: qa[k] for k in ("qa_id", "question", "answer", "answers",
                                  "gold_paper_ids", "qtype", "template",
                                  "hop_count", "evidence_spans", "source")}
        qa_rows.append(row)

    # ---- S2 silver KG: KB subgraph induced by the selected 2-hop paths ----
    nodes = OrderedDict()
    edges = OrderedDict()

    def touch(surface, is_movie):
        nid = "n::" + norm(surface)
        if nid not in nodes:
            nodes[nid] = {"node_id": nid, "name": surface, "type": "entity",
                          "aliases": [surface], "source_docs": [],
                          "_movie": bool(is_movie)}
        return nid

    def add_edge(src, rel, tgt):
        eid = "%s|%s|%s" % (src, rel, tgt)
        if eid not in edges:
            edges[eid] = {"edge_id": eid, "source": src, "target": tgt,
                          "source_id": src, "target_id": tgt,
                          "relation": rel, "relation_type": rel,
                          "source_doc": "", "reliability_score": 1.0}

    nonmovie_links = defaultdict(set)  # non-movie nid -> set(movie passage_id)
    for qa in split_qas:
        relpair = qa["qtype"].split("->")
        r1, r2 = relpair[0], relpair[1]
        t_nid = touch(qa["topic"], False)
        for b, b_pid in zip(qa["gold_movies"], qa["gold_paper_ids"]):
            b_nid = touch(b, True)
            nodes[b_nid]["_movie"] = True
            if b_pid not in nodes[b_nid]["source_docs"]:
                nodes[b_nid]["source_docs"].append(b_pid)
            # KB path triples (KB direction movie->entity) plus reverses
            add_edge(b_nid, r1, t_nid)
            add_edge(t_nid, r1 + "_inv", b_nid)
            nonmovie_links[t_nid].add(b_pid)
            for ans in fwd[r2].get(b, ()):
                a_nid = touch(ans, ans in mapping)
                add_edge(b_nid, r2, a_nid)
                add_edge(a_nid, r2 + "_inv", b_nid)
                if not nodes[a_nid]["_movie"]:
                    nonmovie_links[a_nid].add(b_pid)
    for nid, node in nodes.items():
        if not node["_movie"]:
            node["source_docs"] = sorted(nonmovie_links.get(nid, ()))
        elif not node["source_docs"]:
            # movie node that is an answer entity with a mapped doc
            node["source_docs"] = [nid[3:]] if nid[3:] in pid_to_doc else []
    node_records = []
    for nid in sorted(nodes):
        rec = dict(nodes[nid])
        rec.pop("_movie")
        node_records.append(rec)
    edge_records = [edges[eid] for eid in sorted(edges)]

    # ---- coverage assert: silver gold-title coverage is 1.0 by construction
    vocab = set()
    for n in node_records:
        vocab.add(norm(n["node_id"][3:]))
        vocab.add(norm(n["name"]))
        for a in n["aliases"]:
            vocab.add(norm(a))
    for qa in qa_rows:
        for g in qa["gold_paper_ids"]:
            assert norm(g) in vocab, "silver coverage violated: %s" % g

    return {"qa": qa_rows, "passages": passages, "manifest": manifest,
            "kg_nodes": node_records, "kg_edges": edge_records,
            "n_gold_docs": len(gold_ids), "n_corpus": len(corpus_ids)}


# ---------------------------------------------------------------------------
# Full build (pure; run twice for the determinism assert)
# ---------------------------------------------------------------------------
def build_all():
    fwd, inv, movies = parse_kb(RAW / "kb.txt")
    docs = parse_wiki(RAW / "wiki.txt")
    mapping = map_movies_to_docs(movies, docs)

    qa_rows = parse_qa(RAW / "qa_test_2hop.txt", "test")
    accepted, tpl_report = infer_templates(qa_rows, fwd, inv)
    eligible, stats = eligible_questions(qa_rows, accepted, fwd, inv, mapping, docs)
    pool = sum(len(v) for v in eligible.values())
    if pool < 2 * N_PER_SPLIT:
        dev_rows = parse_qa(RAW / "qa_dev_2hop.txt", "dev")
        acc_dev, rep_dev = infer_templates(dev_rows, fwd, inv)
        accepted.update({k: v for k, v in acc_dev.items() if k not in accepted})
        elig_dev, _ = eligible_questions(dev_rows, accepted, fwd, inv, mapping, docs)
        test_qids = {qa["qa_id"] for lst in eligible.values() for qa in lst}
        for tpl, lst in elig_dev.items():
            for qa in lst:
                if qa["qa_id"] not in test_qids:
                    eligible[tpl].append(qa)
        pool = sum(len(v) for v in eligible.values())

    selected = stratified_select(eligible, 2 * N_PER_SPLIT)
    sub = selected[:N_PER_SPLIT]
    holdout = selected[N_PER_SPLIT:2 * N_PER_SPLIT]
    splits = {"sub200": build_split(sub, mapping, docs, fwd, inv)}
    if len(holdout) >= N_PER_SPLIT:
        splits["holdout200"] = build_split(holdout, mapping, docs, fwd, inv)
    meta = {"pool": pool, "stats": stats, "tpl_report": tpl_report,
            "n_mapped_movies": len(mapping), "n_docs": len(docs),
            "n_templates_accepted_invfwd": len(
                {t for t, v in accepted.items() if v[0] == "inv_fwd"}),
            "seed": SEED}
    return splits, meta


def serialize(splits):
    """Canonical serialization of every artifact for the determinism check."""
    return json.dumps(splits, sort_keys=True, ensure_ascii=False)


def write_split(name, art):
    d = OUT_BASE / name
    (d / "kg_silver").mkdir(parents=True, exist_ok=True)
    with (d / "qa.jsonl").open("w", encoding="utf-8") as fh:
        for row in art["qa"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (d / "passages.jsonl").open("w", encoding="utf-8") as fh:
        for row in art["passages"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (d / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(art["manifest"], fh, ensure_ascii=False, indent=2)
    with (d / "kg_silver" / "proposed_kg_nodes.json").open("w", encoding="utf-8") as fh:
        json.dump(art["kg_nodes"], fh, ensure_ascii=False, indent=2)
    with (d / "kg_silver" / "proposed_kg_edges.json").open("w", encoding="utf-8") as fh:
        json.dump(art["kg_edges"], fh, ensure_ascii=False, indent=2)


def main():
    print("building pass 1 ...", flush=True)
    splits1, meta = build_all()
    print("building pass 2 (determinism check) ...", flush=True)
    splits2, _ = build_all()
    assert serialize(splits1) == serialize(splits2), "non-deterministic build"
    print("determinism check passed", flush=True)

    for name, art in splits1.items():
        write_split(name, art)
        tpl_counts = defaultdict(int)
        for qa in art["qa"]:
            tpl_counts[qa["qtype"]] += 1
        dist = " ".join("%s=%d" % (k, v) for k, v in sorted(tpl_counts.items()))
        print("%s: n_qa=%d gold_docs=%d corpus=%d kg_nodes=%d kg_edges=%d"
              % (name, len(art["qa"]), art["n_gold_docs"], art["n_corpus"],
                 len(art["kg_nodes"]), len(art["kg_edges"])), flush=True)
        print("  relpair distribution: " + dist, flush=True)

    with (OUT_BASE / "build_report.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    log_progress("S1 subset built: pool=%d splits=%s seed=%d"
                 % (meta["pool"], ",".join(sorted(splits1)), SEED))
    log_progress("S2 silver KG built: " + " ".join(
        "%s nodes=%d edges=%d" % (n, len(a["kg_nodes"]), len(a["kg_edges"]))
        for n, a in sorted(splits1.items())) + " coverage=1.0 (asserted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
