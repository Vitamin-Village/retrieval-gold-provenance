# -*- coding: utf-8 -*-
"""Blackbox tests for the KG transplant pipeline (2026-06-11).

Targets (implemented independently; imported lazily so this suite is runnable
both before and after the implementation lands):

  * paper/eval/build_transplant_kgs.py
      Builds <split>/kg_openie_svocab/ with
        - proposed_kg_nodes.json : UNION of kg_openie and kg_silver nodes,
          deduplicated by node_id, silver record wins on collision
        - proposed_kg_edges.json : OpenIE edges only, unchanged set
        - transplant_manifest.json : counts
      Deterministic, leaves source dirs untouched. May optionally build
      <split>/kg_silver_devocab/ (skip, not fail, when absent).

  * paper/eval/_kg_transplant_2wiki.py
      Pure helpers mirroring _kg_swap_2wiki_r2.py: bootstrap CI (seed
      20260611, 10000 resamples, percentile 2.5/97.5), exact two-sided sign
      test (min tail doubled, capped at 1.0), per-question hit extraction
      reading record["metrics"]["hit_at_k"] (NOT a flat field).

ASCII-only, cp949-safe, no network, no API calls. The full evaluation run is
NOT executed here.
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 20260611
BUILDER_MODULE = "paper.eval.build_transplant_kgs"
RUNNER_MODULE = "paper.eval._kg_transplant_2wiki"
BUILDER_SCRIPT = ROOT / "paper" / "eval" / "build_transplant_kgs.py"
REAL_SUB200 = ROOT / "paper" / "external" / "2wiki" / "sub200"

OUT_DIR_NAME = "kg_openie_svocab"
NODES_FILE = "proposed_kg_nodes.json"
EDGES_FILE = "proposed_kg_edges.json"
MANIFEST_FILE = "transplant_manifest.json"

NODE_REQUIRED_FIELDS = ("node_id", "name", "type", "aliases", "source_docs")
EDGE_REQUIRED_FIELDS = ("edge_id", "source", "target", "relation")


# ---------------------------------------------------------------------------
# fixture: a tiny fake split directory
# ---------------------------------------------------------------------------

OPENIE_NODES = [
    {"node_id": "n1", "name": "Alpha", "type": "entity",
     "aliases": ["alpha"], "source_docs": ["d1"]},
    {"node_id": "n2", "name": "Beta-OPENIE", "type": "entity",
     "aliases": [], "source_docs": ["d1", "d2"]},
    {"node_id": "n3", "name": "Gamma", "type": "entity",
     "aliases": ["g"], "source_docs": ["d2"]},
]

SILVER_NODES = [
    {"node_id": "n2", "name": "Beta-SILVER", "type": "entity",
     "aliases": ["beta"], "source_docs": ["d9"], "wikidata_ids": ["Q2"]},
    {"node_id": "n4", "name": "Delta", "type": "entity",
     "aliases": [], "source_docs": ["d3"], "wikidata_ids": ["Q4"]},
]

OPENIE_EDGES = [
    {"edge_id": "e1", "source": "n1", "target": "n2", "relation": "rel_a",
     "relation_type": "verb", "reliability_score": 0.5, "source_doc": "d1",
     "source_docs": ["d1"], "source_id": "n1", "target_id": "n2"},
    {"edge_id": "e2", "source": "n2", "target": "n3", "relation": "rel_b",
     "relation_type": "verb", "reliability_score": 0.7, "source_doc": "d2",
     "source_docs": ["d2"], "source_id": "n2", "target_id": "n3"},
]

SILVER_EDGES = [
    {"edge_id": "s1", "source": "n2", "target": "n4", "relation": "rel_s",
     "relation_type": "wikidata", "reliability_score": 1.0,
     "source_doc": "d9", "source_docs": ["d9"],
     "source_id": "n2", "target_id": "n4"},
]

UNION_NODE_IDS = {"n1", "n2", "n3", "n4"}  # 4 nodes after dedup
OPENIE_EDGE_IDS = {"e1", "e2"}             # silver edge s1 must NOT leak in


def _write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=True, indent=1),
                    encoding="utf-8")


def make_split(base):
    """Create a minimal fake split dir with kg_openie and kg_silver."""
    split = Path(base)
    (split / "kg_openie").mkdir(parents=True, exist_ok=True)
    (split / "kg_silver").mkdir(parents=True, exist_ok=True)
    _write_json(split / "kg_openie" / NODES_FILE, OPENIE_NODES)
    _write_json(split / "kg_openie" / EDGES_FILE, OPENIE_EDGES)
    _write_json(split / "kg_silver" / NODES_FILE, SILVER_NODES)
    _write_json(split / "kg_silver" / EDGES_FILE, SILVER_EDGES)
    # harmless extras some builders may expect alongside the KG dirs
    (split / "qa.jsonl").write_text(
        json.dumps({"qa_id": "q1", "question": "who", "answer": "Alpha"})
        + "\n", encoding="utf-8")
    (split / "passages.jsonl").write_text(
        json.dumps({"doc_id": "d1", "text": "Alpha relates to Beta."})
        + "\n", encoding="utf-8")
    _write_json(split / "manifest.json", {"name": "fixture", "n": 1})
    return split


# ---------------------------------------------------------------------------
# builder driver (blackbox: discover the entry point at runtime)
# ---------------------------------------------------------------------------

_BUILDER_FUNC_CANDIDATES = (
    "build_split", "build_transplant_for_split", "build_for_split",
    "build_one", "build_transplant", "build_transplant_kg", "build_svocab",
    "process_split", "build_dir", "build",
)

_ARGV_CANDIDATES = (
    ["--split-dir"], ["--split"], ["--dir"], ["--splits"], [],
)


def _outputs_present(split_dir):
    out = Path(split_dir) / OUT_DIR_NAME
    return (out / NODES_FILE).exists() and (out / EDGES_FILE).exists()


def run_builder(split_dir):
    """Drive the builder on split_dir via module API, main(argv), or CLI."""
    mod = pytest.importorskip(BUILDER_MODULE)
    split_dir = Path(split_dir)
    errors = []

    for name in _BUILDER_FUNC_CANDIDATES:
        fn = getattr(mod, name, None)
        if not callable(fn):
            continue
        for arg in (split_dir, str(split_dir)):
            try:
                fn(arg)
            except BaseException as exc:  # keep probing other entry points
                errors.append("%s(%r): %s: %s"
                              % (name, type(arg).__name__,
                                 type(exc).__name__, exc))
            if _outputs_present(split_dir):
                return

    main = getattr(mod, "main", None)
    if callable(main):
        for flag in _ARGV_CANDIDATES:
            argv = flag + [str(split_dir)]
            try:
                main(argv)
            except BaseException as exc:
                errors.append("main(%r): %s: %s"
                              % (argv, type(exc).__name__, exc))
            if _outputs_present(split_dir):
                return

    if BUILDER_SCRIPT.exists():
        for flag in _ARGV_CANDIDATES:
            argv = [sys.executable, str(BUILDER_SCRIPT)] + flag \
                + [str(split_dir)]
            try:
                proc = subprocess.run(argv, capture_output=True, text=True,
                                      cwd=str(ROOT), timeout=120)
                if proc.returncode != 0:
                    errors.append("cli %r: rc=%d" % (flag, proc.returncode))
            except Exception as exc:
                errors.append("cli %r: %s: %s"
                              % (flag, type(exc).__name__, exc))
            if _outputs_present(split_dir):
                return

    pytest.skip("could not drive the builder on a temp split dir; "
                "attempts: %s" % ("; ".join(errors) or "none"))


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dir_hashes(root):
    """sha256 of every file under root, keyed by relative posix path."""
    root = Path(root)
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = hashlib.sha256(
                p.read_bytes()).hexdigest()
    return out


def _flatten_ints(obj, acc=None):
    if acc is None:
        acc = []
    if isinstance(obj, bool):
        return acc
    if isinstance(obj, int):
        acc.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten_ints(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_ints(v, acc)
    return acc


def _norm_edges(edges):
    return sorted((json.dumps(e, sort_keys=True) for e in edges))


# ---------------------------------------------------------------------------
# 1. builder union correctness on a temp fixture
# ---------------------------------------------------------------------------

class TestBuilderUnion:
    def test_union_nodes_and_openie_edges(self, tmp_path):
        split = make_split(tmp_path / "split")
        run_builder(split)
        out = split / OUT_DIR_NAME

        nodes = _load(out / NODES_FILE)
        assert isinstance(nodes, list)
        ids = [n["node_id"] for n in nodes]
        assert len(ids) == len(set(ids)), "duplicate node_id in union"
        assert set(ids) == UNION_NODE_IDS

        by_id = {n["node_id"]: n for n in nodes}
        # silver record wins on collision (n2 exists in both sources)
        assert by_id["n2"]["name"] == "Beta-SILVER"
        # openie-only and silver-only nodes both survive
        assert by_id["n1"]["name"] == "Alpha"
        assert by_id["n3"]["name"] == "Gamma"
        assert by_id["n4"]["name"] == "Delta"
        for n in nodes:
            for f in NODE_REQUIRED_FIELDS:
                assert f in n, "node %s missing field %s" % (n.get(
                    "node_id"), f)

        edges = _load(out / EDGES_FILE)
        assert isinstance(edges, list)
        assert {e["edge_id"] for e in edges} == OPENIE_EDGE_IDS
        assert "s1" not in {e["edge_id"] for e in edges}, \
            "silver edge leaked into transplant edges"
        # OpenIE edges pass through unchanged (record-level equality)
        assert _norm_edges(edges) == _norm_edges(OPENIE_EDGES)
        for e in edges:
            for f in EDGE_REQUIRED_FIELDS:
                assert f in e, "edge %s missing field %s" % (e.get(
                    "edge_id"), f)

    def test_node_ordering_stable_across_rebuilds(self, tmp_path):
        a = make_split(tmp_path / "a")
        b = make_split(tmp_path / "b")
        run_builder(a)
        run_builder(b)
        ids_a = [n["node_id"] for n in _load(a / OUT_DIR_NAME / NODES_FILE)]
        ids_b = [n["node_id"] for n in _load(b / OUT_DIR_NAME / NODES_FILE)]
        assert ids_a == ids_b

    def test_manifest_counts(self, tmp_path):
        split = make_split(tmp_path / "split")
        run_builder(split)
        mpath = split / OUT_DIR_NAME / MANIFEST_FILE
        assert mpath.exists(), "transplant_manifest.json missing"
        manifest = _load(mpath)
        assert isinstance(manifest, dict)
        ints = set(_flatten_ints(manifest))
        assert len(UNION_NODE_IDS) in ints, \
            "manifest lacks union node count %d (ints=%s)" % (
                len(UNION_NODE_IDS), sorted(ints))
        assert len(OPENIE_EDGE_IDS) in ints, \
            "manifest lacks edge count %d (ints=%s)" % (
                len(OPENIE_EDGE_IDS), sorted(ints))

    def test_optional_silver_devocab(self, tmp_path):
        split = make_split(tmp_path / "split")
        run_builder(split)
        dev = split / "kg_silver_devocab"
        if not dev.exists():
            pytest.skip("kg_silver_devocab not built (optional condition)")
        nodes = _load(dev / NODES_FILE)
        assert isinstance(nodes, list)
        assert all("node_id" in n for n in nodes)
        edges = _load(dev / EDGES_FILE)
        assert isinstance(edges, list)


# ---------------------------------------------------------------------------
# 2. builder determinism
# ---------------------------------------------------------------------------

class TestBuilderDeterminism:
    def test_two_fresh_runs_byte_identical(self, tmp_path):
        a = make_split(tmp_path / "a")
        b = make_split(tmp_path / "b")
        run_builder(a)
        run_builder(b)
        ha = _dir_hashes(a / OUT_DIR_NAME)
        hb = _dir_hashes(b / OUT_DIR_NAME)
        assert ha, "no output files produced"
        assert ha == hb

    def test_rerun_on_same_dir_byte_identical(self, tmp_path):
        split = make_split(tmp_path / "split")
        run_builder(split)
        first = _dir_hashes(split / OUT_DIR_NAME)
        assert first, "no output files produced"
        run_builder(split)
        second = _dir_hashes(split / OUT_DIR_NAME)
        assert first == second


# ---------------------------------------------------------------------------
# 3. builder leaves source dirs untouched
# ---------------------------------------------------------------------------

class TestBuilderSourceIntegrity:
    def test_sources_unmodified(self, tmp_path):
        split = make_split(tmp_path / "split")
        before_openie = _dir_hashes(split / "kg_openie")
        before_silver = _dir_hashes(split / "kg_silver")
        run_builder(split)
        assert _dir_hashes(split / "kg_openie") == before_openie
        assert _dir_hashes(split / "kg_silver") == before_silver


# ---------------------------------------------------------------------------
# 4. real sub200 artifact (structural validation, only if present on disk)
# ---------------------------------------------------------------------------

class TestRealSub200Artifact:
    @pytest.fixture()
    def real_out(self):
        out = REAL_SUB200 / OUT_DIR_NAME
        if not ((out / NODES_FILE).exists() and (out / EDGES_FILE).exists()):
            pytest.skip("real sub200/kg_openie_svocab not built yet")
        return out

    def test_silver_nodes_all_in_union(self, real_out):
        union_ids = {n["node_id"]
                     for n in _load(real_out / NODES_FILE)}
        silver_ids = {n["node_id"]
                      for n in _load(REAL_SUB200 / "kg_silver" / NODES_FILE)}
        missing = silver_ids - union_ids
        assert not missing, "silver node_ids missing from union: %s" % (
            sorted(missing)[:10])
        openie_ids = {n["node_id"]
                      for n in _load(REAL_SUB200 / "kg_openie" / NODES_FILE)}
        missing_o = openie_ids - union_ids
        assert not missing_o, "openie node_ids missing from union: %s" % (
            sorted(missing_o)[:10])

    def test_edge_count_matches_real_openie(self, real_out):
        out_edges = _load(real_out / EDGES_FILE)
        src_edges = _load(REAL_SUB200 / "kg_openie" / EDGES_FILE)
        assert len(out_edges) == len(src_edges)

    def test_real_artifact_parses_and_has_fields(self, real_out):
        nodes = _load(real_out / NODES_FILE)
        assert isinstance(nodes, list) and nodes
        for n in nodes[:50]:
            for f in NODE_REQUIRED_FIELDS:
                assert f in n
        edges = _load(real_out / EDGES_FILE)
        assert isinstance(edges, list) and edges
        for e in edges[:50]:
            for f in EDGE_REQUIRED_FIELDS:
                assert f in e
        mpath = real_out / MANIFEST_FILE
        if mpath.exists():
            assert isinstance(_load(mpath), dict)


# ---------------------------------------------------------------------------
# runner helpers (sign test / bootstrap / per-question extraction)
# ---------------------------------------------------------------------------

def _runner_mod():
    return pytest.importorskip(RUNNER_MODULE)


def _get_fn(mod, names):
    for n in names:
        fn = getattr(mod, n, None)
        if callable(fn):
            return fn
    pytest.skip("none of %s found in %s" % (names, mod.__name__))


def _sign_fn():
    return _get_fn(_runner_mod(),
                   ("sign_test", "exact_sign_test", "_sign_test"))


def _boot_fn():
    return _get_fn(_runner_mod(),
                   ("boot_ci", "bootstrap_ci", "_boot_ci", "boot"))


def _hits_fn():
    return _get_fn(_runner_mod(),
                   ("per_q_hits", "per_question_hits", "_per_q_hits"))


def _p_of(res):
    """Sign test may return p alone or (pos, neg, p)."""
    if isinstance(res, (tuple, list)):
        return float(res[-1])
    return float(res)


# ---------------------------------------------------------------------------
# 5. exact two-sided sign test
# ---------------------------------------------------------------------------

class TestSignTest:
    def test_five_pos_zero_neg(self):
        res = _sign_fn()([1.0] * 5)
        assert _p_of(res) == pytest.approx(0.0625)  # 2 * 0.5**5
        if isinstance(res, (tuple, list)) and len(res) >= 3:
            assert res[0] == 5 and res[1] == 0

    def test_symmetric_under_sign_flip(self):
        p_pos = _p_of(_sign_fn()([1.0] * 5))
        p_neg = _p_of(_sign_fn()([-1.0] * 5))
        assert p_pos == pytest.approx(p_neg)

    def test_no_discordant_pairs_gives_one(self):
        res = _sign_fn()([0.0, 0.0, 0.0])
        assert _p_of(res) == pytest.approx(1.0)

    def test_capped_at_one(self):
        # n=2, k=1: tail = (C(2,0)+C(2,1))/4 = 0.75, doubled = 1.5 -> cap 1.0
        assert _p_of(_sign_fn()([1.0, -1.0])) == pytest.approx(1.0)

    def test_four_pos_one_neg(self):
        # n=5, k=1: tail = (1+5)/32, doubled = 12/32 = 0.375
        assert _p_of(_sign_fn()([1.0, 1.0, 1.0, 1.0, -1.0])) \
            == pytest.approx(0.375)

    def test_zeros_ignored(self):
        p_with_zeros = _p_of(_sign_fn()([1.0] * 5 + [0.0, 0.0]))
        assert p_with_zeros == pytest.approx(0.0625)


# ---------------------------------------------------------------------------
# 6. bootstrap CI helper
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    DIFFS = [1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0]

    def _ci(self, diffs):
        random.seed(SEED)  # harmless if the helper reseeds internally
        res = _boot_fn()(list(diffs))
        return float(res[0]), float(res[1])

    def test_deterministic_repeated_calls(self):
        first = self._ci(self.DIFFS)
        second = self._ci(self.DIFFS)
        assert first == second

    def test_ci_ordering(self):
        lo, hi = self._ci(self.DIFFS)
        assert lo <= hi

    def test_ci_brackets_plausible_range(self):
        lo, hi = self._ci(self.DIFFS)
        assert -1.0 <= lo <= hi <= 1.0

    def test_degenerate_all_zero(self):
        lo, hi = self._ci([0.0] * 8)
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(0.0)

    def test_degenerate_constant(self):
        lo, hi = self._ci([1.0] * 8)
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 7. per-question hit extraction (metrics.hit_at_k regression guard)
# ---------------------------------------------------------------------------

class TestPerQuestionHits:
    def _payload(self):
        # flat record["hit_at_k"] is a DECOY: the documented pitfall in
        # _kg_swap_2wiki_r2.py is reading the flat field (always wrong)
        # instead of record["metrics"]["hit_at_k"].
        return {"results": {"sysA": {
            "records": [
                {"qa_id": "q1", "hit_at_k": 0.0,
                 "metrics": {"hit_at_k": 1.0}},
                {"qa_id": "q2", "hit_at_k": 1.0,
                 "metrics": {"hit_at_k": 0.0}},
                {"qa_id": 3, "metrics": {"hit_at_k": 1.0}},
            ],
            "summary": {"overall": {"hit_at_k": 2.0 / 3.0}},
        }}}

    def test_reads_nested_metrics_not_flat_field(self):
        res = _hits_fn()(self._payload(), "sysA")
        norm = {str(k): float(v) for k, v in res.items()}
        assert norm["q1"] == pytest.approx(1.0), \
            "flat hit_at_k decoy was read instead of metrics.hit_at_k"
        assert norm["q2"] == pytest.approx(0.0), \
            "flat hit_at_k decoy was read instead of metrics.hit_at_k"
        assert norm["3"] == pytest.approx(1.0)
        assert len(norm) == 3

    def test_values_are_floats(self):
        res = _hits_fn()(self._payload(), "sysA")
        assert all(isinstance(v, float) for v in res.values())


# ---------------------------------------------------------------------------
# 8. import-only check for the runner module
# ---------------------------------------------------------------------------

class TestRunnerImportSmoke:
    def test_import_does_not_execute_main(self):
        mod = _runner_mod()
        assert mod is not None
        # importing must not have triggered the full evaluation; the module
        # should merely expose helpers (at least one recognizable name).
        names = dir(mod)
        assert any(n in names for n in (
            "sign_test", "exact_sign_test", "_sign_test",
            "boot_ci", "bootstrap_ci", "_boot_ci",
            "per_q_hits", "per_question_hits", "_per_q_hits",
            "main")), "runner module exposes no recognizable helpers"

    def test_builder_import_smoke(self):
        mod = pytest.importorskip(BUILDER_MODULE)
        assert mod is not None


# ---------------------------------------------------------------------------
# 9. q-curve partial-provenance conditions (2026-06-12, W1 defense)
# ---------------------------------------------------------------------------

Q_SEED = 20260612
Q_VALUES = (0.25, 0.5, 0.75)
Q_DIRS = {0.25: "kg_openie_svocab_q25",
          0.5: "kg_openie_svocab_q50",
          0.75: "kg_openie_svocab_q75"}
NOPROV_DIR_NAME = "kg_openie_svocab_noprov"
SILVER_SOURCED_IDS = {"n2", "n4"}  # fixture: collision n2 + silver-only n4


def _ceil_div_q(q, n):
    import math
    return int(math.ceil(q * n))


class TestBuilderQCurve:
    def test_q_dirs_built_with_all_files(self, tmp_path):
        split = make_split(tmp_path / "split")
        run_builder(split)
        for q in Q_VALUES:
            qd = split / Q_DIRS[q]
            assert (qd / NODES_FILE).exists(), "%s nodes missing" % Q_DIRS[q]
            assert (qd / EDGES_FILE).exists(), "%s edges missing" % Q_DIRS[q]
            assert (qd / MANIFEST_FILE).exists(), \
                "%s manifest missing" % Q_DIRS[q]

    def test_q_nodes_union_and_edges_match_t1(self, tmp_path):
        split = make_split(tmp_path / "split")
        run_builder(split)
        t1_edges = _norm_edges(_load(split / OUT_DIR_NAME / EDGES_FILE))
        for q in Q_VALUES:
            qd = split / Q_DIRS[q]
            nodes = _load(qd / NODES_FILE)
            ids = [n["node_id"] for n in nodes]
            assert set(ids) == UNION_NODE_IDS
            assert len(ids) == len(set(ids))
            assert _norm_edges(_load(qd / EDGES_FILE)) == t1_edges

    def test_q_kept_fraction_and_emptied_rest(self, tmp_path):
        split = make_split(tmp_path / "split")
        run_builder(split)
        t1_by_id = {n["node_id"]: n
                    for n in _load(split / OUT_DIR_NAME / NODES_FILE)}
        n_silver = len(SILVER_SOURCED_IDS)
        for q in Q_VALUES:
            nodes = _load(split / Q_DIRS[q] / NODES_FILE)
            by_id = {n["node_id"]: n for n in nodes}
            kept = {nid for nid in SILVER_SOURCED_IDS
                    if by_id[nid]["source_docs"]}
            emptied = SILVER_SOURCED_IDS - kept
            assert len(kept) == _ceil_div_q(q, n_silver), \
                "q=%s kept %d != ceil(q*N)=%d" % (
                    q, len(kept), _ceil_div_q(q, n_silver))
            for nid in kept:  # kept nodes carry the SILVER mapping (as T1)
                assert by_id[nid]["source_docs"] == \
                    t1_by_id[nid]["source_docs"]
            for nid in emptied:
                assert by_id[nid]["source_docs"] == []
            # openie-only nodes untouched
            for nid in UNION_NODE_IDS - SILVER_SOURCED_IDS:
                assert by_id[nid]["source_docs"] == \
                    t1_by_id[nid]["source_docs"]
            # non-source_docs fields identical to T1 on every node
            for nid, node in by_id.items():
                stripped = {k: v for k, v in node.items()
                            if k != "source_docs"}
                t1_stripped = {k: v for k, v in t1_by_id[nid].items()
                               if k != "source_docs"}
                assert stripped == t1_stripped

    def test_kept_sets_nested_across_q(self, tmp_path):
        split = make_split(tmp_path / "split")
        run_builder(split)
        kept = {}
        for q in Q_VALUES:
            manifest = _load(split / Q_DIRS[q] / MANIFEST_FILE)
            kept[q] = set(manifest["kept_node_ids"])
        assert kept[0.25] <= kept[0.5] <= kept[0.75], \
            "kept sets not nested (prefix rule violated)"

    def test_manifest_records_q_seed_and_counts(self, tmp_path):
        split = make_split(tmp_path / "split")
        run_builder(split)
        n_silver = len(SILVER_SOURCED_IDS)
        for q in Q_VALUES:
            manifest = _load(split / Q_DIRS[q] / MANIFEST_FILE)
            assert manifest["q"] == pytest.approx(q)
            assert manifest["seed"] == Q_SEED
            assert manifest["n_silver_sourced"] == n_silver
            assert manifest["n_source_docs_kept"] == _ceil_div_q(q, n_silver)
            assert manifest["n_source_docs_emptied"] == \
                n_silver - _ceil_div_q(q, n_silver)
            assert len(manifest["kept_node_ids"]) == \
                manifest["n_source_docs_kept"]
            assert set(manifest["kept_node_ids"]) <= SILVER_SOURCED_IDS

    def test_q_dirs_deterministic_across_fresh_runs(self, tmp_path):
        a = make_split(tmp_path / "a")
        b = make_split(tmp_path / "b")
        run_builder(a)
        run_builder(b)
        for q in Q_VALUES:
            ha = _dir_hashes(a / Q_DIRS[q])
            hb = _dir_hashes(b / Q_DIRS[q])
            assert ha, "no output files produced for %s" % Q_DIRS[q]
            assert ha == hb


class TestQBoundaryEquivalence:
    """q=1.0 must reproduce T1 (kg_openie_svocab) and q=0.0 must reproduce
    T1b (kg_openie_svocab_noprov), node-by-node including source_docs."""

    def test_builder_exposes_boundary_helpers(self):
        mod = pytest.importorskip(BUILDER_MODULE)
        assert callable(getattr(mod, "build_q_nodes", None))
        assert callable(getattr(mod, "verify_q_boundaries", None))

    def test_verify_q_boundaries_on_fixture(self, tmp_path):
        mod = pytest.importorskip(BUILDER_MODULE)
        split = make_split(tmp_path / "split")
        run_builder(split)
        res = mod.verify_q_boundaries(split)
        assert res["q1_matches_t1"] is True
        assert res["q0_matches_t1b"] is True

    def test_q1_q0_node_lists_match_endpoint_files(self, tmp_path):
        mod = pytest.importorskip(BUILDER_MODULE)
        split = make_split(tmp_path / "split")
        run_builder(split)
        t1_nodes = _load(split / OUT_DIR_NAME / NODES_FILE)
        t1b_nodes = _load(split / NOPROV_DIR_NAME / NODES_FILE)
        merged = {n["node_id"]: n for n in t1_nodes}
        q1, kept1 = mod.build_q_nodes(merged, SILVER_SOURCED_IDS, 1.0)
        q0, kept0 = mod.build_q_nodes(merged, SILVER_SOURCED_IDS, 0.0)
        norm = lambda ns: [json.dumps(n, sort_keys=True) for n in ns]  # noqa: E731
        assert norm(q1) == norm(t1_nodes)
        assert norm(q0) == norm(t1b_nodes)
        assert set(kept1) == SILVER_SOURCED_IDS
        assert set(kept0) == set()

    def test_boundary_check_recorded_in_q_manifests(self, tmp_path):
        split = make_split(tmp_path / "split")
        run_builder(split)
        for q in Q_VALUES:
            manifest = _load(split / Q_DIRS[q] / MANIFEST_FILE)
            bc = manifest.get("boundary_check")
            assert bc is not None, "q manifest lacks boundary_check"
            assert bc["q1_matches_t1"] is True
            assert bc["q0_matches_t1b"] is True


class TestRealQArtifacts:
    """Structural validation of the real sub200 q dirs (skip if not built)."""

    @pytest.fixture()
    def real_q_dirs(self):
        dirs = {q: REAL_SUB200 / Q_DIRS[q] for q in Q_VALUES}
        if not all((d / NODES_FILE).exists() for d in dirs.values()):
            pytest.skip("real sub200 q dirs not built yet")
        return dirs

    def test_real_kept_counts_match_ceil_rule(self, real_q_dirs):
        # NOTE: kept nodes retain the SILVER source_docs verbatim, which may
        # be [] for some silver nodes; counting non-empty source_docs would
        # therefore undercount. Verify against the silver records instead.
        silver_by_id = {n["node_id"]: n for n in _load(
            REAL_SUB200 / "kg_silver" / NODES_FILE)}
        n_silver = len(silver_by_id)
        for q, qd in real_q_dirs.items():
            manifest = _load(qd / MANIFEST_FILE)
            assert manifest["n_silver_sourced"] == n_silver
            assert manifest["n_source_docs_kept"] == _ceil_div_q(q, n_silver)
            kept_ids = set(manifest["kept_node_ids"])
            assert len(kept_ids) == _ceil_div_q(q, n_silver)
            assert kept_ids <= set(silver_by_id)
            by_id = {n["node_id"]: n for n in _load(qd / NODES_FILE)}
            for nid, silver_node in silver_by_id.items():
                if nid in kept_ids:
                    assert by_id[nid]["source_docs"] == \
                        silver_node["source_docs"]
                else:
                    assert by_id[nid]["source_docs"] == []

    def test_real_boundaries_verify(self, real_q_dirs):
        mod = pytest.importorskip(BUILDER_MODULE)
        if not (REAL_SUB200 / NOPROV_DIR_NAME / NODES_FILE).exists():
            pytest.skip("real sub200 noprov dir not built")
        res = mod.verify_q_boundaries(REAL_SUB200)
        assert res["q1_matches_t1"] is True
        assert res["q0_matches_t1b"] is True


class TestRunnerQWiring:
    def test_runner_exposes_q_conditions(self):
        mod = _runner_mod()
        qkgs = getattr(mod, "Q_KGS", None)
        assert qkgs is not None, "runner lacks Q_KGS wiring"
        assert [q for q, _ in qkgs] == [0.25, 0.5, 0.75]
        assert [kg for _, kg in qkgs] == [Q_DIRS[q] for q in (0.25, 0.5, 0.75)]
