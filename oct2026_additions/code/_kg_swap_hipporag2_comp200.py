"""HippoRAG 2 KG swap on the 2WikiMultiHop compositional split (comp200).

Protocol: results/twowiki_comp200/PROTOCOL.md, Section 7.

Runs eval/_kg_swap_hipporag2.py UNCHANGED (same HippoRAG config, BGE-M3 Transformers backend,
llama3.1:8b recognition-memory filter, node-provenance KG injection through HippoRAG's OpenIE
cache, OpenIE stubbed to raise, graph verification, LLM gate). The wrapper only
  (a) adds "comp200" to the accepted --split values (the driver's argparse choices are
      sub200/holdout200), and
  (b) points the run directory at results/twowiki_comp200/hipporag2/runs/ so nothing under
      results/hipporag2_swap/ is read as a reusable index or written.
Only --bench 2wiki --split comp200 is allowed.

Run in conda env rag-graphrag-py311 from paper/ (PYTHONUTF8=1, HF_HUB_OFFLINE=1):
    python eval/_kg_swap_hipporag2_comp200.py --split comp200 --kg kg_silver [--index-only]
    python eval/_kg_swap_hipporag2_comp200.py --split comp200 --kg kg_openie
Scoring: eval/_kg_swap_hipporag2_comp200_score.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kg_swap_hipporag2 as H  # noqa: E402

H.OUT = H.ROOT / "results" / "twowiki_comp200" / "hipporag2" / "runs"

_orig_add_argument = argparse.ArgumentParser.add_argument


def _add_argument(self, *args, **kwargs):
    if args and args[0] == "--split" and "choices" in kwargs:
        kwargs["choices"] = list(kwargs["choices"]) + ["comp200"]
    return _orig_add_argument(self, *args, **kwargs)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--split" not in argv or argv[argv.index("--split") + 1] != "comp200":
        sys.exit("this wrapper runs --split comp200 only")
    if "--bench" in argv and argv[argv.index("--bench") + 1] != "2wiki":
        sys.exit("this wrapper runs --bench 2wiki only")
    argparse.ArgumentParser.add_argument = _add_argument
    H.main()
