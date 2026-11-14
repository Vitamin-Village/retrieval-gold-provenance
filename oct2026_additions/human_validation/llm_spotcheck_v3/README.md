# LM spot-check of the GraphRAG-Bench judge (LM labels, not human labels)

Three Claude models labelled the same 100 judge verdicts as the earlier human rater in `../judge_spotcheck_v1/`, on 2026-09-29:
- R1 = claude-opus-5
- R2 = claude-sonnet-5
- R3 = claude-haiku-4-5

Each item was a single request with no other context: the question, the reference answer, and the candidate answer.

The system prompt was `package/GUIDE_v3.md` (SHA-256 78344a75ed3936c1451ef1cd7832e26bcc5ca680eb916f8e0a97598f0ceed19f). This guide extends the human rater's rule with explicit INCORRECT cases:
- contradicting additions;
- uncommitted alternatives;
- missing mechanisms.

The comparison with the human rater is therefore not under an identical guide. No server-side fallback was used, so every label comes from the model named in `raw.jsonl`.

Files:
- `raw.jsonl`: one line per (model, item), with the model id returned by the API, the stop reason, the full reply, the parsed verdict, token usage, and a UTC timestamp.
- `returned/rater_{1,2,3}_spotcheck.xlsx`: the blank sheets from `package/sheets/`, filled with each model's verdicts. The blank sheets hold the `SpotCheck` sheet only; `package/build_spotcheck_v3.py` also writes a `Rater_info` sheet for human raters, which the LM run does not use. Item ids, texts, and order in every blank and filled sheet are identical to the builder's output (checked item by item for all three sheets).
- `result_v3.json` and `result_v3.md`: the scores from `package/score_spotcheck_v3.py`. In these files "rater" and "pairwise_raters" refer to the three models.
- `agreement_with_v1_human.json`: each model's agreement with the earlier human rater.

To re-score without any API call, run from inside this folder `SPOT_RETURNED=returned SPOT_OUT=<an empty folder> python package/score_spotcheck_v3.py` (openpyxl): it rewrites `result_v3.json`, and `result_v3.md` apart from its title line, from the filled sheets in `returned/`. To query the models again, run `python code/llm_spotcheck_claude.py` from `oct2026_additions/` (working-tree path `eval/llm_spotcheck_claude.py`; see Reproduction in `oct2026_additions/README.md`); it needs the anthropic SDK, openpyxl, and an API key.
