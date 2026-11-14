# v4 human ratings, anonymized export

Written by `../export_public_v4.py` from `../result_v4.json` (the output of `../score_v4.py`). The returned workbooks are not released, because their document properties and save paths name the raters' institution and `Rater_info` holds the raters' initials; `returned_sha256.txt` lists their hashes. The integrity results, eligibility, and every statistic are in `../result_v4.md`; the follow-up on its warnings is in `../independence_checks_v4.md`. The sent workbooks are in `../to_send/` and match the hashes in `../SEALED_build.json`.

The `total_time_as_entered` column is the rater's own entry in the total-minutes field; all three raters entered hours there (1.5, 2, 2), so the paper reports `minutes_from_session_times`.
