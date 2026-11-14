# Judge spot-check v4 coordinator checklist

## Before sending (us, not the coordinator)
- Run `python build_package_judge_v4.py --check` in `judge_spotcheck_v4/`. It must report that every sent file and `SEALED_judge_build.json` are reproduced byte for byte. Do not rebuild after a zip has been sent.

## Sending
- Send each rater only their own zip: `to_send/rater_1_judge_v4.zip` to rater 1, and so on. One code per person; note who got which code (keep that note yourself, not in any workbook).
- Raters must not be authors of the paper or involved in the project in any way, and must not have seen any grading of these answers or the paper's results. No medical background is needed.
- Send nothing else from `judge_spotcheck_v4/` (never `SEALED_judge_build.json`, the protocol, the scorer, or any result), and nothing from `judge_spotcheck/`, `judge_spotcheck_v3/`, or `llm_spotcheck_v3/`.
- Tell each rater: work alone, no AI assistants or language models of any kind (including AI translators), no web search, and return the filled workbook directly to you as an attachment, without renaming it. Expected time is 40-60 minutes. Everything else is in their `GUIDE_v4.md`.

## What you must NOT do
- Do not fill in, edit, or "tidy" any sheet, not even formatting or a typo, before or after the rater works on it.
- Do not open a returned workbook in Excel and save it, and do not re-export or convert it.
- Do not forward any workbook, or any part of one, from one rater to another.
- Do not tell a rater a verdict, discuss an item, say what anyone else answered (other raters, earlier raters, or any automatic grader), or say how many items of any kind the sheet contains. Answer only questions about how to use the workbook.
- Do not use AI tools (ChatGPT, Claude, Gemini, Copilot, AI translators, or others) to "help" in any way: not to translate, summarise, pre-fill, check, or complete a workbook or a message to a rater.
- Do not rename returned files.

## What to send back to us
- Each rater's original attachment exactly as received (saved straight from the email or messenger), unrenamed: `rater_k_judge_v4.xlsx`.
- For each file: the date and time of the message it came with, and the channel (email or messenger). A screenshot or forwarded copy of the message header is enough.
- Anything a rater told you about problems with the workbook.

We place the files in `judge_spotcheck_v4/returned/` unchanged and run `score_judge_v4.py`.
