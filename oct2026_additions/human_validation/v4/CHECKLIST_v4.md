# v4 coordinator checklist

## Sending
- Send each rater only their own zip: `to_send/rater_1_v4.zip` to rater 1, and so on. One code per person; note who got which code (keep that note yourself, not in any workbook).
- Raters must not be authors of the paper or involved in the project in any way.
- Send nothing else from `v4/` (never `SEALED_build.json`, the protocol, the scorer, or any result).
- Tell each rater: work alone, no AI assistants or language models of any kind, no web search, return the filled workbook directly to you as an attachment. Everything else is in their instructions.

## What you must NOT do
- Do not fill in, edit, or "tidy" any sheet, not even formatting or a typo, before or after the rater works on it.
- Do not open a returned workbook in Excel and save it, and do not re-export or convert it.
- Do not forward any workbook, or any part of one, from one rater to another.
- Do not tell a rater a bin, discuss an item, or say what other raters answered. Answer only questions about how to use the workbook.
- Do not use AI tools (ChatGPT, Claude, Gemini, Copilot, AI translators, or others) to "help" in any way: not to translate, summarise, pre-fill, check, or complete a workbook or a message to a rater.
- Do not rename returned files.

## What to send back to us
- Each rater's original attachment exactly as received (saved straight from the email or messenger), unrenamed: `rater_k_v4.xlsx`.
- For each file: the date and time of the message it came with, and the channel (email or messenger). A screenshot or forwarded copy of the message header is enough.
- Anything a rater told you about problems with the workbook.

We place the files in `v4/returned/` unchanged and run `score_v4.py`.
