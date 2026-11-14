# Answer-Paper Verification — Second-Rater Rubric v1.0

English translation of the original Korean rater instructions (the system prompt of `rate_spotcheck.py`, which
appends an output directive). Notes addressed to the authors are omitted.

## 0. Role of this document

The verifier is Gemini Flash (Google). The second rater for Cohen's κ is an independent Claude model
(Anthropic) that follows this rubric. The two raters belong to different model families, and the second rater
performs the same task independently, without seeing the verifier's judgments (blind).

## 1. Task definition (identical to the verifier prompt)

Input: one question, a numbered list of candidate paper excerpts (`0..N`), and a final `NONE` option.
Output: the index of the single excerpt that actually states the answer to the question. If no excerpt states the
answer, `NONE`.

Use the same judgment criteria as `VERIFIER_SYSTEM_PROMPT` of `verify_answer_papers.py` (if the rater measured a
different construct, κ would measure a difference in rules rather than reliability).

## 2. Decision rules (conservative, text-based)

1. **Text evidence only.** Judge only from what is actually written in the excerpts. Do not fill gaps with outside
   knowledge or inference ("it is probably this paper").
2. **The answer must be stated.** Choose an index only if the answer to the question is explicitly stated in or
   contained in the excerpt. Do not choose an excerpt that only has a similar topic but not the answer itself.
3. **NONE when uncertain.** If no excerpt clearly states the answer, choose NONE. If two excerpts both seem to
   contain the answer, choose the one that states the answer sentence more directly; if this cannot be decided,
   choose NONE.
4. **Exactly one.** Even if several answers seem correct, choose the single index that states the answer most
   directly.
5. **Surface-form differences are allowed.** If the question and the excerpt express the same entity with different
   abbreviations, synonyms, or unit notation, count it as the answer (the intent of this benchmark's aliases), but
   only when the identity of meaning can be confirmed from the text.
6. **No outside knowledge.** Even if the rater knows the fact, do not choose an index whose excerpt does not
   contain it.

## 3. Output format

For each item, exactly one line: `INDEX: <n>` or `INDEX: NONE`. Output no other explanation (the format is the same
as the verifier's, so the same parser processes both, for a fair comparison).
