# Answer-grading spot-check (v4): rater guide

**한국어 요약.** 한 행마다 질문, 참조 답(reference), 모델이 낸 답(candidate)이 있습니다. 모델 답이 참조 답의 핵심 사실(질문에 실제로 답하는 부분)을 말하면 CORRECT, 아니면 INCORRECT입니다. 두 글만 비교해서는 판단이 안 될 때만 UNSURE를 고릅니다. 의학 지식은 필요 없고 쓰지도 마세요. 혼자 평정하고, ChatGPT·Claude·Gemini 같은 AI 도구나 번역기, 인터넷 검색은 쓰지 마세요. 다른 평정자와 문항을 두고 이야기하지 마세요. 시작·종료 시각과 확인란 3개를 `Rater_info`에 채운 뒤, 파일 이름을 바꾸지 말고 그대로 회신해 주세요.

## Task
Each row of the `Items` sheet gives a medical question, a reference answer, and a candidate answer written by a model. Decide whether the candidate answers the question the way the reference does, and choose CORRECT, INCORRECT, or UNSURE in the `verdict` column. Compare the two texts only; do not use outside medical knowledge, and do not look anything up. Record your start and end times in `Rater_info`.

Expected time: **40-60 minutes**.

## Rules
1. **CORRECT** if the candidate states the reference's key fact, entity, value, or conclusion, that is, the part of the reference that actually answers the question. It may be shorter, worded differently, or leave out secondary details of the reference.
2. **INCORRECT** if the candidate:
   - is empty, "None", "I don't know", or otherwise gives no answer;
   - names a different entity, value, or conclusion;
   - contradicts the reference anywhere, including in extra details it adds;
   - answers a different question;
   - lists several alternatives without committing to the reference's one.
3. Extra information that is consistent with the reference, or that the reference does not address, does not make a candidate incorrect.
4. Abbreviations that the reference itself introduces, and unmistakable synonyms, count as the same entity.
5. For "why" and "how" questions, the candidate must give the reference's main reason or mechanism. Giving only a minor detail of the reference is INCORRECT.
6. **UNSURE** only when equivalence cannot be decided from the two texts. Use it rarely; add a short note.

## Examples (invented)
| question | reference | candidate | verdict | why |
|---|---|---|---|---|
| Which drug is first-line for condition X? | Drug A is first-line, with drug B for patients who cannot take A. | Drug A | CORRECT | States the key fact; the omitted detail is secondary. |
| Which drug is first-line for condition X? | Drug A is first-line. | Drug A or drug B | INCORRECT | Two alternatives without committing to A. |
| Why does condition Y cause fatigue? | Because it lowers red-cell counts, reducing oxygen delivery. | It reduces oxygen delivery through anemia. | CORRECT | Same mechanism in other words. |
| Why does condition Y cause fatigue? | Because it lowers red-cell counts, reducing oxygen delivery. | Patients often feel tired. | INCORRECT | Restates the symptom; no mechanism. |
| What is the five-year survival for stage I Z treated with surgery alone? | About 90%. | About 90%, but only if chemotherapy is added. | INCORRECT | The added condition contradicts the question and reference (surgery alone). |
| What is the five-year survival for stage I Z? | About 90%. | About 90%; follow-up visits are usually yearly. | CORRECT | The extra detail is not addressed by the reference and does not contradict it (rule 3). |
| Name the tumor type. | Basal cell carcinoma (BCC). | BCC | CORRECT | Abbreviation introduced by the reference. |

## How to work
These rules are part of the study. Please follow them exactly.

- **Rate alone.** Do not discuss the items with anyone, including other raters and the coordinator. Do not look at any other rater's workbook, and do not show yours to anyone before you send it.
- **Do not use AI assistants or language models in any way.** This includes ChatGPT, Claude, Gemini, Copilot, AI features built into Excel, a browser, or an editor, and translators built on them. Do not use them to read, translate, summarise, check, or fill anything in this package.
- **Do not search the web, and do not use outside medical knowledge.** Decide each row only from the question, the reference answer, and the candidate answer on that row.
- **Use only this guide and your workbook.** Do not look at the results of the paper this study belongs to.
- Every verdict must be your own. There is no expected number of rows with any verdict.
- The coordinator can answer questions about how to use the workbook (for example, which cell to type in). The coordinator cannot answer questions about any item.
- At the end, `Rater_info` asks you to confirm these rules with yes or no. Please answer truthfully. A "no" causes no problem for you; it only tells us not to use the workbook as an independent human rating.

## Filling the workbook
Open your workbook, `rater_N_judge_v4.xlsx`, in desktop Microsoft Excel or LibreOffice Calc. Do not upload it to an online service (for example Google Sheets or Excel for the web), do not copy its contents into a new file, and do not rename it. The sheets are protected: only the yellow cells can be edited, which keeps the item text unchanged. You can widen columns and enlarge rows.

1. `Read_me`: a short version of these rules.
2. `Rater_info`: your rater code is already filled in. Fill the other yellow cells: your field or background, years working in NLP or IR (0 if none), your relation to the authors (choose from the list), the start and end of each session as date and time (for example `2026-10-02 14:05`; session 2 only if you used a second session), the total minutes you spent rating, the three yes/no statements, and your initials. Use the comments row for anything you want to tell us.
3. `Items` (102 rows): for every row, choose CORRECT, INCORRECT, or UNSURE in `verdict` (drop-down). `note` is optional; add a short note when you choose UNSURE (rule 6).

## Returning your work
1. Check that every row of `Items` has a verdict and that `Rater_info` is complete, including the start and end times and the three yes/no statements.
2. Save the workbook in the same format (.xlsx) under the same name, `rater_N_judge_v4.xlsx`.
3. Send that file, as it is, as an email or messenger attachment **directly to the coordinator**. Do not rename it, convert it (PDF, CSV, screenshots), paste its contents into a message, or upload it to a shared drive or online spreadsheet. Do not send it to other raters.

Thank you for your time.
