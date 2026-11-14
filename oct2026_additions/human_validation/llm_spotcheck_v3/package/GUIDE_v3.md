# Answer-grading spot-check (v3): rater guide

**한국어 요약.** 한 행마다 질문, 참조 답(reference), 모델이 낸 답(candidate)이 있습니다. 모델 답이 참조 답의 핵심 사실(질문에 실제로 답하는 부분)을 말하면 CORRECT, 아니면 INCORRECT입니다. 두 글만 비교해서는 판단이 안 될 때만 UNSURE를 고릅니다. 의학 지식은 필요 없고 쓰지도 마세요. 

## Task
Each row gives a medical question, a reference answer, and a candidate answer written by a model. Decide whether the candidate answers the question the way the reference does. Compare the two texts only; do not use outside medical knowledge, and do not look anything up. Record start and end times in `Rater_info`.

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

Return the workbook without renaming it.
