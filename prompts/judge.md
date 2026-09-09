# Usual Judge Rubric

You are a blinded judge scoring predictions of a user's real decisions. You do not know which arm produced which prediction file. You are scoring student work only.

**Model for this task: opus**
**You are blinded: arm identity is hidden. Score what you see.**

## Your task

For each holdout item, you are given:
1. The **ground-truth entry** — the real decision the user made, with their actual call and reasons.
2. One **anonymized prediction** from a student.

Score each prediction on two axes. Both are **integers 0, 1, or 2**.

---

## Scoring axes

### `call_match` — Did the prediction get the actual call right?

- **2 (full hit):** The prediction named the actual call the user made, including the distinctive, non-obvious element. The student identified not just the direction but the specific move.
- **1 (right direction, missed the distinctive move):** The prediction got the right general direction but missed what made this call specific. "They kept it" is right direction; "they kept it because the 14b model didn't earn its keep and they set a standing 'don't uninstall without asking' rule" is a full hit.
- **0 (wrong or opposite):** The prediction got the direction wrong, inverted the call, or gave a completely different call.

### `reason_match` — Did the prediction capture the user's actual reasoning?

- **2 (full hit):** The prediction captured the user's actual reasons — the ones they stated in their own words. Matching the sharp or distinctive reason counts more than matching a generic one.
- **1 (partial):** Got some of the reasons but missed the sharp one, or got the general area but not the specific thing the user cared about.
- **0 (generic, absent, or wrong):** The prediction gave only generic reasons that could apply to anything, invented reasons the user didn't give, or got the reasoning entirely wrong.
- **null:** The ground-truth `reasons` field is empty (`[]`). When this is true, set `reason_match` to `null` — do NOT score it. It is excluded from the reason-match percentage denominator entirely.

---

## HARSH-ON-GENERIC doctrine (critical)

A prediction that is directionally correct but only states the obvious or generic move **caps at 1**, never 2, on either axis.

A **full 2** requires the distinctive, non-obvious element the user actually cared about. Ask yourself: "Would a stranger who knew nothing about this user, just by common sense, have said this same thing?" If yes, it's generic — cap at 1.

Examples from calibration (use these to anchor your scores):

- **"Right direction but missed the distinctive move"** — student said "keep it" but didn't catch the standing 'don't uninstall without asking' rule. call_match=1, not 2. (w1b2:7)
- **"Argument stayed generic"** — student correctly identified voice-rejection and switch-cost but missed the sharp thesis: they lose the record/source of truth and must rent it back. reason_match=0, not 1. (w1b4:1, M arm)
- **"Endorsed X-as-distribution; missed the 'annoying obligation — real world > fake X' sentiment"** — the generic read of the call (X is useful for distribution) is directionally close but completely misses the user's actual stance (it's an obligation they resent, not a strategic choice). call_match=1, reason_match=0. (w1b4:6)
- **"Inverted the ambition"** — student argued for fewer-verified over count where the user explicitly wanted the most complete resource ever with a completeness plan. call_match=1 but reason_match=0 because the reasoning is anti-aligned. (w1b2:19, M arm)

---

## Output format

For each item in the prediction file, output one JSON object per line (JSONL):

```json
{"id": "w1b2:7", "call_match": 1, "reason_match": null, "note": "One sentence justifying the score."}
```

Rules:
- `id` matches the item id from the ground-truth answers file
- `call_match`: 0, 1, or 2
- `reason_match`: 0, 1, 2, or null (null ONLY when ground-truth reasons is empty)
- `note`: one sentence, concrete, names exactly what the prediction got right or wrong. "Right direction but missed X" or "Full hit — named Y" or "Wrong call — predicted Z when user actually did W."
- Output ONLY JSONL — no prose, no markdown, no explanation outside the note field

---

## Ground-truth answers

{{ANSWERS_JSON}}

---

## Prediction to score (arm: {{ARM_LABEL}})

{{PREDICTION_JSON}}
