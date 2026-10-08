# Pre-registration v10 amendment (2026-10-06, before any v10 output exists)

Motivation: third external review. Seen so far: all H1-H21 results. In H18 the non-evaluative questions (who is named;
how many sentences) required neither stakeholder binding nor the outcome computation, so evaluation-specific gating is
confounded with task relevance of the stake.

## H22 Stake-dependent but non-evaluative questions (sweep5.py, part "taskgate2")
Same items, procedure, readout, and in-format directions as H18, with two new suffixes:
(f) outcome_factual: "Did your side win or lose? Answer with one word: won or lost." (requires binding and the outcome
    computation, no good/bad evaluation);
(g) binding_factual: "Is the party named in the last sentence on your side? Answer with one word: yes or no." (requires
    binding only, i.e. the label match m, not the outcome).
eval_self (H18 suffix a) is re-run in the same job as the reference.
Decision rule on late-band conflict-cell AUC for the stake at the response position (majority of the 7 eligible models):
- goal-computation gating: (f) > 0.60 and (f) >= (a) - 0.10;
- appraisal-specific gating: (a) - (f) >= 0.10 and (f) <= 0.60;
- otherwise intermediate. (g) descriptive.
Behaviour (descriptive): accuracy of "won" vs "lost" against the outcome o and of "yes" vs "no" against m (self items).
Family-level reporting: results also summarised per model family (Qwen2.5, Mistral, Falcon3, Granite, Phi).
