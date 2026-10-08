# Pre-registration v9 amendment (2026-10-06, before any v9 output exists)

Motivation: second external review of the v7 manuscript. Seen so far: all H1-H17 results. Nothing below has been
computed. H20 and H21 reuse saved arrays (projections and raw-position hidden states) but have not been run on them.

## H18 Task gating at the response boundary (sweep4.py, part "taskgate")
Valence stimuli (perspectives self and none, all domains). The same text is followed by one of five suffixes before
the assistant turn: (a) eval_self "For you, was this outcome good or bad? ..." (as in H14); (b) eval_plain "Was this
outcome good or bad? ..." (no agent); (c) factual "Which party is named in the last sentence? Answer with the name
only."; (d) irrelevant "How many sentences does the text contain? Answer with a number."; (e) none (no suffix, as the
main response position). In-format valence direction per suffix (discovery stake-free items, validity gate 0.80),
late-band conflict-cell AUC for the stake at the response position, confirmation self items.
Decision rule (reported whichever holds, majority of the 7 eligible instruct models):
- task-gated: (a) exceeds each of (c), (d) by at least 0.10 and (c), (d) stay at most 0.60;
- response-gated: (c) and (d) both above 0.5 with the bootstrap CI excluding 0.5;
- otherwise mixed. (b) is descriptive (does an evaluative frame without an agent suffice?).

## H19 Goal-explicit key control (make_goalneutral2_stimuli.py)
As H13 but the goal is stated: "<P> and <Q> each had to open the <object>. They were given one brass and one steel
key; only the brass key opens the <object>." Same reading (SST-2 direction, response position, late band) and the same
criterion as H13a (CI above 0.5 in the majority of eligible instruct models). Other perspective descriptive.

## H20 Internal additive decomposition (decompose.py, saved projections)
For each eligible instruct model and layer, regress the z-scored valence projection of confirmation self items on
+/-1 codes of surface polarity s, label match m, and stake outcome o (o = s*m in this coding, so the three regressors are
orthogonal in the balanced design). Projections: in-format valence at the end of text and at the response position
(acts2 inf_valence_*), and the H14 trajectory arrays. Stimulus-bootstrap CIs (500).
Predictions (majority of models): (i) at the response position, late-band beta_s > 0 with CI excluding 0 (the lexical
component persists inside the representation); (ii) late-band beta_o > middle-band beta_o; (iii) beta_o > beta_s in the
late band at the response position. Descriptive: depth profiles of beta_s, beta_m, beta_o; first depth (>= 0.1) at
which beta_m and beta_o reach half their maximum (d_m, d_o) and whether d_m < d_o.

## H21 Lexical-leakage audit of canonical probing data (leak_audit.py, saved raw-position states)
SST-2 and GoEmotions-valence sets used for the directions (2000 texts each). 5-fold CV accuracy of (a) a unigram
bag-of-words logistic regression and (b) a per-layer L2 logistic probe on last-token states (standardised features).
Prediction: bag-of-words reaches at least 90% of the best-layer probe accuracy in both datasets (lexical recoverability
of the labels). Descriptive: middle-band probe accuracy, and probe accuracy on the out-of-fold items bag-of-words gets
wrong. Contrast reported alongside: bag-of-words is at chance on our stimuli by construction (0.500, H0).
