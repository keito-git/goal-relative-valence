# Pre-registration v6 amendment (2026-10-06, before any output of the second control exists)

Motivation: internal review — a single non-affective attribute (left/right) cannot show that rebinding at
judgement is specific to valence rather than to left/right. Seen so far: all H1-H10 results.

## H11 Second non-affective relational control (make_control2_stimuli.py; wooden vs metal)
Identical XOR structure and procedure as H6 with the attribute "wooden vs metal" (different semantic type from the
spatial left/right). Direction from 2,000 stake-free wood/metal sentences. Same metrics, bands, bootstrap, and
transition fit. Same reading rule as H6:
- GENERIC (per position) if, in the majority of the 7 eligible instruct models, mid-band AUC <= 0.15 AND late-minus-mid
  CI excludes 0 AND |d*_control2 - d*_valence| <= 0.10.
- SPECIFIC otherwise.
Additional pre-specified comparison: at the response position, late-band AUC of control2 below that of valence in the
majority of eligible models. Behaviour: sign consistency of B_surf' across models (reported).
Run in projection-only mode for all models (disk constraint); OLMo-2-7B-Instruct is run but excluded from counts.
