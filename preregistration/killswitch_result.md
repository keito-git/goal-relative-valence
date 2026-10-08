# Kill-switch result (Qwen2.5-7B-Instruct, discovery domains, raw position) — 2026-10-05 14:2x UTC

K0 lexical leak: LODO outcome probe at layer 0 = 0.500 (<= 0.60). NOTE: trivially satisfied because the raw
  last token is "." for every item, so layer-0 states are identical. Supplementary bag-of-words LODO check
  added below (not in v1; reported as supplementary).
K1 behaviour: self 0.927, other 0.871 (>= 0.85) PASS
K2 LODO outcome probe max over layers >= 1: 0.925 (L15, L16) PASS
K3 sst2/diffmeans auc_surface_none max: 1.000 PASS

Decision: GO. Confirmatory analysis proceeds as pre-registered (L* chosen on discovery domains per model,
H1/H2 on confirmation domains).

Exploratory observations on discovery domains (not confirmatory):
- SST-2 valence direction is dominated by surface polarity in mid layers (d_surface 4-12 at L5-L18) while
  d_outcome_self rises to ~1.8 at L15-16 (in-plane floor p95 ~0.94).
- At the last layer, d_outcome_self 0.97 > d_surface_self 0.57, while other-perspective d_outcome 0.23.
