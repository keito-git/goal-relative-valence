# Pre-registration v3 amendment (2026-10-05 ~15:35 UTC)

Seen so far: representational profiles of qwen25_7b_it (discovery + confirmation) and olmo2_7b_it
(confirmation, behaviour < 0.85). Ablation H4 for qwen25_7b_it (holds on confirmation/mid, not on discovery;
exploratory decomposition shows the B_surf reduction is within the range of in-plane random ablations).
NOT yet inspected: any result of mistral_7b_it, falcon3_7b_it, granite31_8b_it, qwen25_7b_base,
olmo2_7b_base, phi4, qwen25_14b_it, qwen25_32b_it.

## H5 (depth transition from stimulus-bound to goal-relative valence)
Confirmation domains, sst2/diffmeans, self perspective. Appraisal share r(L) = d_out(L) / (|d_out(L)| + |d_surf(L)|).
L_s = layer of maximal d_surf (NaN-safe); L_f = final layer.
H5a: r(L_f) > r(L_s).   H5b: r(L_f) > 0.5.
Evaluated per model on the 8 not-yet-inspected models; confirmatory summary = one-sided sign test over instruct
models with behaviour accuracy >= 0.85 (base models reported separately, exploratory).
The same quantities for the logistic and logistic_pca50 estimators are reported as robustness.
