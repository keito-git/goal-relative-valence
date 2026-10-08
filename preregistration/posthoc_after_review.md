# Post-hoc analyses added after internal review (2026-10-05, ~17:00-17:30 UTC) — NOT pre-registered

(Correction: an earlier commit 8abccee intended to add this file and three scripts, but the shell chain stopped
after a failed copy; only item 4 was recorded. This commit restores the full record. Timing of each analysis is
as described below; all were run after the confirmatory results H1-H5 were known.)

Motivation (internal review): (i) H5b rests on the final layer, where the SST-2 direction separates
stake-free wording less well (auc_surface_none 0.48-0.74); (ii) H5a is near-guaranteed by defining L_s as the argmax
of d_surf; (iii) Qwen2.5-7B-Instruct and OLMo-2-7B-Instruct profiles had been seen before prereg v3.

1. sst_validity.py: split-half AUC of the SST-2 direction on held-out SST-2 sentences, per layer.
2. cross_source.py: GoEmotions direction -> SST-2 and SST-2 direction -> GoEmotions AUC, per layer.
3. band_boot.py: fixed relative-depth bands (mid [0.20,0.45], late [0.60,0.90], final layer excluded); conflict-cell
   AUC and appraisal share, 1,000 item-level bootstrap resamples; raw and gen positions. Band edges were chosen by
   the author after seeing Figure 1 profiles of seen models; they are reported as a sensitivity analysis.
4. clf_baseline.py: off-the-shelf sentiment classifiers on confirmation self items.
5. abl_acc.py: exploratory conflict-cell accuracy under ablation (valence vs 20 in-plane random directions).
Corrected confirmatory count for H5 restricted to models unseen at v3: mistral, falcon, granite, phi4,
qwen 14B, qwen 32B (instruct, behaviour >= 0.85).
