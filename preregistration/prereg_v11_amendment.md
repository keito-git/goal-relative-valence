# Pre-registration v11 amendment (2026-10-07): robustness analyses

These analyses were specified after all main-paper results (H1-H22) were known. They are robustness checks, not
independent confirmatory evidence, and every result will be reported in the appendix whatever its direction.
Nothing below has been computed.

## R1 Band-boundary sensitivity (band_sensitivity.py; saved per-layer profiles)
Middle-band upper edge in {0.35, 0.40, 0.45, 0.50}, late-band lower edge in {0.50, 0.55, 0.60, 0.65} (middle lower edge
0.20, late upper edge 0.90 fixed; combinations with an empty or overlapping band skipped). For each combination and
both readout positions (SST-2 directions, confirmation, self): late-minus-middle conflict-cell AUC per model; report the
model mean and the number of models with a positive difference. Robust if the difference is positive in every eligible
model for every combination at the response position.

## R2 Per-domain and leave-one-domain-out robustness (domain_robust.py)
Confirmation domains (lottery, horse race, lawsuit, hiring), self items. End of text: SST-2 directions from saved
states. Response position: SST-2 directions from re-extracted projections (sweep6.py, part goalproj; forward passes
only). Per model x domain: middle- and late-band conflict-cell AUC and late-minus-middle; B_surf per domain from saved
behaviour. Leave-one-domain-out: the same summaries with one confirmation domain removed. Robust if late > middle in
every model for every held-out set at the response position.

## R3 Naturalistic surface-polarity relabelling (natural_relabel.py)
Relabel the 484 naturalistic sentences with two further sentiment classifiers (DistilBERT fine-tuned on SST-2;
cardiffnlp/twitter-roberta-base-sentiment-latest, positive vs negative probability). Re-run the naturalistic analysis
(middle/late conflict-cell AUC at both positions) (a) with each classifier's labels and (b) on the subset where all three
classifiers agree. Report all.

## R4 Threshold sensitivity (threshold_sensitivity.py; saved summaries)
Behavioural eligibility threshold in {0.80, 0.85, 0.90}; validity gate in {0.75, 0.80, 0.85}. Recompute the main counts
(late > middle at both positions; B_surf > 0; control specificity; task gating; goal-relevant key control) under each.

## R5 Family-level summary
Average the three Qwen2.5 models within family and report the main directional results over five families.

## R6 Patching placebo (sweep6.py, part placebo)
For the 7 eligible instruct models, 80 pairs each: patch activations from an item that differs only in the order of the
two parties in the first sentence (same stake, same outcome, same event) at the stake-label tokens and at the response
position, layer by layer, and compare the normalised effect with the stake-flip patching. Expected near 0 everywhere.
