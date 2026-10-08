# Pre-registration v2 amendment (2026-10-05 ~14:40 UTC), before any ablation is run and before any
behavioural data on CONFIRMATION domains or on models other than qwen25_7b_it / olmo2_7b_it is inspected.

## What has been seen (exploratory, discovery domains only)
Behavioural interference: accuracy congruent vs conflict cells (surface == outcome vs !=):
qwen25_7b_it self 0.997 vs 0.857, other 0.998 vs 0.743; olmo2_7b_it self 0.993 vs 0.655, other 0.923 vs 0.542.

## H3 (behavioural interference; confirmatory on CONFIRMATION domains, all models)
Per model: B_surf = mean(good-bad logit | surface=1) - mean(... | surface=0), averaged within outcome levels.
H3: B_surf > 0 and conflict accuracy < congruent accuracy (self perspective), in the majority of models with
behaviour accuracy >= 0.85; sign test across models (one-sided).

## H4 (causal mediation by the stake-free valence code) — ablate.py
Directional ablation of the SST-2 diff-of-means direction (per layer, estimated at the chat-generation position)
from decoder-layer outputs at all positions, in two bands: all layers, and mid band [nL/4, 3nL/4).
Controls: 20 in-plane random directions (top-50 PC span of SST-2 activations per layer); GoEmotions direction
as replication.
Metrics on self items, CONFIRMATION domains: B_surf, B_out (outcome effect within surface levels),
interference ratio R = B_surf / B_out, conflict accuracy.
H4: ablating the valence direction reduces R by more than the 95th percentile of the reduction under in-plane
random ablation (one-sided, per band). Primary band = mid. Reported regardless of direction.
Reading: H4 true -> the stimulus-bound valence code causally carries the interference; H4 false -> the
interference does not route through this code.
Models for H4: qwen25_7b_it (also used for discovery), olmo2_7b_it, mistral_7b_it; others if time allows.
