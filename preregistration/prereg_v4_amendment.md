# Pre-registration v4 amendment (2026-10-06, before any sweep2 output exists)

Motivation: external advice + internal review — rule out "generic lexical-to-relational contextualisation",
quantify the transition, test prompt robustness, and add a causal information-flow analysis.
Seen so far: all valence results reported in the draft (10 models). Nothing of the control task, the extra prompts,
or patching has been computed.

## H6 Non-affective relational control (make_control_stimuli.py; same XOR structure with left/right)
Direction: difference of means between stake-free "left" and "right" sentences (control_source.jsonl), per layer and
position. Metric: conflict-cell AUC for o' = "the stakeholder's object is on the left" (self, confirmation domains),
mid band [0.20,0.45] and late band [0.60,0.90] (final layer excluded), raw and gen positions; transition depth d*
from H7. Readings, decided now:
- GENERIC if, in the majority of the 7 eligible instruct models (behaviour >= 0.85 on the valence task), the control
  shows mid-band AUC <= 0.15 AND a late-minus-mid increase whose stimulus-bootstrap 95% CI excludes 0 AND
  |d*_control - d*_valence| <= 0.10 (relative depth, same model and position).
  -> claim: appraisal inherits a general lexical-to-relational computation.
- SPECIFIC otherwise; report which criterion fails and in which direction.
Behaviour: left/right answer; B_surf' (effect of the word "left" controlling for o') reported alongside valence B_surf.

## H7 Transition depth
For each model, position, and task: fit AUC(d) = lo + (hi - lo) / (1 + exp(-k (d - d*))) by least squares to the
conflict-cell AUC from the layer of minimum AUC to the last non-final layer (d = relative depth). Report d*, k, hi;
fits with R^2 < 0.8 are reported but flagged.

## H8 Prompt robustness
5 question templates x 2 option orders (sweep2.TEMPLATES), confirmation self items. For each template/order:
B_surf > 0 counted over the 7 eligible models. Claim of robustness if B_surf > 0 in >= 6/7 models for every
template/order combination; otherwise report the exceptions.

## H9 Activation patching (information flow)
Clean/corrupt pairs differ only in the stake sentence label (outcome flips; surface identical). For each layer i,
patch the corrupt run's layer-i output into the clean run (a) at the stake-label token positions, (b) at the final
(response) position. Normalised effect E = (g_patched - g_clean) / (g_corrupt - g_clean), averaged over pairs with
|g_corrupt - g_clean| > 1. Crossing depth = first relative depth where E_final > E_label.
Prediction (directional, reported regardless): the crossing depth lies within +-0.15 of d*_valence (gen position)
in the majority of eligible models.
