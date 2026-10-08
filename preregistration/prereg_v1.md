# Pre-registration v1: Is LLM internal valence goal-relative or stimulus-bound?

Committed 2026-10-05 (13:51 UTC) before any activation of any model on these stimuli is extracted.
Code: make_stimuli.py, extract.py, analyze.py committed alongside this file.

## Question
When the event text is held fixed, does the model's general valence code (estimated from
stake-free sentiment text) track the outcome *for the stakeholder* (goal-relative, appraisal-like)
or the surface polarity of the event wording (stimulus-bound)?

## Stimuli (make_stimuli.py, seed 20261006)
Context with two neutral labels (order counterbalanced) + stake sentence binding a stakeholder to
one label + event sentence in which label B either won/was approved/... (surface = positive) or
lost/was rejected/... (surface = negative). match = (B is the stakeholder's label).
outcome = 1 iff surface == match (XOR structure). Surface and outcome are orthogonal (balanced 2x2);
every label token appears equally often in every outcome cell, so outcome has no bag-of-words cue.
Perspectives: self ("You ..."), other (named third party), none (no stake sentence).
Domains: discovery = sports, election, auction, chess, grant; confirmation (held out, not analysed
until the discovery decision is recorded) = lottery, race, court, hiring.

## Valence sources (stake-free)
SST-2 (1,000 positive + 1,000 negative, >= 6 words) primary; GoEmotions positive vs negative
emotions (1,000 + 1,000) secondary. Directions per layer: diff-of-means (primary), L2 logistic,
logistic on PCA-50. Floor: 500 uniform random directions in the top-50 PC span of the source
activations at that layer (in-plane random floor).

## Representation
Primary: hidden state at the last token of the plain text ("raw"), all layer outputs incl. embeddings.
Secondary (instruct models): last token of the chat-templated prompt with generation header ("gen").

## Metrics (self perspective unless stated)
d_outcome = Cohen's d of the projection between outcome levels, averaged within surface levels.
d_surface = Cohen's d between surface levels, averaged within outcome levels.
auc_outcome_conflict = AUC of the projection for outcome on cells where surface != outcome.

## Kill switch (discovery model Qwen/Qwen2.5-7B-Instruct, discovery domains)
K0 lexical leak: leave-one-domain-out (LODO) outcome probe at layer 0 <= 0.60.
   If violated -> stimuli are revised (not a topic kill), and this prereg is amended before rerun.
K1 behaviour: good/bad answer accuracy on self items >= 0.85. If violated, Qwen2.5-14B-Instruct is
   tried as discovery model; if no model reaches 0.85 -> KILL.
K2 the model encodes outcome: LODO outcome probe >= 0.75 at some layer >= 1. Else -> KILL.
K3 the transferred direction is a valid valence readout on this stimulus format: sst2/diffmeans
   auc_surface_none >= 0.70 at some layer. Else -> KILL (measurement invalid).
No outcome-direction gate: both results (goal-relative or stimulus-bound) are reportable.

## Confirmatory analysis (after kill switch passes)
Layer L* per model := layer maximising sst2/diffmeans d_outcome_self on DISCOVERY domains.
On CONFIRMATION domains at L*:
H1 (goal-relative component): d_outcome_self > 0 and |d_outcome_self| > in-plane floor p95.
H2 (dominance): |d_outcome_self| > |d_surface_self|.
Reading: H1 false -> stimulus-bound valence; H1 true & H2 true -> predominantly goal-relative;
H1 true & H2 false -> mixed code.
Models (confirmation): OLMo-2-1124-7B-Instruct, Mistral-7B-Instruct-v0.3, phi-4, Falcon3-7B-Instruct,
granite-3.1-8b-instruct, Qwen2.5-14B-Instruct, Qwen2.5-32B-Instruct. Base-vs-instruct (Qwen2.5-7B,
OLMo-2-1124-7B) and self-vs-other are exploratory. A model with behaviour accuracy < 0.85 is reported
but excluded from H1/H2 summaries. Robustness: all three direction estimators and the GoEmotions source
are reported; a claim is stated only if the sign of d_outcome agrees across the three estimators.
