# Pre-registration v8 amendment (2026-10-06, before any sweep3 output exists)

Motivation: external review. Seen so far: all H1-H12 results, other-perspective band results (computed from saved
outputs before this registration), and the lexical-prior regression on saved behaviour.

## H13 Goal-relevant but non-valenced control (make_goalneutral_stimuli.py)
"... were each given a key, one brass and one steel; only the brass key opens the <object>. <stake>. <B> received the
brass/steel key." The word is lexically neutral; the outcome (stakeholder got the goal-serving key) has the same XOR
structure. Read with (a) the SST-2 valence direction (in-run, same procedure) and (b) an in-format brass/steel direction
(discovery stake-free items). Confirmation self items, late band [0.60, 0.90], response position.
H13a (appraisal without valenced words): SST-2 valence conflict-cell AUC for the outcome > 0.5 with the stimulus-
bootstrap 95% CI excluding 0.5, in the majority of eligible instruct models.
H13b (descriptive): whether the in-format brass/steel direction crosses 0.5 (rebinding of a goal-relevant neutral word).
Behaviour: accuracy and the effect of the word "brass" controlling for the outcome.

## H14 Position trajectory
Valence stimuli with the evaluative question; in-format directions at three positions: end of stimulus text, end of
question, response position (last token). Late-band conflict-cell AUC per position; prediction (directional, reported
regardless): increases from end of stimulus to response position in the majority of eligible models.

## H15 Token x layer patching map (Qwen2.5-7B, Mistral-7B, Qwen2.5-14B)
Positions: stake label, stake-sentence end, event label, event verb, event end, response position; 80 pairs/model.
Descriptive: depth profile of each position's normalised effect.

## H16 Base versus instruct, matched plain continuation ("For you, this outcome was")
Qwen2.5-7B, OLMo-2-7B base and instruct. In-format direction at the continuation position; late-band conflict AUC.
Descriptive comparison (base vs instruct); behaviour " good" vs " bad".

## H17 Multi-LLM validation of the naturalistic sentences (replacing human validation, which the lab cannot run)
Each eligible instruct model answers which party won; agreement with the generation label is reported per model.
