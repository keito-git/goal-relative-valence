# Pre-registration v5 amendment (2026-10-06, before any natural-set activation or behaviour is computed)

Natural counterfactual set: 502 news-style two-party outcome sentences generated with gpt-4.1-mini (gen_natural.py,
no emotion words, half winner-subject / half loser-subject), kept if RoBERTa-large-MNLI entails "<winner> won."
(p > 0.8) and does not entail "<loser> won." (p < 0.2): 484 kept. Surface polarity s := SiEBERT positive
probability > 0.5 (an operational lexical/surface label). Self items: "You supported X. <sentence>" for both parties
(968 items, 50% conflict cells); stake-free items: the sentence alone.

## H10 Natural replication
Valence direction: SST-2 difference of means (as before), raw and gen positions; metric: conflict-cell AUC for the
outcome; bands as before; stimulus bootstrap (1,000).
REPLICATES if, in the majority of eligible instruct models, (i) the late-minus-mid increase has a 95% CI excluding 0
and (ii) the mid-band AUC is below 0.5. Behaviour: B_surf > 0 (surface = SiEBERT label, controlling for outcome)
counted over eligible models. Reported regardless of outcome. Qwen2.5-7B-Instruct and Mistral-7B-Instruct are run in
a separate pass because their sweep started before this part was added.
