# Pre-registration v7 amendment (2026-10-06 ~05:20 UTC, before any in-format projection is computed)

Motivation (internal review): the control directions were estimated from out-of-format source
sentences and are weak readouts at the response position (stake-free control AUC: location 0.61-0.89, material
0.42-0.59), so the absence of stakeholder rebinding for the controls cannot show specificity. Seen so far: all
H1-H11 results.

## H12 In-format directions with a validity gate
For each task (valence = win/loss stimuli, location, material), at each layer and position (raw, gen), estimate a
difference-of-means direction for the SURFACE label from stake-free ('none') items of the DISCOVERY domains only;
project all items. Evaluate on CONFIRMATION domains:
- validity: AUC of the projection for surface polarity on confirmation 'none' items, late band [0.60, 0.90];
- conflict-cell AUC for the stakeholder outcome on confirmation self items, mid and late bands (as before).
Validity gate: a task passes in a model/position if its late-band validity AUC >= 0.80.
Reading at the response position, decided now:
- SPECIFIC REBINDING if, among eligible instruct models where valence and the control both pass the gate, valence's
  late-band conflict AUC exceeds the control's in the majority AND the control's late-band conflict AUC <= 0.60,
  separately for each control.
- If fewer than 4 eligible models pass the gate for a control, the comparison for that control is INCONCLUSIVE.
The same quantities are reported at the end of the text. Run for all 8 instruct models (OLMo excluded from counts).


Correction (2026-10-06): the header time "~05:20 UTC" is wrong; the commit time (a2f1460, 05:08 UTC) is authoritative. No in-format projection existed before that commit.
