"""R4/R5 (prereg v11): sensitivity of the main counts to the eligibility threshold and the validity gate, and a
family-level summary. Writes results/threshold_sensitivity.json and outputs/table_thresholds.tex."""
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
(BASE / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
RES = BASE / "results"
INSTRUCT = ["qwen25_7b_it", "mistral_7b_it", "falcon3_7b_it", "granite31_8b_it", "olmo2_7b_it", "phi4", "qwen25_14b_it", "qwen25_32b_it"]
FAMILY = {"qwen25_7b_it": "Qwen2.5", "qwen25_14b_it": "Qwen2.5", "qwen25_32b_it": "Qwen2.5", "mistral_7b_it": "Mistral",
          "falcon3_7b_it": "Falcon3", "granite31_8b_it": "Granite", "olmo2_7b_it": "OLMo-2", "phi4": "Phi-4"}
S = json.load(open(RES / "confirmatory_summary.json"))
I = json.load(open(RES / "informat_summary.json"))
W4 = json.load(open(RES / "sweep4_summary.json"))
T3 = json.load(open(RES / "sweep3_summary.json"))
BS = json.load(open(RES / "band_sensitivity.json"))
key = "0.45_0.60"
dgen = BS["gen"][key]["per_model"]


def late_gen(t):
    L = json.load(open(RES / f"{t}_confirmation_gen.json"))["layers"]
    p = np.array([l["sst2/diffmeans"]["auc_outcome_conflict_self"] for l in L], float); n = len(p) - 1
    return float(np.nanmean([p[k] for k in range(n) if 0.6 <= k / n <= 0.9]))


def gating_ok(t, v):
    r = W4[t]
    ev = [r[f"tg_{k}"]["late"] for k in ("eval_self", "eval_plain") if r[f"tg_{k}"]["validity_late"] >= v]
    ne = [r[f"tg_{k}"]["late"] for k in ("factual", "irrelevant") if r[f"tg_{k}"]["validity_late"] >= v]
    if not ev or not ne:
        return None
    return all(x <= 0.60 for x in ne) and all(max(ev) - x >= 0.10 for x in ne)


rows, out = [], {}
for e in (0.80, 0.85, 0.90):
    elig = [t for t in INSTRUCT if S[t]["behaviour_conf"]["self"]["acc"] >= e]
    for v in (0.75, 0.80, 0.85):
        r = {"n": len(elig),
             "late_gt_mid": sum(dgen[t] > 0 for t in elig),
             "late_gt_half": sum(late_gen(t) > 0.5 for t in elig),
             "bsurf_pos": sum(S[t]["behaviour_conf"]["self"]["B_surf"] > 0 for t in elig)}
        for ctrl in ("location", "material"):
            g = [t for t in elig if I[t]["valence_gen"]["validity_late"] >= v and I[t][f"{ctrl}_gen"]["validity_late"] >= v]
            r[f"spec_{ctrl}"] = f"{sum(I[t]['valence_gen']['late'] > I[t][f'{ctrl}_gen']['late'] and I[t][f'{ctrl}_gen']['late'] <= 0.60 for t in g)}/{len(g)}"
        gs = [gating_ok(t, v) for t in elig]
        r["gating"] = f"{sum(x is True for x in gs)}/{sum(x is not None for x in gs)}"
        r["key"] = sum(T3[t]["gn_sst_gen"]["late_ci"][0] > 0.5 for t in elig)
        out[f"{e:.2f}_{v:.2f}"] = r
        rows.append(f"{e:.2f} & {v:.2f} & {r['n']} & {r['late_gt_mid']} & {r['late_gt_half']} & {r['bsurf_pos']} & "
                    f"{r['spec_location']} & {r['spec_material']} & {r['gating']} & {r['key']} \\\\")
# family-level summary (Qwen2.5 sizes averaged)
fam = {}
for t in INSTRUCT:
    if S[t]["behaviour_conf"]["self"]["acc"] < 0.85:
        continue
    fam.setdefault(FAMILY[t], []).append(t)
fs = {f: {"late_minus_mid_gen": float(np.mean([dgen[t] for t in ts])), "late_gen": float(np.mean([late_gen(t) for t in ts])),
          "bsurf": float(np.mean([S[t]["behaviour_conf"]["self"]["B_surf"] for t in ts])),
          "tg_eval": float(np.mean([W4[t]["tg_eval_self"]["late"] for t in ts])),
          "tg_nonstake": float(np.mean([max(W4[t]["tg_factual"]["late"], W4[t]["tg_irrelevant"]["late"]) for t in ts])),
          "key": float(np.mean([T3[t]["gn_sst_gen"]["late"] for t in ts]))} for f, ts in fam.items()}
out["family"] = fs
json.dump(out, open(RES / "threshold_sensitivity.json", "w"), indent=1)
tab = [r"\begin{tabular}{cccccccccc}", r"\toprule",
       r"Elig. & Gate & $n$ & late$>$mid & late$>$0.5 & $B_{\mathrm{surf}}>0$ & loc. & mat. & gating & key \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
(BASE / "outputs" / "table_thresholds.tex").write_text("\n".join(tab) + "\n")
print("\n".join(rows)); print(json.dumps(fs, indent=0))
