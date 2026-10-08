"""Appendix tables and macros for the robustness analyses (prereg v11)."""
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
(BASE / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
RES = BASE / "results"
OUT = BASE / "outputs"
LAB = {"qwen25_7b_it": "Qwen2.5-7B", "mistral_7b_it": "Mistral-7B", "falcon3_7b_it": "Falcon3-7B", "granite31_8b_it": "Granite-3.1-8B",
       "olmo2_7b_it": "OLMo-2-7B$^\\dagger$", "phi4": "Phi-4", "qwen25_14b_it": "Qwen2.5-14B", "qwen25_32b_it": "Qwen2.5-32B"}


def table_band():
    B = json.load(open(RES / "band_sensitivity.json"))
    lines = [r"\begin{tabular}{l" + "c" * len(B["late_lo"]) + "}", r"\toprule",
             "Middle upper edge & " + " & ".join(f"late from {x:.2f}" for x in B["late_lo"]) + r" \\", r"\midrule"]
    for pos, name in (("raw", "End of text"), ("gen", "Response position")):
        lines.append(rf"\multicolumn{{{1 + len(B['late_lo'])}}}{{l}}{{\textit{{{name}}}}} \\")
        for mh in B["mid_hi"]:
            cells = []
            for ll in B["late_lo"]:
                g = B[pos].get(f"{mh:.2f}_{ll:.2f}")
                cells.append("--" if g is None else f"{g['mean_elig']:.2f} ({g['n_pos_elig']}/{g['n_elig']})")
            lines.append(f"{mh:.2f} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_band.tex").write_text("\n".join(lines) + "\n")


def table_domain():
    f = RES / "domain_robust.json"
    if not f.exists():
        return
    D = json.load(open(f)); doms = ["lottery", "race", "court", "hiring"]
    names = {"lottery": "lottery", "race": "horse race", "court": "lawsuit", "hiring": "hiring"}
    lines = [r"\begin{tabular}{l" + "c" * 8 + "}", r"\toprule",
             r" & \multicolumn{4}{c}{late $-$ middle, response position} & \multicolumn{4}{c}{$B_{\mathrm{surf}}$} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             "Model & " + " & ".join(names[d] for d in doms) + " & " + " & ".join(names[d] for d in doms) + r" \\", r"\midrule"]
    for t, lab in LAB.items():
        if t not in D or f"gen_dom_{doms[0]}" not in D[t]:
            continue
        r = D[t]
        lines.append(lab + " & " + " & ".join(f"{r[f'gen_dom_{d}']['diff']:+.2f}" for d in doms) + " & "
                     + " & ".join(f"{r[f'bsurf_dom_{d}']:+.2f}" for d in doms) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_domain.tex").write_text("\n".join(lines) + "\n")


def table_natural():
    f = RES / "natural_relabel.json"
    if not f.exists():
        return
    N = json.load(open(f))
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{DistilBERT} & \multicolumn{2}{c}{Twitter-RoBERTa} & \multicolumn{2}{c}{all agree} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Model & mid & late & mid & late & mid & late \\", r"\midrule"]
    for t, lab in LAB.items():
        if t not in N:
            continue
        r = N[t]
        lines.append(lab + " & " + " & ".join(f"{r[f'{k}_gen'][b]:.2f}" for k in ("distilbert", "twitter_roberta", "agree") for b in ("mid", "late")) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_natrelabel.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    table_band(); table_domain(); table_natural()
    print(sorted(p.name for p in OUT.glob("table_*.tex")))


def fig_placebo():
    """Mean normalised patching effect across eligible models: stake flip vs order-swap placebo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"], "mathtext.fontset": "cm",
                         "pdf.fonttype": 42, "font.size": 7})
    d = RES / "placebo"
    files = sorted(d.glob("*.json")) if d.exists() else []
    if not files:
        return None
    S = json.load(open(RES / "confirmatory_summary.json"))
    grid = np.linspace(0.02, 1.0, 40); curves = {}; summ = {}
    for f in files:
        t = f.stem
        if S[t]["behaviour_conf"]["self"]["acc"] < 0.85:
            continue
        P = json.load(open(f))
        summ[t] = {"n": len(P)}
        for k in ("flip_label", "flip_final", "placebo_label", "placebo_final"):
            M = np.mean([p["effects"][k] for p in P], 0); x = (np.arange(len(M)) + 1) / len(M)
            curves.setdefault(k, []).append(np.interp(grid, x, M))
            summ[t][k + "_maxabs"] = float(np.max(np.abs(M)))
    fig, ax = plt.subplots(figsize=(3.1, 1.9))
    sty = {"flip_label": ("#0072B2", "--", "stake flip, label"), "flip_final": ("#0072B2", "-", "stake flip, response"),
           "placebo_label": ("#999999", "--", "placebo, label"), "placebo_final": ("#999999", "-", "placebo, response")}
    for k, (c, ls, lab) in sty.items():
        Y = np.array(curves[k]); ax.plot(grid, Y.mean(0), color=c, ls=ls, lw=1.1, label=lab)
        ax.fill_between(grid, Y.min(0), Y.max(0), color=c, alpha=0.12, lw=0)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("relative depth"); ax.set_ylabel("normalised patching effect")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=5.5, loc="center right")
    fig.savefig(BASE / "outputs" / "figures" / "fig_placebo.pdf", bbox_inches="tight"); plt.close(fig)
    json.dump(summ, open(RES / "placebo_summary.json", "w"), indent=1)
    return summ


def macros():
    out = []
    mac = lambda k, v, f="{:.2f}": out.append(f"\\newcommand{{\\{k}}}{{{f.format(v)}}}")
    S = json.load(open(RES / "confirmatory_summary.json"))
    el = [t for t in LAB if S[t]["behaviour_conf"]["self"]["acc"] >= 0.85]
    B = json.load(open(RES / "band_sensitivity.json"))
    for pos, a in (("raw", "Raw"), ("gen", "Gen")):
        g = B[pos]
        mac(f"bandN{a}", len(g), "{:d}")
        mac(f"bandAllPos{a}", sum(v["n_pos_elig"] == v["n_elig"] for v in g.values()), "{:d}")
        mac(f"bandMin{a}", min(v["mean_elig"] for v in g.values())), mac(f"bandMax{a}", max(v["mean_elig"] for v in g.values()))
    D = json.load(open(RES / "domain_robust.json")); doms = ["lottery", "race", "court", "hiring"]
    for pos, a in (("raw", "Raw"), ("gen", "Gen")):
        v = [D[t][f"{pos}_dom_{d}"]["diff"] for t in el for d in doms]
        mac(f"domPos{a}", sum(x > 0 for x in v), "{:d}"), mac(f"domN{a}", len(v), "{:d}"), mac(f"domMin{a}", min(v))
        v = [D[t][f"{pos}_lodo_{d}"]["diff"] for t in el for d in doms]
        mac(f"lodoPos{a}", sum(x > 0 for x in v), "{:d}"), mac(f"lodoMin{a}", min(v))
    v = [D[t][f"{pos}_lodo_{d}"]["late"] for t in el for d in doms for pos in ("gen",)]
    mac("lodoLateHalfGen", sum(x > 0.5 for x in v), "{:d}")
    v = [D[t][f"bsurf_dom_{d}"] for t in el for d in doms]
    mac("domBsurfPos", sum(x > 0 for x in v), "{:d}"), mac("domBsurfMin", min(v), "{:.2f}")
    N = json.load(open(RES / "natural_relabel.json"))
    mac("natAgreeN", N["n_agree"], "{:d}"), mac("natSentN", N["n_sentences"], "{:d}")
    mac("natAgreeDistil", N["agreement"]["distilbert"]), mac("natAgreeTwitter", N["agreement"]["twitter_roberta"])
    for k, a in (("distilbert", "Distil"), ("twitter_roberta", "Twitter"), ("agree", "Agree")):
        mac(f"nat{a}CIPos", sum(N[t][f"{k}_{p}"]["diff_ci"][0] > 0 for t in el for p in ("raw", "gen")), "{:d}")
        lg = [N[t][f"{k}_gen"]["late"] for t in el]
        mac(f"nat{a}LateMin", min(lg)), mac(f"nat{a}LateMax", max(lg)), mac(f"nat{a}LateHalf", sum(x > 0.5 for x in lg), "{:d}")
    mac("natCIN", 2 * len(el), "{:d}")
    P = json.load(open(RES / "placebo_summary.json"))
    pl = [max(P[t]["placebo_label_maxabs"], P[t]["placebo_final_maxabs"]) for t in P]
    fl = [min(P[t]["flip_label_maxabs"], P[t]["flip_final_maxabs"]) for t in P]
    mac("placeboMaxMin", min(pl)), mac("placeboMaxMax", max(pl)), mac("flipMaxMin", min(fl)), mac("nPlacebo", len(P), "{:d}")
    T = json.load(open(RES / "threshold_sensitivity.json"))["family"]
    mac("nFam", len(T), "{:d}")
    mac("famKeyAbove", sum(v["key"] > 0.5 for v in T.values()), "{:d}")
    mac("famEvalAbove", sum(v["tg_eval"] > v["tg_nonstake"] + 0.1 for v in T.values()), "{:d}")
    mac("famLateAbove", sum(v["late_gen"] > 0.5 and v["late_minus_mid_gen"] > 0 for v in T.values()), "{:d}")
    mac("famBsurfPos", sum(v["bsurf"] > 0 for v in T.values()), "{:d}")
    k = sum(v["bsurf"] > 0 for v in T.values()); n = len(T)
    from math import comb
    mac("famBsurfP", sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n, "{:.3f}")
    (OUT / "numbers_robust.tex").write_text("\n".join(out) + "\n")
    print("\n".join(out))
