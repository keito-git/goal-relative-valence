"""Figures and LaTeX number macros for the goal-relative valence paper (reads results/*.json)."""
import json
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results"
FIG = BASE / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

INSTRUCT = [("qwen25_7b_it", "Qwen2.5-7B-Inst."), ("mistral_7b_it", "Mistral-7B-Inst."),
            ("falcon3_7b_it", "Falcon3-7B-Inst."), ("granite31_8b_it", "Granite-3.1-8B-Inst."),
            ("olmo2_7b_it", "OLMo-2-7B-Inst."), ("phi4", "Phi-4 (14B)"),
            ("qwen25_14b_it", "Qwen2.5-14B-Inst."), ("qwen25_32b_it", "Qwen2.5-32B-Inst.")]
BASEM = [("qwen25_7b_base", "Qwen2.5-7B (base)"), ("olmo2_7b_base", "OLMo-2-7B (base)")]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#999999"]


def load(tag, split="confirmation", pos="raw"):
    f = RES / f"{tag}_{split}_{pos}.json"
    return json.load(open(f)) if f.exists() else None


def profile(res, est="sst2/diffmeans"):
    L = res["layers"]
    n = len(L) - 1
    x = np.array([l["layer"] / n for l in L])
    do = np.array([l[est]["d_outcome_self"] for l in L], dtype=float)
    ds = np.array([l[est]["d_surface_self"] for l in L], dtype=float)
    fl = np.array([l.get(est.split("/")[0] + "/null_absd_outcome_self_p95", np.nan) for l in L], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = do / (np.abs(do) + np.abs(ds))
    return x, do, ds, fl, r


R_MIN_DENOM = 0.2  # r is undefined when both effects are ~0; such layers are not drawn (stated in the text)


def fig_depth(pos="raw", est="sst2/diffmeans", name="fig_depth"):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.3), gridspec_kw={"wspace": 0.28})
    grid = np.linspace(0, 1, 41)
    D_out, D_surf = [], []
    for (tag, lab), c in zip(INSTRUCT, COLORS):
        res = load(tag, pos=pos)
        if res is None:
            continue
        x, do, ds, fl, r = profile(res, est)
        D_out.append(np.interp(grid, x, np.nan_to_num(do)))
        D_surf.append(np.interp(grid, x, np.nan_to_num(ds)))
        r = np.where(np.abs(do) + np.abs(ds) < R_MIN_DENOM, np.nan, r)
        axes[1].plot(x, r, color=c, lw=1.1, label=lab)
    for D, col, lab, ls in ((np.array(D_surf), "#D55E00", "surface wording ($d_{\\mathrm{surf}}$)", "--"),
                            (np.array(D_out), "#0072B2", "stakeholder outcome ($d_{\\mathrm{out}}$)", "-")):
        axes[0].fill_between(grid, D.min(0), D.max(0), color=col, alpha=0.18, lw=0)
        axes[0].plot(grid, D.mean(0), color=col, lw=1.4, ls=ls, label=lab)
    axes[0].set_xlabel("relative depth")
    axes[0].set_ylabel("Cohen's $d$ on valence projection")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].axhline(0.5, color="gray", lw=0.6, ls=":")
    axes[1].set_xlabel("relative depth")
    axes[1].set_ylabel("appraisal share $r$")
    axes[1].set_ylim(-0.1, 1.0)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.25), handlelength=1.4,
               columnspacing=1.0)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_behaviour():
    tags = [t for t in INSTRUCT + BASEM if (RES / "confirmatory_summary.json").exists()]
    S = json.load(open(RES / "confirmatory_summary.json"))
    rows = [(lab, S[t]["behaviour_conf"]["self"]) for t, lab in INSTRUCT + BASEM if t in S]
    fig, ax = plt.subplots(figsize=(3.1, 2.2))
    y = np.arange(len(rows))
    ax.barh(y + 0.18, [b["acc_congruent"] for _, b in rows], height=0.34, color="#BBBBBB", label="congruent")
    ax.barh(y - 0.18, [b["acc_conflict"] for _, b in rows], height=0.34, color="#D55E00", label="conflict")
    ax.set_yticks(y, [lab for lab, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("appraisal accuracy (self, held-out domains)")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / "fig_behaviour.pdf", bbox_inches="tight")
    plt.close(fig)
    return tags


def numbers():
    """Emit \\newcommand macros so that every number in the text comes from the result files."""
    S = json.load(open(RES / "confirmatory_summary.json"))
    H5 = json.load(open(RES / "h5_summary.json")) if (RES / "h5_summary.json").exists() else {}
    out = []

    def mac(name, val, fmt="{:.2f}"):
        out.append(f"\\newcommand{{\\{name}}}{{{fmt.format(val)}}}")

    abbrev = {"qwen25_7b_it": "QwenSeven", "mistral_7b_it": "Mistral", "falcon3_7b_it": "Falcon",
              "granite31_8b_it": "Granite", "olmo2_7b_it": "Olmo", "phi4": "Phi", "qwen25_14b_it": "QwenFourteen",
              "qwen25_32b_it": "QwenThirtyTwo", "qwen25_7b_base": "QwenSevenBase", "olmo2_7b_base": "OlmoBase"}
    for t, a in abbrev.items():
        if t in S:
            b = S[t]["behaviour_conf"]["self"]
            mac(f"acc{a}", b["acc"], "{:.3f}")
            mac(f"accCong{a}", b["acc_congruent"], "{:.3f}")
            mac(f"accConf{a}", b["acc_conflict"], "{:.3f}")
            mac(f"bsurf{a}", b["B_surf"])
            mac(f"bout{a}", b["B_out"])
            mac(f"dOut{a}", S[t]["diffmeans"]["d_outcome_self"])
            mac(f"dSurf{a}", S[t]["diffmeans"]["d_surface_self"])
            mac(f"floor{a}", S[t]["floor_p95"])
            mac(f"Lstar{a}", S[t]["L*"], "{:d}")
        if t in H5:
            d = H5[t]["diffmeans"]
            mac(f"rLs{a}", d["r_Ls"])
            mac(f"rLf{a}", d["r_Lf"])
    # cross-model counts over instruct models; eligibility = behaviour accuracy >= 0.85 (prereg v1)
    inst = [t for t, _ in INSTRUCT if t in S and t in H5]
    elig = [t for t in inst if S[t]["behaviour_conf"]["self"]["acc"] >= 0.85]
    mac("nInstruct", len(inst), "{:d}")
    mac("nEligible", len(elig), "{:d}")
    mac("nHOne", sum(S[t]["H1"] for t in elig), "{:d}")
    mac("nHTwo", sum(S[t]["H2"] for t in elig), "{:d}")
    mac("nAgree", sum(S[t]["sign_agree_estimators"] for t in elig), "{:d}")
    mac("nHThree", sum(S[t]["H3"] for t in elig), "{:d}")
    mac("nBsurfPos", sum(S[t]["behaviour_conf"]["self"]["B_surf"] > 0 for t in S), "{:d}")
    mac("nAllModels", len(S), "{:d}")
    mac("nHFiveA", sum(H5[t]["diffmeans"]["H5a"] for t in elig), "{:d}")
    mac("nHFiveB", sum(H5[t]["diffmeans"]["H5b"] for t in elig), "{:d}")
    for est, a in (("logistic", "Logit"), ("logistic_pca50", "Pca")):
        mac(f"nHFiveA{a}", sum(H5[t][est]["H5a"] for t in elig), "{:d}")
        mac(f"nHFiveB{a}", sum(H5[t][est]["H5b"] for t in elig), "{:d}")
    rls = [H5[t]["diffmeans"]["r_Ls"] for t in elig]
    rlf = [H5[t]["diffmeans"]["r_Lf"] for t in elig]
    mac("rLsMin", min(rls)), mac("rLsMax", max(rls)), mac("rLfMin", min(rlf)), mac("rLfMax", max(rlf))
    # one-sided sign test p for H5a over eligible models
    k, n = sum(H5[t]["diffmeans"]["H5a"] for t in elig), len(elig)
    mac("pSignHFiveA", sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n, "{:.4f}")
    cross = [H5[t]["diffmeans"]["crossover_layer"] / H5[t]["diffmeans"]["L_f"] for t in elig
             if H5[t]["diffmeans"]["crossover_layer"] is not None]
    mac("crossMin", min(cross)), mac("crossMax", max(cross))
    # robustness: GoEmotions-derived direction, computed from the per-layer confirmation results
    def h5_src(t, src):
        C = load(t)["layers"]
        ds = np.array([l[src]["d_surface_self"] for l in C], float)
        do = np.array([l[src]["d_outcome_self"] for l in C], float)
        Ls, Lf = int(np.nanargmax(ds)), len(C) - 1
        r = lambda L: do[L] / (abs(do[L]) + abs(ds[L]))
        return r(Lf) > r(Ls), r(Lf) > 0.5
    g = [h5_src(t, "goemo/diffmeans") for t in elig]
    mac("nHFiveAGoemo", sum(a for a, _ in g), "{:d}")
    mac("nHFiveBGoemo", sum(b for _, b in g), "{:d}")
    # validity of the SST-2 direction in deep layers: split-half AUC on held-out SST-2 sentences
    V = json.load(open(RES / "sst2_splithalf_auc.json"))
    def late_band(vals):  # same band as band_boot.py: relative depth [0.60, 0.90], final layer excluded
        n = len(vals) - 1
        return float(np.nanmean([v for L, v in enumerate(vals) if 0.6 <= L / n <= 0.9 and L < n]))
    late = [late_band(V[t]) for t in V]
    fin = [V[t][-1] for t in V]
    mac("sstLateMin", min(late)), mac("sstLateMax", max(late)), mac("sstFinalMin", min(fin)), mac("sstFinalMax", max(fin))
    # pre-registered H5 restricted to models unseen at prereg v3
    unseen = ["mistral_7b_it", "falcon3_7b_it", "granite31_8b_it", "phi4", "qwen25_14b_it", "qwen25_32b_it"]
    ku = sum(H5[t]["diffmeans"]["H5a"] and H5[t]["diffmeans"]["H5b"] for t in unseen if t in H5)
    mac("nUnseen", len(unseen), "{:d}"), mac("nHFiveUnseen", ku, "{:d}")
    mac("pHFiveUnseen", sum(comb(len(unseen), i) for i in range(ku, len(unseen) + 1)) / 2 ** len(unseen), "{:.3f}")
    # H3 sign tests
    k3 = sum(S[t]["H3"] for t in elig)
    mac("pHThree", sum(comb(len(elig), i) for i in range(k3, len(elig) + 1)) / 2 ** len(elig), "{:.4f}")
    kb = sum(S[t]["behaviour_conf"]["self"]["B_surf"] > 0 for t in S)
    mac("pBsurf", sum(comb(len(S), i) for i in range(kb, len(S) + 1)) / 2 ** len(S), "{:.4f}")
    # band analysis (post hoc): conflict AUC ranges and CI exclusion counts
    for pos, a in (("raw", "Raw"), ("gen", "Gen")):
        f = RES / f"band_boot_{pos}.json"
        if not f.exists():
            continue
        Bd = json.load(open(f))
        mid = [Bd[t]["self"]["mid"]["conflict_auc"] for t in Bd]
        late = [Bd[t]["self"]["late"]["conflict_auc"] for t in Bd]
        lr = [Bd[t]["self"]["late"]["r"] for t in Bd]
        mac(f"midAUC{a}Min", min(mid)), mac(f"midAUC{a}Max", max(mid))
        mac(f"lateAUC{a}Min", min(late)), mac(f"lateAUC{a}Max", max(late))
        mac(f"lateR{a}Min", min(lr)), mac(f"lateR{a}Max", max(lr))
        mac(f"nBand{a}", len(Bd), "{:d}")
        mac(f"nBandCI{a}", sum(Bd[t]["self"]["late_minus_mid_auc_ci"][0] > 0 for t in Bd), "{:d}")
        d_lo = [Bd[t]["self"]["late_minus_mid_auc_ci"][0] for t in Bd]
        mac(f"diffCI{a}MinLo", min(d_lo))
        so = sum(Bd[t]["self"]["late"]["r"] > Bd[t]["other"]["late"]["r"] for t in Bd)
        mac(f"nSelfOther{a}", so, "{:d}")
        mac(f"lateROther{a}Min", min(Bd[t]["other"]["late"]["r"] for t in Bd))
        mac(f"lateROther{a}Max", max(Bd[t]["other"]["late"]["r"] for t in Bd))
    E = json.load(open(RES / "ablation_conflict_acc_explore.json"))
    for t, a in (("qwen25_7b_it", "QwenSeven"), ("olmo2_7b_it", "Olmo"), ("mistral_7b_it", "Mistral")):
        mac(f"ablAccNone{a}", E[t]["none"], "{:.3f}"), mac(f"ablAccVal{a}", E[t]["valence"], "{:.3f}")
        mac(f"ablAccRandMax{a}", E[t]["random_max"], "{:.3f}")
    A = json.load(open(RES / "ablation_summary.json"))
    for t, a in (("qwen25_7b_it", "QwenSeven"), ("olmo2_7b_it", "Olmo"), ("mistral_7b_it", "Mistral")):
        for sp, b in (("confirmation", "Conf"), ("discovery", "Disc")):
            v = A[t][sp]["mid/valence"]
            mac(f"ablRed{b}{a}", v["R_reduction"], "{:+.3f}"), mac(f"ablP{b}{a}", v["rand_reduction_p95"], "{:+.3f}")
    # post hoc sensitivity analyses
    Braw = json.load(open(RES / "band_boot_raw.json"))
    mac("nLateAboveHalfRaw", sum(Braw[t]["self"]["late"]["conflict_auc"] > 0.5 for t in Braw), "{:d}")
    mac("nLateAboveHalfRawElig", sum(Braw[t]["self"]["late"]["conflict_auc"] > 0.5 for t in elig), "{:d}")
    kb7 = sum(S[t]["behaviour_conf"]["self"]["B_surf"] > 0 for t in elig)
    mac("nBsurfPosElig", kb7, "{:d}")
    mac("pBsurfElig", sum(comb(len(elig), i) for i in range(kb7, len(elig) + 1)) / 2 ** len(elig), "{:.4f}")
    for t, a in (("qwen25_7b_it", "QwenSeven"), ("qwen25_14b_it", "QwenFourteen"), ("qwen25_32b_it", "QwenThirtyTwo")):
        mac(f"Rbeh{a}", S[t]["behaviour_conf"]["self"]["R"], "{:.3f}")
    for pos, a in (("raw", "Raw"), ("gen", "Gen")):
        vals = []
        for t in (Braw if pos == "raw" else json.load(open(RES / "band_boot_gen.json"))):
            res = load(t, pos=pos)
            n = len(res["layers"]) - 1
            vals.append(np.nanmean([l["sst2/diffmeans"]["auc_surface_none"] for l in res["layers"]
                                    if 0.6 <= l["layer"] / n <= 0.9 and l["layer"] < n]))
        mac(f"noneLate{a}Min", min(vals)), mac(f"noneLate{a}Max", max(vals))
        Dm = json.load(open(RES / f"domain_band_{pos}.json"))
        pairs = [(Dm[t][d]["mid"], Dm[t][d]["late"]) for t in Dm for d in Dm[t]]
        mac(f"nDomPairs{a}", len(pairs), "{:d}"), mac(f"nDomUp{a}", sum(l > m for m, l in pairs), "{:d}")
    # model-level bootstrap of the mean late-minus-mid conflict AUC (between-model variance)
    rng = np.random.default_rng(20261006)
    for pos, a in (("raw", "Raw"), ("gen", "Gen")):
        Bd = json.load(open(RES / f"band_boot_{pos}.json"))
        dif = np.array([Bd[t]["self"]["late"]["conflict_auc"] - Bd[t]["self"]["mid"]["conflict_auc"] for t in Bd])
        bs = [rng.choice(dif, len(dif)).mean() for _ in range(10000)]
        mac(f"modelDiff{a}", dif.mean()), mac(f"modelDiff{a}Lo", np.percentile(bs, 2.5)), mac(f"modelDiff{a}Hi", np.percentile(bs, 97.5))
        late = {t: Bd[t]["self"]["late"] for t in Bd}
        if pos == "raw":
            for t, nm in (("qwen25_7b_it", "QwenSeven"), ("falcon3_7b_it", "Falcon")):
                mac(f"lateAUCRaw{nm}", late[t]["conflict_auc"]), mac(f"lateAUCRaw{nm}Hi", late[t]["conflict_auc_ci"][1])
    # prereg v4/v5 (H6-H10) from sweep2_summary.json
    if (RES / "sweep2_summary.json").exists():
        W = json.load(open(RES / "sweep2_summary.json"))
        el2 = [t for t in elig if t in W]
        mac("nSweep", len(el2), "{:d}")
        for pos, a in (("raw", "Raw"), ("gen", "Gen")):
            generic = 0
            for t in el2:
                c, v = W[t][f"control_{pos}"], W[t][f"valence_{pos}"]
                dc, dv = (c["fit"] or {}).get("d_star"), (v["fit"] or {}).get("d_star")
                if c["mid"] <= 0.15 and c["diff_ci"][0] > 0 and dc is not None and dv is not None and abs(dc - dv) <= 0.10:
                    generic += 1
            mac(f"nGeneric{a}", generic, "{:d}")
            cl = [W[t][f"control_{pos}"]["late"] for t in el2]
            mac(f"ctrlLate{a}Min", min(cl)), mac(f"ctrlLate{a}Max", max(cl))
            cm = [W[t][f"control_{pos}"]["mid"] for t in el2]
            mac(f"ctrlMid{a}Min", min(cm)), mac(f"ctrlMid{a}Max", max(cm))
            ds = [(W[t][f"valence_{pos}"]["fit"] or {}).get("d_star") for t in el2]
            ds = [d for d in ds if d is not None]
            mac(f"dstar{a}Min", min(ds)), mac(f"dstar{a}Max", max(ds))
            r2 = [(W[t][f"valence_{pos}"]["fit"] or {}).get("r2", 0) for t in el2]
            mac(f"fitRtwo{a}Min", min(r2))
        Bg = json.load(open(RES / "band_boot_gen.json"))
        mac("nCtrlBelowVal", sum(W[t]["control_gen"]["late"] < Bg[t]["self"]["late"]["conflict_auc"] for t in el2), "{:d}")
        cr = [(t, W[t].get("patching", {}).get("crossing_depth")) for t in el2]
        cr = [(t, c) for t, c in cr if c is not None]
        mac("crossPMin", min(c for _, c in cr)), mac("crossPMax", max(c for _, c in cr))
        mac("nCrossAligned", sum(abs(c - (W[t]["valence_gen"]["fit"] or {}).get("d_star", 9)) <= 0.15 for t, c in cr), "{:d}")
        mac("nCross", len(cr), "{:d}")
        # H11 second control (wooden/metal), same rule as H6
        if all("control2_gen" in W[t] for t in el2):
            for pos, a in (("raw", "Raw"), ("gen", "Gen")):
                g = 0
                for t in el2:
                    c, v = W[t][f"control2_{pos}"], W[t][f"valence_{pos}"]
                    dc, dv = (c["fit"] or {}).get("d_star"), (v["fit"] or {}).get("d_star")
                    if c["mid"] <= 0.15 and c["diff_ci"][0] > 0 and dc is not None and dv is not None and abs(dc - dv) <= 0.10:
                        g += 1
                mac(f"nGenericTwo{a}", g, "{:d}")
                cl = [W[t][f"control2_{pos}"]["late"] for t in el2]
                mac(f"ctrlTwoLate{a}Min", min(cl)), mac(f"ctrlTwoLate{a}Max", max(cl))
            Bg2 = json.load(open(RES / "band_boot_gen.json"))
            mac("nCtrlTwoBelowVal", sum(W[t]["control2_gen"]["late"] < Bg2[t]["self"]["late"]["conflict_auc"] for t in el2), "{:d}")
            b2 = [W[t]["control2_behaviour"]["B_surf"] for t in el2]
            mac("ctrlTwoBsurfMin", min(b2), "{:+.2f}"), mac("ctrlTwoBsurfMax", max(b2), "{:+.2f}")
            mac("nCtrlTwoBsurfPos", sum(b > 0 for b in b2), "{:d}")
            ca2 = [W[t]["control2_behaviour"]["acc_conflict"] for t in el2]
            mac("ctrlTwoConfAccMin", min(ca2)), mac("ctrlTwoConfAccMax", max(ca2))
        # sensitivity: H6 criterion with both fits started at relative depth 0.2
        for pos, a in (("raw", "Raw"), ("gen", "Gen")):
            g = 0
            for t in el2:
                c = W[t][f"control_{pos}"]
                dv, _ = fit_from(W[t][f"valence_{pos}"]["profile"], 0.2)
                dc, _ = fit_from(c["profile"], 0.2)
                if c["mid"] <= 0.15 and c["diff_ci"][0] > 0 and dv is not None and dc is not None and abs(dc - dv) <= 0.10:
                    g += 1
            mac(f"nGenericFixed{a}", g, "{:d}")
            if all("control2_gen" in W[t] for t in el2):
                g2 = 0
                for t in el2:
                    c = W[t][f"control2_{pos}"]
                    dv, _ = fit_from(W[t][f"valence_{pos}"]["profile"], 0.2)
                    dc, _ = fit_from(c["profile"], 0.2)
                    if c["mid"] <= 0.15 and c["diff_ci"][0] > 0 and dv is not None and dc is not None and abs(dc - dv) <= 0.10:
                        g2 += 1
                mac(f"nGenericFixedTwo{a}", g2, "{:d}")
        flagged = sum(((W[t][f"{task}_{pos}"]["fit"] or {}).get("r2", 1) < 0.8)
                      for t in el2 for task in ("valence", "control") for pos in ("raw", "gen"))
        mac("nFlaggedFits", flagged, "{:d}")
        if all("control2_gen" in W[t] for t in el2):
            mac("nFlaggedFitsTwo", sum(((W[t][f"control2_{pos}"]["fit"] or {}).get("r2", 1) < 0.8)
                                       for t in el2 for pos in ("raw", "gen")), "{:d}")
            for pos, a in (("raw", "Raw"), ("gen", "Gen")):
                v2 = []
                for t in el2:
                    na = np.array(W[t][f"control2_{pos}"]["none_auc"], float); n = len(na) - 1
                    v2.append(np.nanmean([na[L] for L in range(n) if 0.6 <= L / n <= 0.9]))
                mac(f"ctrlTwoNoneLate{a}Min", min(v2)), mac(f"ctrlTwoNoneLate{a}Max", max(v2))
        # control direction validity on stake-free control items, late band, and control behavioural bias sign
        for pos, a in (("raw", "Raw"), ("gen", "Gen")):
            v = []
            for t in el2:
                na = np.array(W[t][f"control_{pos}"]["none_auc"], float); n = len(na) - 1
                v.append(np.nanmean([na[L] for L in range(n) if 0.6 <= L / n <= 0.9]))
            mac(f"ctrlNoneLate{a}Min", min(v)), mac(f"ctrlNoneLate{a}Max", max(v))
        cbs = [W[t]["control_behaviour"]["B_surf"] for t in el2]
        mac("ctrlBsurfMin", min(cbs), "{:+.2f}"), mac("ctrlBsurfMax", max(cbs), "{:+.2f}")
        mac("nCtrlBsurfPos", sum(b > 0 for b in cbs), "{:d}")
        gaps = [W[t]["patching"]["crossing_depth"] - (W[t]["valence_gen"]["fit"] or {}).get("d_star", np.nan)
                for t in el2 if W[t].get("patching", {}).get("crossing_depth") is not None]
        mac("gapMin", min(gaps), "{:+.2f}"), mac("gapMax", max(gaps), "{:+.2f}")
        # control behaviour
        cb = [W[t]["control_behaviour"] for t in el2]
        mac("ctrlConfAccMin", min(b["acc_conflict"] for b in cb)), mac("ctrlConfAccMax", max(b["acc_conflict"] for b in cb))
        # H8 prompts
        keys = sorted({k for t in el2 for k in W[t]["prompts"]})
        pos_counts = [sum(W[t]["prompts"][k]["B_surf"] > 0 for t in el2 if k in W[t]["prompts"]) for k in keys]
        mac("nPromptVariants", len(keys), "{:d}"), mac("minPromptPos", min(pos_counts), "{:d}")
        # H10 natural
        nat = [t for t in el2 if "natural_raw" in W[t]]
        if nat:
            mac("nNat", len(nat), "{:d}")
            for pos, a in (("raw", "Raw"), ("gen", "Gen")):
                mac(f"nNatCI{a}", sum(W[t][f"natural_{pos}"]["diff_ci"][0] > 0 for t in nat), "{:d}")
                mac(f"nNatMidBelow{a}", sum(W[t][f"natural_{pos}"]["mid"] < 0.5 for t in nat), "{:d}")
                nm = [W[t][f"natural_{pos}"]["mid"] for t in nat]; nl = [W[t][f"natural_{pos}"]["late"] for t in nat]
                mac(f"natMid{a}Min", min(nm)), mac(f"natMid{a}Max", max(nm)), mac(f"natLate{a}Min", min(nl)), mac(f"natLate{a}Max", max(nl))
            mac("nNatBsurf", sum(W[t]["natural_behaviour"]["B_surf"] > 0 for t in nat), "{:d}")
            nb = [W[t]["natural_behaviour"] for t in nat]
            mac("natConfAccMin", min(b["acc_conflict"] for b in nb)), mac("natConfAccMax", max(b["acc_conflict"] for b in nb))
    # H12 in-format directions with validity gate (prereg v7)
    fI = RES / "informat_summary.json"
    if fI.exists():
        I = json.load(open(fI))
        el3 = [t for t in elig if t in I]
        mac("nInf", len(el3), "{:d}")
        for ctrl, a in (("location", "Loc"), ("material", "Mat")):
            gated = [t for t in el3 if I[t]["valence_gen"]["validity_late"] >= 0.80 and I[t][f"{ctrl}_gen"]["validity_late"] >= 0.80]
            spec = [t for t in gated if I[t]["valence_gen"]["late"] > I[t][f"{ctrl}_gen"]["late"] and I[t][f"{ctrl}_gen"]["late"] <= 0.60]
            mac(f"nGated{a}", len(gated), "{:d}"), mac(f"nSpec{a}", len(spec), "{:d}")
            cl = [I[t][f"{ctrl}_gen"]["late"] for t in el3]; cv = [I[t][f"{ctrl}_gen"]["validity_late"] for t in el3]
            mac(f"inf{a}LateMin", min(cl)), mac(f"inf{a}LateMax", max(cl))
            mac(f"inf{a}ValidMin", min(cv)), mac(f"inf{a}ValidMax", max(cv))
        vl = [I[t]["valence_gen"]["late"] for t in el3]; vv = [I[t]["valence_gen"]["validity_late"] for t in el3]
        mac("infValLateMin", min(vl)), mac("infValLateMax", max(vl)), mac("infValValidMin", min(vv)), mac("infValValidMax", max(vv))
        mac("nInfValAboveHalf", sum(v > 0.5 for v in vl), "{:d}")
        for ctrl, a in (("location", "Loc"), ("material", "Mat")):
            gated = [t for t in el3 if I[t]["valence_gen"]["validity_late"] >= 0.80 and I[t][f"{ctrl}_gen"]["validity_late"] >= 0.80]
            mac(f"nStrict{a}", sum(I[t]["valence_gen"]["late"] > 0.5 and I[t][f"{ctrl}_gen"]["late"] < 0.5 for t in gated), "{:d}")
            mac(f"nCtrlMoves{a}", sum(I[t][f"{ctrl}_gen"]["diff_ci"][0] > 0 for t in el3), "{:d}")
            gv = [I[t][f"{ctrl}_gen"]["validity_late"] for t in gated]
            mac(f"inf{a}ValidGatedMin", min(gv))
        mac("nRawValBelowLoc", sum(I[t]["valence_raw"]["late"] < I[t]["location_raw"]["late"] for t in el3), "{:d}")
        mac("infQwenThirtyTwoVal", I["qwen25_32b_it"]["valence_gen"]["late"]), mac("infQwenThirtyTwoLoc", I["qwen25_32b_it"]["location_gen"]["late"])
        rl = [I[t]["valence_raw"]["late"] for t in el3]
        mac("infValRawLateMin", min(rl)), mac("infValRawLateMax", max(rl))
        cr = [max(I[t]["location_raw"]["late"], I[t]["material_raw"]["late"]) for t in el3]
        mac("infCtrlRawLateMax", max(cr))
    # other perspective (saved band results), prereg v8 (sweep3), lexical-prior regression
    Bgo = json.load(open(RES / "band_boot_gen.json"))
    inst_g = [t for t in elig if t in Bgo]
    og = [Bgo[t]["other"]["late"]["conflict_auc"] for t in inst_g]
    mac("otherLateGenMin", min(og)), mac("otherLateGenMax", max(og))
    mac("nOtherAboveHalf", sum(v > 0.5 for v in og), "{:d}"), mac("nOtherGen", len(og), "{:d}")
    mac("nOtherCI", sum(Bgo[t]["other"]["late_minus_mid_auc_ci"][0] > 0 for t in inst_g), "{:d}")
    mac("nSelfAboveOther", sum(Bgo[t]["self"]["late"]["conflict_auc"] > Bgo[t]["other"]["late"]["conflict_auc"] for t in inst_g), "{:d}")
    if (RES / "prior_regression.json").exists():
        PR = json.load(open(RES / "prior_regression.json"))
        mac("nBetaLPos", sum(PR[t]["ci_bL"][0] > 0 for t in PR), "{:d}"), mac("nBetaModels", len(PR), "{:d}")
        for t, a in (("qwen25_7b_it", "QwenSeven"), ("qwen25_14b_it", "QwenFourteen"), ("qwen25_32b_it", "QwenThirtyTwo"),
                     ("qwen25_7b_base", "QwenSevenBase"), ("olmo2_7b_base", "OlmoBase"), ("olmo2_7b_it", "Olmo")):
            if t in PR:
                mac(f"betaA{a}", PR[t]["bA"], "{:.1f}"), mac(f"betaL{a}", PR[t]["bL"], "{:.1f}")
    if (RES / "sweep3_summary.json").exists():
        T3 = json.load(open(RES / "sweep3_summary.json"))
        gn = [t for t in elig if t in T3 and "gn_sst_gen" in T3[t]]
        if gn:
            v = [T3[t]["gn_sst_gen"]["late"] for t in gn]
            mac("nGN", len(gn), "{:d}"), mac("gnSstGenMin", min(v)), mac("gnSstGenMax", max(v))
            mac("nGNCI", sum(T3[t]["gn_sst_gen"]["late_ci"][0] > 0.5 for t in gn), "{:d}")
            mac("gnSstRawMax", max(T3[t]["gn_sst_raw"]["late"] for t in gn))
            vi = [T3[t]["gn_inf_gen"]["late"] for t in gn]
            mac("gnInfGenMin", min(vi)), mac("gnInfGenMax", max(vi)), mac("nGNInfAbove", sum(x > 0.5 for x in vi), "{:d}")
            vo = [T3[t]["gn_sst_gen_other"]["late"] for t in gn]
            mac("gnOtherMin", min(vo)), mac("gnOtherMax", max(vo)), mac("nGNOtherAbove", sum(x > 0.5 for x in vo), "{:d}")
            bb = [T3[t]["gn_behaviour"] for t in gn]
            mac("gnAccMin", min(b["acc"] for b in bb)), mac("gnAccMax", max(b["acc"] for b in bb))
            mac("nGNBsurfPos", sum(b["B_surf"] > 0 for b in bb), "{:d}")
        tj = [t for t in elig if t in T3 and "traj_response" in T3[t]]
        if tj:
            for nm, a in (("text_end", "Text"), ("question_end", "Question"), ("response", "Resp")):
                v = [T3[t][f"traj_{nm}"]["late"] for t in tj]
                mac(f"traj{a}Min", min(v)), mac(f"traj{a}Max", max(v))
            mac("nTraj", len(tj), "{:d}")
            mac("nTrajUp", sum(T3[t]["traj_question_end"]["late"] > T3[t]["traj_text_end"]["late"] + 0.1 for t in tj), "{:d}")
        nv = [T3[t]["natval"]["agree"] for t in T3 if "natval" in T3[t]]
        if nv:
            mac("natValMin", min(nv), "{:.3f}"), mac("natValMax", max(nv), "{:.3f}"), mac("nNatVal", len(nv), "{:d}")
        for t, a in (("qwen25_7b_base", "QwenSevenBase"), ("qwen25_7b_it", "QwenSeven"), ("olmo2_7b_base", "OlmoBase"), ("olmo2_7b_it", "Olmo")):
            if t in T3 and "base_cont" in T3[t]:
                mac(f"cont{a}", T3[t]["base_cont"]["late"])
    K = json.load(open(RES / "clf_baseline.json"))
    mac("clfAUCMax", max(v["conflict_auc"] for v in K.values()))
    mac("clfBoutMax", max(abs(v["B_out"]) for v in K.values()))
    mac("clfBsurfMin", min(v["B_surf"] for v in K.values()))
    X = json.load(open(RES / "cross_source_auc.json"))
    g2s = [late_band(X[t]["goemo_to_sst2"]) for t in X]
    s2g = [late_band(X[t]["sst2_to_goemo"]) for t in X]
    mac("crossGSMin", min(g2s)), mac("crossGSMax", max(g2s)), mac("crossSGMin", min(s2g)), mac("crossSGMax", max(s2g))
    if (RES / "sweep4_summary.json").exists():  # prereg v9: task gating (H18) and goal-explicit key control (H19)
        W4 = json.load(open(RES / "sweep4_summary.json"))
        tg = [t for t in elig if t in W4 and "tg_none" in W4[t]]
        for k, a in (("eval_self", "Eval"), ("eval_plain", "Plain"), ("factual", "Fact"), ("irrelevant", "Irr"), ("none", "None")):
            v = [W4[t][f"tg_{k}"]["late"] for t in tg]
            mac(f"tg{a}Min", min(v)), mac(f"tg{a}Max", max(v))
        mac("nTG", len(tg), "{:d}")
        gated = [t for t in tg if all(W4[t][f"tg_{k}"]["late"] <= 0.60 for k in ("factual", "irrelevant"))
                 and all(W4[t]["tg_eval_self"]["late"] - W4[t][f"tg_{k}"]["late"] >= 0.10 for k in ("factual", "irrelevant"))]
        mac("nTGGated", len(gated), "{:d}")
        mac("nTGValidFail", sum(W4[t][f"tg_{k}"]["validity_late"] < 0.80 for t in tg
                                for k in ("eval_self", "eval_plain", "factual", "irrelevant", "none")), "{:d}")
        g2 = [t for t in elig if t in W4 and "gn2_sst_gen" in W4[t]]
        v = [W4[t]["gn2_sst_gen"]["late"] for t in g2]
        mac("gnTwoMin", min(v)), mac("gnTwoMax", max(v)), mac("nGNTwo", len(g2), "{:d}")
        mac("nGNTwoCI", sum(W4[t]["gn2_sst_gen"]["late_ci"][0] > 0.5 for t in g2), "{:d}")
        mac("gnTwoAccMin", min(W4[t]["gn2_behaviour"]["acc"] for t in g2)), mac("gnTwoAccMax", max(W4[t]["gn2_behaviour"]["acc"] for t in g2))
    if (RES / "sweep5_summary.json").exists():  # prereg v10: stake-dependent non-evaluative questions (H22)
        W5 = json.load(open(RES / "sweep5_summary.json"))
        t5 = [t for t in elig if t in W5]
        for k, a in (("eval_self", "EvalB"), ("outcome_factual", "OutF"), ("binding_factual", "BindF")):
            v = [W5[t][k]["late"] for t in t5]
            mac(f"tg{a}Min", min(v)), mac(f"tg{a}Max", max(v))
        mac("nTGB", len(t5), "{:d}")
        mac("nGoalComp", sum(W5[t]["outcome_factual"]["late"] > 0.60 and W5[t]["outcome_factual"]["late"] >= W5[t]["eval_self"]["late"] - 0.10 for t in t5), "{:d}")
        mac("nApprSpec", sum(W5[t]["eval_self"]["late"] - W5[t]["outcome_factual"]["late"] >= 0.10 and W5[t]["outcome_factual"]["late"] <= 0.60 for t in t5), "{:d}")
        mac("nOutFAbove", sum(W5[t]["outcome_factual"]["late_ci"][0] > 0.5 for t in t5), "{:d}")
        mac("nBindFAbove", sum(W5[t]["binding_factual"]["late_ci"][0] > 0.5 for t in t5), "{:d}")
        mac("tgOutAccMin", min(W5[t]["beh_outcome_acc"] for t in t5)), mac("tgOutAccMax", max(W5[t]["beh_outcome_acc"] for t in t5))
        mac("tgBindAccMin", min(W5[t]["beh_binding_acc"] for t in t5)), mac("tgBindAccMax", max(W5[t]["beh_binding_acc"] for t in t5))
        mac("nTGBValidFail", sum(W5[t][k]["validity_late"] < 0.80 for t in t5 for k in ("eval_self", "outcome_factual", "binding_factual")), "{:d}")
    if (RES / "decompose_summary.json").exists():  # prereg v9: additive decomposition of the readout (H20)
        D = json.load(open(RES / "decompose_summary.json"))
        dm = [t for t in elig if t in D and "inf_gen" in D[t]]
        for src, a in (("inf_gen", "Gen"), ("inf_raw", "Raw"), ("traj_response", "Resp")):
            s = [D[t][src]["late_s"] for t in dm]; o = [D[t][src]["late_o"] for t in dm]
            mac(f"dec{a}SMin", min(s)), mac(f"dec{a}SMax", max(s)), mac(f"dec{a}OMin", min(o)), mac(f"dec{a}OMax", max(o))
            mac(f"nDec{a}SPos", sum(D[t][src]["late_s_ci"][0] > 0 for t in dm), "{:d}")
            mac(f"nDec{a}OUp", sum(D[t][src]["late_o"] > D[t][src]["mid_o"] and D[t][src]["mid_o_ci"][1] < D[t][src]["late_o_ci"][0] for t in dm), "{:d}")
            mac(f"nDec{a}OgtS", sum(D[t][src]["late_o_minus_s_ci"][0] > 0 for t in dm), "{:d}")
            mac(f"nDec{a}MbeforeO", sum(D[t][src]["d_m"] is not None and D[t][src]["d_o"] is not None and D[t][src]["d_m"] < D[t][src]["d_o"] for t in dm), "{:d}")
        mac("nDec", len(dm), "{:d}")
    if (RES / "leak_audit.json").exists():  # prereg v9: lexical-leakage audit (H21)
        La = json.load(open(RES / "leak_audit.json"))
        lt = [t for t in La if not t.startswith("_")]
        mac("nLeak", len(lt), "{:d}")
        for ds, a in (("sst2", "Sst"), ("goemo", "Go")):
            mac(f"leakBow{a}", La["_bow"][ds])
            r = [La[t][ds]["bow_ratio"] for t in lt]; b = [La[t][ds]["best"] for t in lt]
            h = [La[t][ds]["acc_bow_wrong"][La[t][ds]["best_layer"]] for t in lt]
            mac(f"leakRatio{a}Min", r and min(r)), mac(f"leakRatio{a}Max", max(r))
            mac(f"leakBest{a}Min", min(b)), mac(f"leakBest{a}Max", max(b))
            mac(f"leakHard{a}Min", min(h)), mac(f"leakHard{a}Max", max(h))
            mac(f"nLeakRatio{a}", sum(x >= 0.90 for x in r), "{:d}")
    (BASE / "outputs" / "numbers.tex").write_text("\n".join(out) + "\n")


def table_main():
    """Main results table (confirmation domains, self perspective): one row per model."""
    S = json.load(open(RES / "confirmatory_summary.json"))
    H5 = json.load(open(RES / "h5_summary.json"))
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Model & $d_{\mathrm{out}}$ & $d_{\mathrm{surf}}$ & $r(\ell_s)$ & $r(\ell_f)$ \\",
             r"\midrule"]
    for group in (INSTRUCT, BASEM):
        for t, lab in group:
            if t not in S or t not in H5:
                continue
            b = S[t]["behaviour_conf"]["self"]
            dm = S[t]["diffmeans"]
            h = H5[t]["diffmeans"]
            star = "" if b["acc"] >= 0.85 else r"$^\dagger$"
            lines.append(f"{lab}{star} & "
                         f"{dm['d_outcome_self']:.2f} & {dm['d_surface_self']:.2f} & "
                         f"{h['r_Ls']:.2f} & {h['r_Lf']:.2f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (BASE / "outputs" / "table_main.tex").write_text("\n".join(lines) + "\n")


def table_ablation():
    """H4 table: mid-band ablation on confirmation domains (self)."""
    A = json.load(open(RES / "ablation_summary.json"))
    names = {"qwen25_7b_it": "Qwen2.5-7B-Inst.", "olmo2_7b_it": "OLMo-2-7B-Inst.", "mistral_7b_it": "Mistral-7B-Inst."}
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Model & Ablation & $B_{\mathrm{surf}}$ & $B_{\mathrm{out}}$ & $R$ & Confl. & Cong. \\", r"\midrule"]
    for t, lab in names.items():
        if t not in A:
            continue
        c = A[t]["confirmation"]
        rows = [("none", c["none"]), ("valence (SST-2)", c["mid/valence"]), ("valence (GoEmo.)", c["mid/goemo"])]
        for i, (n, v) in enumerate(rows):
            lines.append(f"{lab if i == 0 else ''} & {n} & {v['B_surf']:.2f} & {v['B_out']:.2f} & {v['R']:.3f} & "
                         f"{v['acc_conflict']:.3f} & {v['acc_congruent']:.3f} \\\\")
        v = c["mid/valence"]
        lines.append(f" & random (p95 of $\\Delta R$) & & & {v['rand_reduction_p95']:+.3f} & & \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (BASE / "outputs" / "table_ablation.tex").write_text("\n".join(lines) + "\n")


def fig_conflict(name="fig_conflict"):
    """Main figure: conflict-cell AUC (0 = valence follows the word, 1 = follows the stake) vs relative depth."""
    fig, ax0 = plt.subplots(1, 1, figsize=(3.1, 2.2))
    axes = [ax0]
    for ax, pos, title in ((ax0, "raw", "end of text"),):
        for (tag, lab), c in zip(INSTRUCT, COLORS):
            res = load(tag, pos=pos)
            if res is None:
                continue
            L = res["layers"]
            n = len(L) - 1
            x = np.array([l["layer"] / n for l in L])
            y = np.array([l["sst2/diffmeans"]["auc_outcome_conflict_self"] for l in L], float)
            ax.plot(x[1:], y[1:], color=c, lw=1.1, label=lab)
        ax.axhline(0.5, color="gray", lw=0.6, ls=":")
        for lo, hi in ((0.2, 0.45), (0.6, 0.9)):
            ax.axvspan(lo, hi, color="#EEEEEE", zorder=0, lw=0)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("relative depth")
        ax.set_ylim(-0.02, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("conflict-cell AUC for the stake")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.42), handlelength=1.4,
               columnspacing=1.0, fontsize=6)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_mechanism(name="fig_mechanism"):
    """Figure 2: (a) valence conflict AUC, (b) left/right control conflict AUC, (c) patching information flow;
    all at the response position, one line per instruct model."""
    f = RES / "sweep2_summary.json"
    if not f.exists():
        return
    W = json.load(open(f))
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.1), gridspec_kw={"wspace": 0.32})
    for (tag, lab), c in zip(INSTRUCT, COLORS):
        if tag not in W:
            continue
        w = W[tag]
        Iinf = json.load(open(RES / "informat_summary.json")) if (RES / "informat_summary.json").exists() else {}
        for ax, key, ls in ((axes[0], "valence_gen", "-"), (axes[1], "location_gen", "-"), (axes[1], "material_gen", ":")):
            src = w if key == "valence_gen" else Iinf.get(tag, {})
            if key not in src:
                continue
            y = np.array(src[key]["profile"], float)
            x = np.arange(len(y)) / (len(y) - 1)
            ax.plot(x[1:], y[1:], color=c, lw=1.0, ls=ls, label=lab if key == "valence_gen" else None)
        if "patching" in w:
            El, Ef = np.array(w["patching"]["E_label"]), np.array(w["patching"]["E_final"])
            x = (np.arange(len(El)) + 1) / len(El)
            axes[2].plot(x, El, color=c, lw=0.9, ls="--")
            axes[2].plot(x, Ef, color=c, lw=1.0)
    titles = ["(a) valence (SST-2 direction)", "(b) non-affective controls", "(c) patching the stake"]
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=8)
        ax.set_xlabel("relative depth")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:2]:
        ax.axhline(0.5, color="gray", lw=0.6, ls=":")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("conflict-cell AUC for the stake")
    axes[2].set_ylabel("normalised patching effect")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.27), handlelength=1.4,
               columnspacing=1.0)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def table_summary():
    """Table 1: one row per instruct model (confirmation domains, self)."""
    Braw = json.load(open(RES / "band_boot_raw.json"))
    Bgen = json.load(open(RES / "band_boot_gen.json"))
    W = json.load(open(RES / "sweep2_summary.json")) if (RES / "sweep2_summary.json").exists() else {}
    S = json.load(open(RES / "confirmatory_summary.json"))
    fmt = lambda v: "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.2f}"
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{end of text} & & & \\",
             r"\cmidrule(lr){2-3}",
             r"Model & mid & late & cross & natural & $B_{\mathrm{surf}}$ \\",
             r"\midrule"]
    for t, lab in INSTRUCT:
        if t not in Braw:
            continue
        w = W.get(t, {})
        dstar = (w.get("valence_gen", {}).get("fit") or {}).get("d_star")
        cross = w.get("patching", {}).get("crossing_depth")
        Iinf = json.load(open(RES / "informat_summary.json")) if (RES / "informat_summary.json").exists() else {}
        ctrl = Iinf.get(t, {}).get("location_gen", {}).get("late")
        ctrl2 = Iinf.get(t, {}).get("material_gen", {}).get("late")
        nat = w.get("natural_gen", {}).get("late")
        star = "" if S[t]["behaviour_conf"]["self"]["acc"] >= 0.85 else r"$^\dagger$"
        lines.append(f"{lab}{star} & {Braw[t]['self']['mid']['conflict_auc']:.2f} & {Braw[t]['self']['late']['conflict_auc']:.2f} & "
                     f"{fmt(cross)} & {fmt(nat)} & {S[t]['behaviour_conf']['self']['B_surf']:+.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (BASE / "outputs" / "table_summary.tex").write_text("\n".join(lines) + "\n")


def table_prompts():
    """Appendix: surface bias per question format and option order across eligible instruct models."""
    W = json.load(open(RES / "sweep2_summary.json"))
    S = json.load(open(RES / "confirmatory_summary.json"))
    tags = [t for t, _ in INSTRUCT if t in W and S[t]["behaviour_conf"]["self"]["acc"] >= 0.85]
    names = ["good/bad", "benefit/harm", "favorable/unfavorable", "positive/negative", "happy/sad"]
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Question & Order & $B_{\mathrm{surf}}>0$ & mean $B_{\mathrm{surf}}$ & conflict acc. (min--max) \\", r"\midrule"]
    for k, nm in enumerate(names):
        for rev, lab in ((0, "positive first"), (1, "negative first")):
            key = f"t{k}_rev{rev}"
            vals = [W[t]["prompts"][key] for t in tags if key in W[t]["prompts"]]
            if not vals:
                continue
            bs = [v["B_surf"] for v in vals]; ca = [v["acc_conflict"] for v in vals]
            lines.append(f"{nm if rev == 0 else ''} & {lab} & {sum(b > 0 for b in bs)}/{len(bs)} & {np.mean(bs):+.2f} & "
                         f"{min(ca):.2f}--{max(ca):.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (BASE / "outputs" / "table_prompts.tex").write_text("\n".join(lines) + "\n")


def _sig(d, lo, hi, k, d0):
    return lo + (hi - lo) / (1 + np.exp(-k * (d - d0)))


def fit_from(prof, start_depth):
    """Sensitivity fit: same sigmoid as analyze_sweep2.fit_transition, but from a fixed relative depth for every task."""
    from scipy.optimize import curve_fit
    prof = np.array(prof, float); n = len(prof) - 1
    d = np.arange(len(prof)) / n
    m = (d >= start_depth) & (np.arange(len(prof)) < n) & np.isfinite(prof)
    x, y = d[m], prof[m]
    try:
        p, _ = curve_fit(_sig, x, y, p0=[y.min(), y.max(), 20.0, float(np.median(x))],
                         bounds=([-0.1, -0.1, 0.1, 0.0], [1.1, 1.1, 200.0, 1.0]), maxfev=20000)
        r2 = 1 - np.sum((y - _sig(x, *p)) ** 2) / np.sum((y - y.mean()) ** 2)
        return float(p[3]), float(r2)
    except Exception:
        return None, None


def table_fits():
    """Appendix: transition fits (pre-registered start at the AUC minimum, and fixed start at depth 0.2)."""
    W = json.load(open(RES / "sweep2_summary.json"))
    S = json.load(open(RES / "confirmatory_summary.json"))
    tags = [t for t, _ in INSTRUCT if t in W]
    lab = dict(INSTRUCT)
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Model & Readout & $d^*$ val. ($R^2$) & $d^*$ ctrl ($R^2$) & $d^*_{0.2}$ val. & $d^*_{0.2}$ ctrl \\", r"\midrule"]
    out = {}
    for t in tags:
        for pos in ("raw", "gen"):
            fv, fc = W[t][f"valence_{pos}"]["fit"] or {}, W[t][f"control_{pos}"]["fit"] or {}
            sv, _ = fit_from(W[t][f"valence_{pos}"]["profile"], 0.2)
            sc, _ = fit_from(W[t][f"control_{pos}"]["profile"], 0.2)
            out[(t, pos)] = (sv, sc)
            flag = lambda f: (f"{f.get('d_star', float('nan')):.2f} ({f.get('r2', float('nan')):.2f})"
                              + ("$^\\ast$" if f.get("r2", 1) < 0.8 else ""))
            name = lab[t] + ("" if S[t]["behaviour_conf"]["self"]["acc"] >= 0.85 else r"$^\dagger$")
            lines.append(f"{name if pos == 'raw' else ''} & {'end of text' if pos == 'raw' else 'response'} & "
                         f"{flag(fv)} & {flag(fc)} & {sv if sv is None else f'{sv:.2f}'} & {sc if sc is None else f'{sc:.2f}'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (BASE / "outputs" / "table_fits.tex").write_text("\n".join(lines) + "\n")
    return out


def fig_informat(name="fig_informat"):
    """In-format directions at the response position: conflict-cell AUC for valence, location, material."""
    f = RES / "informat_summary.json"
    if not f.exists():
        return
    I = json.load(open(f))
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.0), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, task, title in zip(axes, ("valence", "location", "material"), ("(a) valence", "(b) location", "(c) material")):
        for (tag, lab), c in zip(INSTRUCT, COLORS):
            if tag not in I:
                continue
            y = np.array(I[tag][f"{task}_gen"]["profile"], float)
            x = np.arange(len(y)) / (len(y) - 1)
            ax.plot(x[1:], y[1:], color=c, lw=1.0, label=lab if task == "valence" else None)
        ax.axhline(0.5, color="gray", lw=0.6, ls=":")
        ax.set_title(title, fontsize=8); ax.set_xlabel("relative depth"); ax.set_ylim(-0.02, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("conflict-cell AUC for the stake")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.3), handlelength=1.4, columnspacing=1.0)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)



def fig_controls_source(name="fig_controls_source"):
    """Appendix: controls read with out-of-format (source-sentence) directions, end of text and response position."""
    W = json.load(open(RES / "sweep2_summary.json"))
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.0), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, pos, title in ((axes[0], "raw", "end of text"), (axes[1], "gen", "response position")):
        for (tag, lab), c in zip(INSTRUCT, COLORS):
            if tag not in W:
                continue
            for key, ls in (("control", "-"), ("control2", ":")):
                if f"{key}_{pos}" not in W[tag]:
                    continue
                y = np.array(W[tag][f"{key}_{pos}"]["profile"], float); x = np.arange(len(y)) / (len(y) - 1)
                ax.plot(x[1:], y[1:], color=c, lw=1.0, ls=ls, label=lab if key == "control" else None)
        ax.axhline(0.5, color="gray", lw=0.6, ls=":"); ax.set_title(title, fontsize=8); ax.set_xlabel("relative depth")
        ax.set_ylim(-0.02, 1.02); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("conflict-cell AUC for the stake")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.3), handlelength=1.4, columnspacing=1.0)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)

def fig_patchmap(name="fig_patchmap"):
    """Appendix: token x layer patching map (normalised effect by position across depth) for the patched models."""
    T3 = json.load(open(RES / "sweep3_summary.json"))
    tags = [t for t, _ in INSTRUCT if t in T3 and "patchmap" in T3[t]]
    lab = dict(INSTRUCT)
    names = {"stake_label": "stake label", "stake_end": "stake sentence end", "event_label": "event label",
             "event_verb": "event verb", "event_end": "event end", "response": "response position"}
    cols = ["#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#CC79A7", "#000000"]
    fig, axes = plt.subplots(1, len(tags), figsize=(6.3, 2.0), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, t in zip(np.atleast_1d(axes), tags):
        E = T3[t]["patchmap"]["effects"]
        for (k, nm), c in zip(names.items(), cols):
            y = np.array(E[k]); x = (np.arange(len(y)) + 1) / len(y)
            ax.plot(x, y, color=c, lw=1.1, label=nm)
        ax.set_title(lab[t], fontsize=8); ax.set_xlabel("relative depth"); ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("normalised patching effect")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.2), handlelength=1.4, columnspacing=0.8)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def table_bands():
    """Band-averaged conflict AUC (self) with bootstrap 95% CIs, raw and gen positions."""
    Braw = json.load(open(RES / "band_boot_raw.json"))
    Bgen = json.load(open(RES / "band_boot_gen.json")) if (RES / "band_boot_gen.json").exists() else {}
    f = lambda d: f"{d['conflict_auc']:.2f} [{d['conflict_auc_ci'][0]:.2f}, {d['conflict_auc_ci'][1]:.2f}]"
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r" & \multicolumn{2}{c}{end of text} & \multicolumn{2}{c}{response position} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Model & mid & late & mid & late \\", r"\midrule"]
    for group in (INSTRUCT, BASEM):
        for t, lab in group:
            if t not in Braw:
                continue
            g = Bgen.get(t)
            gm = f(g["self"]["mid"]) if g else "--"
            gl = f(g["self"]["late"]) if g else "--"
            lines.append(f"{lab} & {f(Braw[t]['self']['mid'])} & {f(Braw[t]['self']['late'])} & {gm} & {gl} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (BASE / "outputs" / "table_bands.tex").write_text("\n".join(lines) + "\n")


def fig_gating(name="fig_gating"):
    """Figure (prereg v9): (a) task gating at the response boundary; (b) additive decomposition at the response position."""
    W4 = json.load(open(RES / "sweep4_summary.json")); D = json.load(open(RES / "decompose_summary.json"))
    S = json.load(open(RES / "confirmatory_summary.json"))
    elig = [t for t, _ in INSTRUCT if S[t]["behaviour_conf"]["self"]["acc"] >= 0.85]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.1), gridspec_kw={"wspace": 0.28, "width_ratios": [1.1, 1]})
    W5 = json.load(open(RES / "sweep5_summary.json")) if (RES / "sweep5_summary.json").exists() else {}
    keys = [("tg_eval_self", "good or bad\nfor you?"), ("tg_eval_plain", "good\nor bad?")]
    if W5:
        keys += [("outcome_factual", "did your\nside win?"), ("binding_factual", "your\nside?")]
    keys += [("tg_none", "no\nquestion"), ("tg_factual", "who is\nnamed?"), ("tg_irrelevant", "how many\nsentences?")]
    for (tag, lab), c in zip(INSTRUCT, COLORS):
        if tag not in elig or tag not in W4:
            continue
        y = [(W4[tag] if k.startswith("tg_") else W5[tag])[k]["late"] for k, _ in keys]
        axes[0].plot(range(len(keys)), y, color=c, lw=0.9, marker="o", ms=3, label=lab)
    axes[0].set_xticks(range(len(keys))); axes[0].set_xticklabels([l for _, l in keys], fontsize=5.6)
    axes[0].axhline(0.5, color="gray", lw=0.6, ls=":"); axes[0].set_ylim(-0.02, 1.2)
    axes[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    groups = [(0, 1, "all models"), (2, 3, "task-dependent")] if W5 else [(0, 1, "all models")]
    for lo, hi, txt in groups:
        axes[0].plot([lo - 0.25, hi + 0.25], [1.08, 1.08], color="gray", lw=0.6)
        axes[0].text((lo + hi) / 2, 1.1, txt, ha="center", va="bottom", fontsize=6)
    axes[0].set_ylabel("conflict-cell AUC for the stake")
    axes[0].set_title("(a) question before the response", fontsize=8)
    grid = np.linspace(0.05, 0.95, 37)
    for comp, col, nm in (("beta_s", "#D55E00", "surface $s$"), ("beta_m", "#999999", "label match $m$"), ("beta_o", "#0072B2", "stake outcome $o$")):
        Y = []
        for t in elig:
            v = np.array(D[t]["inf_gen"][comp][:-1], float); x = np.arange(len(v)) / len(v)
            Y.append(np.interp(grid, x, v))
        Y = np.array(Y)
        axes[1].plot(grid, Y.mean(0), color=col, lw=1.3, label=nm)
        axes[1].fill_between(grid, Y.min(0), Y.max(0), color=col, alpha=0.15, lw=0)
    axes[1].axhline(0, color="gray", lw=0.6, ls=":")
    axes[1].set_xlabel("relative depth"); axes[1].set_ylabel("standardised coefficient")
    axes[1].set_title("(b) decomposition at the response position", fontsize=8)
    axes[1].legend(frameon=False, loc="upper left", fontsize=6.5)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.33), handlelength=1.4, columnspacing=1.0)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ── figures and tables of the results section ──────────────────────────────
def _elig():
    S = json.load(open(RES / "confirmatory_summary.json"))
    return S, [t for t, _ in INSTRUCT if S[t]["behaviour_conf"]["self"]["acc"] >= 0.85]


def fig_profiles(name="fig_profiles"):
    """Section 4.3: conflict-cell AUC across depth at the end of the text and at the response position."""
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.0), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, pos, title in ((axes[0], "raw", "(a) end of text"), (axes[1], "gen", "(b) response position")):
        for (tag, lab), c in zip(INSTRUCT, COLORS):
            res = load(tag, pos=pos)
            if res is None:
                continue
            L = res["layers"]; n = len(L) - 1
            x = np.array([l["layer"] / n for l in L]); y = np.array([l["sst2/diffmeans"]["auc_outcome_conflict_self"] for l in L], float)
            ax.plot(x[1:], y[1:], color=c, lw=1.0, label=lab)
        for lo, hi in ((0.2, 0.45), (0.6, 0.9)):
            ax.axvspan(lo, hi, color="#EEEEEE", zorder=0, lw=0)
        ax.axhline(0.5, color="gray", lw=0.6, ls=":"); ax.set_ylim(-0.02, 1.02)
        ax.set_title(title, fontsize=8); ax.set_xlabel("relative depth"); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("conflict-cell AUC for the stake")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.30), handlelength=1.4, columnspacing=1.0)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


def fig_controls(name="fig_controls"):
    """Section 4.4: (a) neutral controls at the response position; (b) goal-relevant key control per model."""
    S, elig = _elig()
    I = json.load(open(RES / "informat_summary.json")); T3 = json.load(open(RES / "sweep3_summary.json"))
    fig, axes = plt.subplots(2, 1, figsize=(3.1, 3.75), gridspec_kw={"hspace": 0.75, "height_ratios": [1.5, 1]})
    ax = axes[0]
    for (tag, lab), c in zip(INSTRUCT, COLORS):
        if tag not in I:
            continue
        for key, ls in (("location_gen", "-"), ("material_gen", ":")):
            y = np.array(I[tag][key]["profile"], float); x = np.arange(len(y)) / (len(y) - 1)
            ax.plot(x[1:], y[1:], color=c, lw=0.9, ls=ls)
    ax.plot([], [], color="gray", ls="-", label="location"); ax.plot([], [], color="gray", ls=":", label="material")
    ax.axhline(0.5, color="gray", lw=0.6, ls=":"); ax.set_ylim(-0.02, 0.75); ax.set_yticks([0, 0.25, 0.5, 0.75])
    ax.set_xlabel("relative depth"); ax.set_ylabel("conflict-cell AUC")
    ax.set_title("(a) neutral controls, response position", fontsize=8)
    ax.legend(frameon=False, fontsize=6, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    tags = [t for t, _ in INSTRUCT if t in elig]; lab = dict(INSTRUCT)
    xs = np.arange(len(tags)); w = 0.38
    v = [T3[t]["gn_sst_gen"]["late"] for t in tags]; ci = np.array([T3[t]["gn_sst_gen"]["late_ci"] for t in tags])
    ax.bar(xs - w / 2, v, w, color="#0072B2", label="valence (SST-2)", yerr=[np.array(v) - ci[:, 0], ci[:, 1] - np.array(v)], error_kw={"lw": 0.6})
    ax.bar(xs + w / 2, [T3[t]["gn_inf_gen"]["late"] for t in tags], w, color="#BBBBBB", label="brass vs steel")
    ax.axhline(0.5, color="gray", lw=0.6, ls=":"); ax.set_ylim(0, 1.05)
    ax.set_xticks(xs); ax.set_xticklabels([lab[t].replace("-Inst.", "") for t in tags], rotation=35, ha="right", fontsize=5.8)
    ax.set_ylabel("late-band AUC"); ax.set_title("(b) goal-relevant key control", fontsize=8)
    ax.set_title("(b) goal-relevant key control", fontsize=8, pad=14)
    ax.legend(frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, borderaxespad=0.1); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


def fig_task(name="fig_task"):
    """Section 4.5: conflict-cell AUC at the response position after different questions."""
    S, elig = _elig()
    W4 = json.load(open(RES / "sweep4_summary.json")); W5 = json.load(open(RES / "sweep5_summary.json"))
    keys = [("tg_eval_self", "good or bad\nfor you?"), ("tg_eval_plain", "good\nor bad?"), ("outcome_factual", "did your\nside win?"),
            ("binding_factual", "your\nside?"), ("tg_none", "no\nquestion"), ("tg_factual", "who is\nnamed?"), ("tg_irrelevant", "how many\nsentences?")]
    fig, ax = plt.subplots(figsize=(3.1, 2.0))
    for (tag, lab), c in zip(INSTRUCT, COLORS):
        if tag not in elig:
            continue
        y = [(W4[tag] if k.startswith("tg_") else W5[tag])[k]["late"] for k, _ in keys]
        ax.plot(range(len(keys)), y, color=c, lw=0.9, marker="o", ms=2.5, label=lab)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([l for _, l in keys], fontsize=5.6)
    ax.axhline(0.5, color="gray", lw=0.6, ls=":"); ax.set_ylim(-0.02, 1.04); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("late-band AUC"); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


def fig_patch(name="fig_patch"):
    """Section 4.6: normalised effect of patching the stake label at the label tokens and at the response position."""
    W = json.load(open(RES / "sweep2_summary.json"))
    fig, ax = plt.subplots(figsize=(3.1, 1.7))
    for (tag, lab), c in zip(INSTRUCT, COLORS):
        if tag not in W or "patching" not in W[tag]:
            continue
        El, Ef = np.array(W[tag]["patching"]["E_label"]), np.array(W[tag]["patching"]["E_final"]); x = (np.arange(len(El)) + 1) / len(El)
        ax.plot(x, El, color=c, lw=0.9, ls="--"); ax.plot(x, Ef, color=c, lw=1.0)
    ax.plot([], [], color="gray", ls="--", label="at the stake label"); ax.plot([], [], color="gray", ls="-", label="at the response position")
    ax.set_xlabel("relative depth"); ax.set_ylabel("normalised effect"); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


def fig_prior(name="fig_prior"):
    """Section 4.8: additive decomposition of the valence readout at the response position."""
    S, elig = _elig()
    D = json.load(open(RES / "decompose_summary.json"))
    fig, ax = plt.subplots(figsize=(3.1, 1.75))
    grid = np.linspace(0.05, 0.95, 37)
    for comp, col, nm in (("beta_s", "#D55E00", "surface $s$"), ("beta_m", "#999999", "label match $m$"), ("beta_o", "#0072B2", "stake outcome $o$")):
        Y = np.array([np.interp(grid, np.arange(len(D[t]["inf_gen"][comp]) - 1) / (len(D[t]["inf_gen"][comp]) - 1), D[t]["inf_gen"][comp][:-1]) for t in elig])
        ax.plot(grid, Y.mean(0), color=col, lw=1.2, label=nm); ax.fill_between(grid, Y.min(0), Y.max(0), color=col, alpha=0.15, lw=0)
    ax.axhline(0, color="gray", lw=0.5, ls=":"); ax.set_xlabel("relative depth"); ax.set_ylabel("coefficient")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


def table_transition():
    """Section 4.3: band averages, late-minus-middle CI, and transition depth per model."""
    S = json.load(open(RES / "confirmatory_summary.json"))
    Br = json.load(open(RES / "band_boot_raw.json")); Bg = json.load(open(RES / "band_boot_gen.json")); W = json.load(open(RES / "sweep2_summary.json"))
    rows = [r"\begin{tabular}{lccccc}", r"\toprule",
            r" & \multicolumn{2}{c}{end of text} & \multicolumn{3}{c}{response position} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
            r"Model & mid & late & mid & late & $d^*$ \\", r"\midrule"]
    for t_, lab in INSTRUCT + BASEM:
        if t_ not in Br:
            continue
        star = "" if S[t_]["behaviour_conf"]["self"]["acc"] >= 0.85 else r"$^\dagger$"
        r = Br[t_]["self"]; g = Bg.get(t_, {}).get("self")
        fit = (W.get(t_, {}).get("valence_gen", {}) or {}).get("fit") or {}
        resp = (f"{g['mid']['conflict_auc']:.2f} & {g['late']['conflict_auc']:.2f} & "
                + (f"{fit['d_star']:.2f}" if fit else "--")) if g else "-- & -- & --"
        rows.append(f"{lab}{star} & {r['mid']['conflict_auc']:.2f} & {r['late']['conflict_auc']:.2f} & {resp} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (BASE / "outputs" / "table_transition.tex").write_text("\n".join(rows) + "\n")


def table_behaviour():
    """Section 4.8: behavioural accuracy, surface bias, and regression coefficients per model."""
    S = json.load(open(RES / "confirmatory_summary.json")); PR = json.load(open(RES / "prior_regression.json"))
    rows = [r"\begin{tabular}{lcccccc}", r"\toprule",
            r"Model & acc. & cong. & confl. & $B_{\mathrm{surf}}$ & $\beta_A$ & $\beta_L$ \\", r"\midrule"]
    for t_, lab in INSTRUCT + BASEM:
        if t_ not in S or t_ not in PR:
            continue
        b = S[t_]["behaviour_conf"]["self"]; star = "" if b["acc"] >= 0.85 else r"$^\dagger$"
        rows.append(f"{lab}{star} & {b['acc']:.2f} & {b['acc_congruent']:.2f} & {b['acc_conflict']:.2f} & {b['B_surf']:+.2f} & "
                    f"{PR[t_]['bA']:.1f} & {PR[t_]['bL']:.1f} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (BASE / "outputs" / "table_behaviour.tex").write_text("\n".join(rows) + "\n")


def table_leak():
    """Section 4.2: lexical leakage in the probing data versus our stimuli."""
    L = json.load(open(RES / "leak_audit.json")); tags = [t for t in L if not t.startswith("_")]
    rows = [r"\begin{tabular}{lccc}", r"\toprule", r"Data & bag of words & best probe & ratio \\", r"\midrule"]
    for ds, nm in (("sst2", "SST-2"), ("goemo", "GoEmotions")):
        b = [L[t][ds]["best"] for t in tags]; r = [L[t][ds]["bow_ratio"] for t in tags]
        rows.append(f"{nm} & {L['_bow'][ds]:.2f} & {min(b):.2f}--{max(b):.2f} & {min(r):.2f}--{max(r):.2f} \\\\")
    rows.append(r"Our stimuli & 0.50 & -- & -- \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (BASE / "outputs" / "table_leak.tex").write_text("\n".join(rows) + "\n")


def table_natural():
    """Section 4.7: naturalistic sentences per model."""
    S = json.load(open(RES / "confirmatory_summary.json"))
    W = json.load(open(RES / "sweep2_summary.json")); T3 = json.load(open(RES / "sweep3_summary.json"))
    rows = [r"\begin{tabular}{lccccc}", r"\toprule",
            r" & \multicolumn{2}{c}{end of text} & \multicolumn{2}{c}{response} & \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r"Model & mid & late & mid & late & agree \\", r"\midrule"]
    for t, lab in INSTRUCT:
        if t not in W or "natural_gen" not in W[t]:
            continue
        star = "" if S[t]["behaviour_conf"]["self"]["acc"] >= 0.85 else r"$^\dagger$"
        a = T3.get(t, {}).get("natval", {}).get("agree")
        rows.append(f"{lab}{star} & {W[t]['natural_raw']['mid']:.2f} & {W[t]['natural_raw']['late']:.2f} & "
                    f"{W[t]['natural_gen']['mid']:.2f} & {W[t]['natural_gen']['late']:.2f} & {('--' if a is None else f'{a:.2f}')} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (BASE / "outputs" / "table_natural.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    fig_profiles(); fig_controls(); fig_task(); fig_patch(); fig_prior(); table_leak(); table_natural(); table_transition(); table_behaviour()
    fig_gating()
    fig_mechanism()
    fig_patchmap()
    fig_controls_source()
    fig_informat()
    table_prompts()
    table_fits()
    table_summary()
    fig_conflict()
    if (RES / "band_boot_raw.json").exists():
        table_bands()
    table_ablation()
    table_main()
    fig_depth()
    fig_depth(est="goemo/diffmeans", name="fig_depth_goemo")
    fig_behaviour()
    numbers()
    print(sorted(p.name for p in FIG.glob("*.pdf")))
