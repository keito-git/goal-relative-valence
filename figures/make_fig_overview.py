"""Figure 1 (overview), restrained style with framed panels.
(a)-(d) describe the method; (e) is a separate frame summarising the account supported by the results.
Colours: orange = wording, blue = stake (outcome for the stakeholder), grey = everything else.

Output: outputs/figures/fig_overview.pdf
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

BASE = Path(__file__).resolve().parents[1]
(BASE / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
OUT = BASE / "outputs" / "figures" / "fig_overview.pdf"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"], "mathtext.fontset": "cm",
                     "pdf.fonttype": 42, "font.size": 7, "axes.linewidth": 0.6})
WORD, STAKE, GREY, DARK, FRAME = "#D55E00", "#0072B2", "#8C8C8C", "#222222", "#B5B5B5"

fig = plt.figure(figsize=(6.3, 2.6))
TOP_B, TOP_H = 0.275, 0.72            # top row (figure fraction)
X0 = [0.004, 0.2585, 0.513, 0.7675]; PW = 0.2265


def frame(x, y, w, h, title):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008", transform=fig.transFigure,
                                      fill=False, ec=FRAME, lw=0.7))
    fig.text(x + 0.008, y + h - 0.024, title, fontsize=7.2, fontweight="bold", color=DARK, ha="left", va="top")


def text_runs(ax, x, y, runs, fs=6.5):
    r = fig.canvas.get_renderer()
    for s, c, ul in runs:
        t = ax.text(x, y, s, fontsize=fs, color=c, ha="left", va="baseline")
        bb = t.get_window_extent(r).transformed(ax.transAxes.inverted())
        if ul:
            ax.plot([bb.x0, bb.x1], [y - 0.025, y - 0.025], color=WORD, lw=1.0)
        x = bb.x1
    return x


titles = ["(a) Stimuli", "(b) Stake-free valence direction", "(c) Readout positions", "(d) Word or stake?"]
for x, t in zip(X0, titles):
    frame(x, TOP_B, PW, TOP_H, t)
# inner axes leave room for the panel title
# axes[1] holds the readout panel and axes[2] the direction panel; they are placed as (c) and (b)
axes = [fig.add_axes([x + 0.006, TOP_B + 0.010, PW - 0.012, TOP_H - 0.098]) for x in (X0[0], X0[2], X0[1])]
for ax in axes:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# (a) Stimuli -----------------------------------------------------------------------------------------
ax = axes[0]
for y0, side, outcome, note in ((0.70, "Red", "good for you", None), (0.36, "Blue", "bad for you", "wording and stake disagree")):
    ax.add_patch(Rectangle((0.03, y0 - 0.02), 0.94, 0.30, fill=False, ec=FRAME, lw=0.6))
    text_runs(ax, 0.07, y0 + 0.18, [(f"You support {side}. ", DARK, False), ("Red ", DARK, False), ("won", WORD, True), (".", DARK, False)])
    ax.text(0.07, y0 + 0.075, r"$\rightarrow$ " + outcome, fontsize=6.5, color=STAKE, ha="left", va="baseline")
    if note:
        ax.text(0.07, y0 + 0.005, note, fontsize=5.6, color=GREY, ha="left", va="baseline", style="italic")
ax.plot([0.04, 0.10], [0.235, 0.235], color=WORD, lw=1.0)
ax.text(0.13, 0.235, "wording: positive in both", fontsize=6.2, color=WORD, ha="left", va="center")
ax.text(0.04, 0.115, r"$\rightarrow$", fontsize=6.4, color=STAKE, ha="left", va="center")
ax.text(0.13, 0.115, "stake: is the outcome good for", fontsize=6.2, color=STAKE, ha="left", va="center")
ax.text(0.13, 0.02, "the stakeholder?", fontsize=6.2, color=STAKE, ha="left", va="center")

# (b) Readout positions -------------------------------------------------------------------------------
ax = axes[1]
toks = ["You", "\u2026", "Red", "won", ".", "chat"]
ncol, nrow = len(toks), 5
gx0, gw, gy0, gh = 0.15, 0.78, 0.33, 0.47
cw, ch = gw / ncol, gh / nrow
for i in range(ncol):
    for j in range(nrow):
        col = "#F2C9AD" if i == 4 else ("#B9D5EA" if i == 5 else "#E6E6E6")
        ax.add_patch(Rectangle((gx0 + i * cw + 0.006, gy0 + j * ch + 0.006), cw - 0.012, ch - 0.012, color=col, lw=0))
    ax.add_patch(Rectangle((gx0 + i * cw + 0.006, 0.20), cw - 0.012, 0.09, fill=False, ec=STAKE if i == 5 else FRAME, lw=0.6))
    ax.text(gx0 + (i + 0.5) * cw, 0.245, toks[i], fontsize=5.8, ha="center", va="center", color=DARK)
for j, lab in ((0, "1"), (2, r"$\ell$"), (4, "$L$")):
    ax.text(gx0 - 0.03, gy0 + (j + 0.5) * ch, lab, fontsize=6.3, ha="right", va="center", color=GREY)
ax.text(gx0 - 0.02, gy0 + gh + 0.01, "layer", fontsize=5.6, ha="right", va="bottom", color=GREY)
for i, lab, c, dx in ((4, "end of text", WORD, -0.10), (5, "response position", STAKE, 0.0)):
    x = gx0 + (i + 0.5) * cw
    ax.annotate("", xy=(x, gy0 + gh + 0.005), xytext=(x, gy0 + gh + 0.075),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=0.7, mutation_scale=6))
ax.text(gx0 + 4.5 * cw - 0.01, gy0 + gh + 0.09, "end of text", fontsize=6, color=WORD, ha="right", va="bottom")
ax.text(gx0 + 5.5 * cw + 0.01, gy0 + gh + 0.09, "response\nposition", fontsize=6, color=STAKE, ha="center", va="bottom", linespacing=1.0)
ax.text(0.02, 0.115, "end of text: last token of the stimulus", fontsize=5.6, color=WORD, ha="left", va="center")
ax.text(0.02, 0.03, "response position: start of the answer", fontsize=5.7, color=STAKE, ha="left", va="center")

# (c) Stake-free valence direction -------------------------------------------------------------------
ax = axes[2]
rng = np.random.default_rng(3)
pos = rng.normal([0.66, 0.70], 0.03, (12, 2)); neg = rng.normal([0.22, 0.48], 0.03, (12, 2))
ax.scatter(pos[:, 0], pos[:, 1], marker="+", s=15, color=DARK, lw=0.7)
ax.scatter(neg[:, 0], neg[:, 1], marker="_", s=15, color=DARK, lw=0.9)
mp, mn = pos.mean(0), neg.mean(0)
ax.add_patch(FancyArrowPatch(mn, mp, arrowstyle="-|>", mutation_scale=7, color=DARK, lw=0.9))
ax.text(0.66, 0.82, "positive texts", fontsize=5.9, color=DARK, ha="center", va="bottom")
ax.text(0.22, 0.39, "negative texts", fontsize=5.9, color=DARK, ha="center", va="top")
ax.text(0.42, 0.66, "valence\ndirection $w_\\ell$", fontsize=5.9, color=DARK, ha="right", va="center", linespacing=1.0)
q = np.array([0.74, 0.40]); u = (mp - mn) / np.linalg.norm(mp - mn); proj = mn + np.dot(q - mn, u) * u
ax.plot(*q, marker="o", ms=3.2, color=STAKE); ax.plot([q[0], proj[0]], [q[1], proj[1]], ls=":", color=STAKE, lw=0.8)
ax.text(q[0] + 0.04, q[1], "stimulus,\nscore $p_\\ell$", fontsize=5.9, color=STAKE, ha="left", va="center", linespacing=1.0)
ax.text(0.02, 0.19, "Learned from texts without a stakeholder,", fontsize=5.5, color=DARK, ha="left", va="center")
ax.text(0.02, 0.105, "the direction cannot encode a stake;", fontsize=5.5, color=DARK, ha="left", va="center")
ax.text(0.02, 0.02, "each stimulus is scored by projection.", fontsize=5.5, color=DARK, ha="left", va="center")

# (d) Word or stake? ---------------------------------------------------------------------------------
ax = fig.add_axes([X0[3] + 0.048, TOP_B + 0.15, PW - 0.062, TOP_H - 0.245])
x = np.linspace(0, 1, 200)
dip = 0.5 - 0.45 * np.exp(-((x - 0.30) / 0.11) ** 2)
resp = np.where(x < 0.36, dip, np.maximum(dip, 0.05 + 0.92 / (1 + np.exp(-(x - 0.56) / 0.06))))
eot = np.where(x < 0.36, dip, np.maximum(dip, 0.05 + 0.50 / (1 + np.exp(-(x - 0.60) / 0.07))))
ax.axhspan(0.85, 1.05, color="#E8F0F7", lw=0); ax.axhspan(-0.03, 0.15, color="#FBEDE4", lw=0)
ax.plot(x, resp, color=STAKE, lw=1.2); ax.plot(x, eot, color=WORD, lw=1.2, ls="--")
ax.axhline(0.5, color=GREY, lw=0.5, ls=":")
ax.set_xlim(0, 1); ax.set_ylim(-0.03, 1.05); ax.set_xticks([0, 1]); ax.set_yticks([0, 0.5, 1])
ax.tick_params(labelsize=5.8, length=2, pad=1.5); ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("relative depth", fontsize=6, labelpad=0.5); ax.set_ylabel("AUC", fontsize=6, labelpad=1)
ax.text(0.03, 0.99, "follows the stake", fontsize=5.8, color=STAKE, ha="left", va="top", style="italic")
ax.text(0.97, 0.02, "follows the wording", fontsize=5.8, color=WORD, ha="right", va="bottom", style="italic")
ax.text(0.02, 0.53, "neither", fontsize=5.4, color=GREY, ha="left", va="bottom")
ax.text(0.98, 0.80, "response", fontsize=5.8, color=STAKE, ha="right", va="center")
ax.text(0.98, 0.45, "end of text", fontsize=5.8, color=WORD, ha="right", va="center")
fig.text(X0[3] + PW / 2, TOP_B + 0.03, "items where wording and stake disagree", fontsize=5.7, color=GREY, ha="center",
         va="center", style="italic")

# order of the method: (a) -> (b) -> (c) -> (d); (d) is summarised in (e)
ymid = TOP_B + TOP_H * 0.5
for i in range(3):
    fig.patches.append(FancyArrowPatch((X0[i] + PW + 0.0005, ymid), (X0[i + 1] - 0.0005, ymid), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=7, color=GREY, lw=0.9))
fig.patches.append(FancyArrowPatch((X0[3] + PW / 2, TOP_B - 0.004), (X0[3] + PW / 2, 0.203), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=7, color=GREY, lw=0.9))
fig.text(X0[3] + PW / 2 + 0.008, (TOP_B + 0.200) / 2, "summarised in (e)", fontsize=5.6, color=GREY, ha="left", va="center", style="italic")

# (e) Account supported by the results ---------------------------------------------------------------
frame(0.004, 0.005, 0.990, 0.195, "(e) Account supported by the results")
axb = fig.add_axes([0.01, 0.010, 0.98, 0.135]); axb.set_xlim(0, 1); axb.set_ylim(0, 1); axb.axis("off")
items = [("lexical valuation", "'Red won' is positive", "middle layers", WORD),
         ("relational binding", "'You' = Blue", "later layers", GREY),
         ("goal-relative appraisal", "'Red won' is bad for you", "response position", STAKE),
         ("judgement", "'bad', yet pulled by the wording", "output", DARK)]
cx = [0.12, 0.38, 0.64, 0.89]
for (name, ex, where, c), x0 in zip(items, cx):
    axb.text(x0, 0.66, name, fontsize=6.8, color=c, ha="center", va="center", fontweight="bold")
    axb.text(x0, 0.22, f"{ex}  ({where})", fontsize=5.9, color=DARK, ha="center", va="center", style="italic")
for a, b in zip(cx[:-1], cx[1:]):
    axb.add_patch(FancyArrowPatch((a + 0.105, 0.66), (b - 0.105, 0.66), arrowstyle="-|>", mutation_scale=7, color=GREY, lw=0.7))


fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
print(OUT)
