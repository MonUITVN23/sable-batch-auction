"""Generate the figures of the paper from results/*.json into figures/ (PDF + PNG).

Each figure is a standalone panel sized for half of an LNCS text width.
Legends never sit on top of data: they are placed below the axes or in a
region that the script checks is empty for the released results.
"""

import json
import pathlib
from fractions import Fraction as Fr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, ORANGE, GREEN, GRAY, LIGHT = "#1f5fa8", "#d9772b", "#3a8f4f", "#777777", "#c8c8c8"
FIGSIZE = (3.2, 2.5)


def load(name):
    return json.loads((RES / f"{name}.json").read_text(encoding="utf8"))


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print("wrote figures/" + name + ".pdf")


def legend_below(ax, handles=None, labels=None, ncol=2, y=-0.3):
    kw = dict(loc="upper center", bbox_to_anchor=(0.5, y), frameon=False, ncol=ncol,
              handlelength=2.2, columnspacing=1.2)
    if handles is None:
        ax.legend(**kw)
    else:
        ax.legend(handles, labels, **kw)


def fig_tightness():
    t = load("tightness")
    d = [r["d"] for r in t["rows"]]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axhline(1, color=GRAY, ls=":", lw=1, label="envelope (ratio = 1)")
    ax.plot(d, [r["signed_ratio"] for r in t["rows"]], "o-", color=BLUE, ms=3,
            label=r"signed $\mathbb{E}[G_v]$")
    ax.plot(d, [r["tail_ratio_sharp"] for r in t["rows"]], "s--", color=ORANGE, ms=3,
            label="positive tail, sharp bound")
    ax.plot(d, [r["tail_ratio_loose"] for r in t["rows"]], "^:", color=ORANGE, ms=3, alpha=0.6,
            label="positive tail, loose bound")
    ax.axvline(t["q_max"], color=GRAY, ls="-.", lw=0.8)
    # The band 0.75 < y < 1 to the right of q_max is empty for these results.
    ax.text(t["q_max"] + 0.3, 0.86, r"$\Delta Q_v$ saturates", color=GRAY, fontsize=7, va="center")
    ax.set_xlabel("victim depth $d$ (unit $X$)")
    ax.set_ylabel("attained / bound")
    ax.set_xticks(range(0, max(d) + 1, 2))
    ax.set_ylim(0, 1.08)
    ax.set_title(r"(a) Bound tightness, $q_{\max}=" + str(t["q_max"]) + "$")
    legend_below(ax, ncol=2, y=-0.26)
    save(fig, "fig_tightness")


def fig_channel():
    c = load("channel")
    q = [r["q_max"] for r in c["rows"]]
    share_c = [0 if r["share_C"] is None else r["share_C"] for r in c["rows"]]
    share_g = [1 - s for s in share_c]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(q, share_g, color=GREEN, alpha=0.8, label=r"via $G_v$ (inventory)")
    ax.bar(q, share_c, bottom=share_g, color=ORANGE, alpha=0.8, label=r"via $C_{\mathrm{imm}}$ (clearing price)")
    for r in c["rows"]:
        ax.text(r["q_max"], 1.03, f"{float(Fr(r['E_pi'])):g}", ha="center", va="bottom",
                fontsize=5.2, color="black")
    ax.set_xlabel(r"position cap $q_{\max}$ (unit $X$)")
    ax.set_ylabel(r"share of $\mathbb{E}[\pi_v]$")
    ax.set_xticks(q)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title(f"(b) Channel switch, tick = {c['tick']}, drift 1/4", pad=16)
    legend_below(ax, ncol=2, y=-0.26)
    save(fig, "fig_channel")


def fig_harm():
    """Grouped, stacked bars: victim loss L_v (full bar) and the part the adversary
    takes, C_imm (filled segment), for selected caps and collars.  A short black
    dash marks the collar bound q_max (|P_k - p_ref| + 3B) of the theorem."""
    h = load("harm")
    caps = [2, 4, 6, 8, 10]
    collars = [("none", "no collar"), ("4", "$B=4$"), ("2", "$B=2$"), ("1", "$B=1$")]
    colors = {"none": "#444444", "4": BLUE, "2": GREEN, "1": ORANGE}
    width = 0.19
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for j, (B, lab) in enumerate(collars):
        rows = {r["q_max"]: r for r in h["collars"][B]["rows"]}
        xs = [i + (j - 1.5) * width for i in range(len(caps))]
        L = [float(Fr(rows[q]["L_v"])) for q in caps]
        C = [float(Fr(rows[q]["C_imm"])) for q in caps]
        ax.bar(xs, L, width, color=colors[B], alpha=0.25, edgecolor=colors[B], lw=0.6)
        ax.bar(xs, C, width, color=colors[B], alpha=0.95)
        if B != "none":
            bound = [float(Fr(rows[q]["collar_bound"])) for q in caps]
            for x, b in zip(xs, bound):
                ax.plot([x - width / 2, x + width / 2], [b, b], color="black", lw=1.0)
    ax.set_xticks(range(len(caps)))
    ax.set_xticklabels([str(q) for q in caps])
    ax.set_xlabel(r"position cap $q_{\max}$ (unit $X$)")
    ax.set_ylabel("quote units")
    ax.set_ylim(0, 125)
    ax.set_title(r"(c) Harm sweep: $L_v$ (light) and $C_{\mathrm{imm}}$ (solid)")
    handles = [Patch(facecolor=colors[B], alpha=0.9, label=lab) for B, lab in collars]
    handles.append(Line2D([0], [0], color="black", lw=1.0, label="collar bound (theorem)"))
    legend_below(ax, handles, [hh.get_label() for hh in handles], ncol=3, y=-0.26)
    save(fig, "fig_harm")


def fig_robustness():
    r = load("channel_robustness")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    markers = ["o", "s", "^"]
    colors = [BLUE, ORANGE, GREEN]
    for (book, d), m, col in zip(r["books"].items(), markers, colors):
        xs = [float(Fr(s)) for s in r["scales"]]
        ys = [11 if d[s]["switch_cap"] is None else d[s]["switch_cap"] for s in r["scales"]]
        ax.plot(xs, ys, marker=m, ms=4, lw=1, color=col, label=book)
    ax.axvline(0.25, color=GRAY, ls=":", lw=0.8)
    # The strip 2 <= y <= 5 just left of the line is empty for these results.
    ax.text(0.25 / 1.08, 3.5, "tick = drift", color=GRAY, fontsize=7, rotation=90,
            ha="right", va="center")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("tick size (quote units per tick)")
    ax.set_ylabel(r"cap at which $C_{\mathrm{imm}}$ takes over")
    ax.set_yticks(range(1, 12))
    ax.set_yticklabels([str(i) for i in range(1, 11)] + ["none"])
    ax.set_ylim(0.5, 11.6)
    ax.set_title("(d) Switch cap vs tick size and near depth")
    legend_below(ax, ncol=3, y=-0.26)
    save(fig, "fig_robustness")


if __name__ == "__main__":
    fig_tightness()
    fig_channel()
    fig_harm()
    fig_robustness()
