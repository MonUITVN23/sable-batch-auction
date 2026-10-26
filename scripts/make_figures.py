"""Generate the figures of the paper from results/*.json into figures/ (PDF + PNG)."""

import json
import pathlib
from fractions import Fraction as Fr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "figure.dpi": 150})
BLUE, ORANGE, GREEN, GRAY = "#1f5fa8", "#d9772b", "#3a8f4f", "#777777"


def load(name):
    return json.loads((RES / f"{name}.json").read_text(encoding="utf8"))


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/" + name + ".pdf")


def fig_tightness():
    t = load("tightness")
    d = [r["d"] for r in t["rows"]]
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    ax.axhline(1, color=GRAY, ls=":", lw=1, label="envelope (ratio = 1)")
    ax.plot(d, [r["signed_ratio"] for r in t["rows"]], "o-", color=BLUE, ms=3, label=r"signed $\mathbb{E}[G_v]$")
    ax.plot(d, [r["tail_ratio_sharp"] for r in t["rows"]], "s--", color=ORANGE, ms=3, label=r"positive tail, sharp bound")
    ax.plot(d, [r["tail_ratio_loose"] for r in t["rows"]], "^:", color=ORANGE, ms=3, alpha=0.6, label=r"positive tail, loose bound")
    ax.axvline(t["q_max"], color=GRAY, ls="-.", lw=0.8)
    ax.text(t["q_max"] + 0.2, 0.05, r"$\Delta Q_v$ saturates", color=GRAY, fontsize=7)
    ax.set_xlabel("victim depth $d$ (unit $X$)")
    ax.set_ylabel("attained / bound")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"(a) Bound tightness, single-victim fixture, $q_{{\\max}}={t['q_max']}$")
    ax.legend(loc="center right", frameon=False)
    save(fig, "fig_tightness")


def fig_channel():
    c = load("channel")
    q = [r["q_max"] for r in c["rows"]]
    share_c = [0 if r["share_C"] is None else r["share_C"] for r in c["rows"]]
    share_g = [1 - s for s in share_c]
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    ax.bar(q, share_g, color=GREEN, alpha=0.75, label=r"via $G_v$ (inventory)")
    ax.bar(q, share_c, bottom=share_g, color=ORANGE, alpha=0.75, label=r"via $C_{\mathrm{imm}}$ (clearing price)")
    for r in c["rows"]:
        ax.text(r["q_max"], 1.02, f"{float(Fr(r['E_pi'])):.2g}", ha="center", fontsize=6, color=GRAY)
    ax.set_xlabel(r"position cap $q_{\max}$ (unit $X$)")
    ax.set_ylabel(r"share of $\mathbb{E}[\pi_v]$")
    ax.set_xticks(q)
    ax.set_ylim(0, 1.15)
    ax.set_title(f"(b) Channel switch, tick = {c['tick']}, drift 1/4 per unit")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False, ncol=2)
    save(fig, "fig_channel")


def fig_harm():
    h = load("harm")
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    styles = {"none": ("-", "k"), "8": ("--", ORANGE), "4": ("-.", BLUE), "2": (":", GREEN), "1": ((0, (1, 1)), GRAY)}
    for B, sweep in h["collars"].items():
        q = [r["q_max"] for r in sweep["rows"]]
        ls, col = styles[B]
        ax.plot(q, [float(Fr(r["L_v"])) for r in sweep["rows"]], ls=ls, color=col, lw=1.2,
                label=("no collar" if B == "none" else f"collar $B={B}$"))
        ax.plot(q, [float(Fr(r["C_imm"])) for r in sweep["rows"]], ls=ls, color=col, lw=1.2, marker="o", ms=2.5)
    ax.set_xlabel(r"position cap $q_{\max}$ (unit $X$)")
    ax.set_ylabel("quote units")
    ax.set_title("(c) Victim loss $L_v$ (lines) and $C_{\\mathrm{imm}}$ (markers)")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    save(fig, "fig_harm")


def fig_robustness():
    r = load("channel_robustness")
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    markers = ["o", "s", "^"]
    for (book, d), m in zip(r["books"].items(), markers):
        xs = [float(Fr(s)) for s in r["scales"]]
        ys = [11 if d[s]["switch_cap"] is None else d[s]["switch_cap"] for s in r["scales"]]
        ax.plot(xs, ys, marker=m, ms=4, lw=1, label=book)
    ax.axvline(0.25, color=GRAY, ls=":", lw=0.8)
    ax.text(0.26, 1.3, "tick = drift", color=GRAY, fontsize=7)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("tick size (quote units per tick)")
    ax.set_ylabel(r"cap at which $C_{\mathrm{imm}}$ takes over")
    ax.set_yticks(range(1, 12))
    ax.set_yticklabels([str(i) for i in range(1, 11)] + ["none"])
    ax.set_title("(d) Switch cap vs tick size and near depth")
    ax.legend(frameon=False)
    save(fig, "fig_robustness")


if __name__ == "__main__":
    fig_tightness()
    fig_channel()
    fig_harm()
    fig_robustness()
