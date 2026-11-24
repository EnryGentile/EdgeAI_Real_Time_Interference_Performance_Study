#!/usr/bin/env python3
"""
plot_quality.py - Grafici delle metriche qualitative prodotte da evaluate_quality.py.

Legge <quality_tasks.csv> e confronta ogni risposta sotto stressor con la risposta della baseline
sullo stesso prompt (stesso modello, device, benchmark e sample_id; riferimento: la ripetizione di
baseline con il numero più basso, come in evaluate_quality.py).

Grafici prodotti (uno per file, in --out):
  1_identiche.png             % di risposte identiche / non identiche alla baseline, per configurazione
  2_bertscore_non_identiche.png  distribuzione del BERTScore vs baseline delle sole risposte non identiche
  3_formato_correttezza.png   % format_ok e correct delle non identiche: sotto stress e in baseline
  4_transizioni.png           come cambia la correttezza rispetto alla baseline (non identiche)
  5_identiche_famiglia_N.png  % di identiche per famiglia di stressor x N (dipendenza dal carico)
e il riepilogo numerico quality_identity_summary.csv.

Per MMLU le metriche sono format_ok / correct, per Spider sql_valid / exec_match.

Esempio:
  python statistical_analysis/plot_quality.py --tasks figures/quality_reports/mmlu/quality_tasks.csv
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

BINS = [(1.0, np.inf, "= 1,00"), (0.95, 1.0, "0,95–1,00"), (0.90, 0.95, "0,90–0,95"),
        (0.85, 0.90, "0,85–0,90"), (0.80, 0.85, "0,80–0,85"), (-np.inf, 0.80, "< 0,80")]
METRICS = {"mmlu": ("format_ok", "correct"), "spider": ("sql_valid", "exec_match")}
N_ORDER = ["N25", "N50", "N75", "N100"]
C_SAME, C_DIFF = "#2E6DB4", "#E07B39"


def cfg_label(model: str, device: str) -> str:
    return f"{model.upper()} {device.upper()}"


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def load(tasks_csv: Path):
    rows = list(csv.DictReader(open(tasks_csv, encoding="utf-8")))
    bench = Counter(r["benchmark"] for r in rows).most_common(1)[0][0]
    # riferimento baseline: ripetizione più bassa per ogni prompt
    base = {}
    for r in sorted((r for r in rows if r["exp_id"].startswith("baseline_") and r["status"] == "ok"),
                    key=lambda r: -int(r["repetition"] or 0)):
        base[(r["model_id"].lower(), r["device"], r["benchmark"], r["sample_id"])] = r
    stress = []
    for r in rows:
        if r["exp_id"].startswith("baseline_") or r["status"] != "ok" or r["identical_to_base"] == "":
            continue
        b = base.get((r["model_id"].lower(), r["device"], r["benchmark"], r["sample_id"]))
        if b is None:
            continue
        r["_base"] = b
        r["_cfg"] = cfg_label(r["model_id"], r["device"])
        stress.append(r)
    return bench, stress


def summarize(bench, stress):
    fmt_k, cor_k = METRICS.get(bench, METRICS["mmlu"])
    groups = defaultdict(list)
    for r in stress:
        groups[r["_cfg"]].append(r)
    order = sorted(groups) + ["Totale"]
    groups["Totale"] = stress
    out = {}
    for g in order:
        rs = groups[g]
        nd = [r for r in rs if r["identical_to_base"] == "0"]
        bins = Counter()
        for r in nd:
            v = r.get("bert_f1_vs_base")
            if v in (None, ""):
                bins["n/d"] += 1
                continue
            v = float(v)
            for lo, hi, lab in BINS[1:]:
                if lo <= v < hi:
                    bins[lab] += 1
                    break
        tr = Counter((r["_base"][cor_k], r[cor_k]) for r in nd)
        out[g] = {
            "n": len(rs), "n_diff": len(nd),
            "ident_pct": pct(len(rs) - len(nd), len(rs)), "diff_pct": pct(len(nd), len(rs)),
            "bert": {lab: pct(bins[lab], len(nd)) for _, _, lab in BINS[1:]},
            "fmt_stress": pct(sum(r[fmt_k] == "1" for r in nd), len(nd)),
            "fmt_base": pct(sum(r["_base"][fmt_k] == "1" for r in nd), len(nd)),
            "cor_stress": pct(sum(r[cor_k] == "1" for r in nd), len(nd)),
            "cor_base": pct(sum(r["_base"][cor_k] == "1" for r in nd), len(nd)),
            "giusta_sbagliata": pct(tr[("1", "0")], len(nd)),
            "sbagliata_giusta": pct(tr[("0", "1")], len(nd)),
            "invariata": pct(tr[("1", "1")] + tr[("0", "0")], len(nd)),
        }
    return order, groups, out, (fmt_k, cor_k)


def plot_identity(order, S, bench, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(order))
    same = [S[g]["ident_pct"] for g in order]
    diff = [S[g]["diff_pct"] for g in order]
    b1 = ax.bar(x, same, color=C_SAME, label="Identiche alla baseline")
    b2 = ax.bar(x, diff, bottom=same, color=C_DIFF, label="Non identiche")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_y() + b.get_height() / 2, f"{b.get_height():.1f}%",
                    ha="center", va="center", color="white", fontsize=10)
    ax.set_xticks(x, [f"{g}\n(n={S[g]['n']:,})".replace(",", ".") for g in order])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% delle risposte sotto stressor")
    ax.set_title(f"Risposte identiche alla baseline sullo stesso prompt ({bench.upper()})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    fig.savefig(out_dir / "1_identiche.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bert(order, S, bench, out_dir):
    labs = [lab for _, _, lab in BINS[1:]]
    cmap = plt.get_cmap("RdYlGn_r")
    colors = [cmap(i / (len(labs) - 1)) for i in range(len(labs))]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    for lab, col in zip(labs, colors):
        vals = np.array([S[g]["bert"][lab] for g in order])
        ax.barh(y, vals, left=left, color=col, label=lab, edgecolor="white", linewidth=0.5)
        for yi, v, l in zip(y, vals, left):
            if v >= 4:
                ax.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=9,
                        color="white" if lab in (labs[0], labs[-1]) else "black")
        left += vals
    ax.set_yticks(y, [f"{g}\n(n={S[g]['n_diff']:,})".replace(",", ".") for g in order])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% delle risposte non identiche")
    ax.set_title(f"BERTScore F1 rispetto alla baseline, solo risposte non identiche ({bench.upper()})")
    ax.legend(title="BERTScore", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(labs), frameon=False)
    ax.grid(axis="y", visible=False)
    fig.savefig(out_dir / "2_bertscore_non_identiche.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_fmt_cor(order, S, keys, bench, out_dir):
    fmt_k, cor_k = keys
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    x = np.arange(len(order))
    w = 0.38
    for ax, k, name in [(axes[0], "fmt", fmt_k), (axes[1], "cor", cor_k)]:
        bb = ax.bar(x - w / 2, [S[g][f"{k}_base"] for g in order], w, color="#9DB4CF", label="Baseline (stesso prompt)")
        bs = ax.bar(x + w / 2, [S[g][f"{k}_stress"] for g in order], w, color=C_DIFF, label="Sotto stressor")
        for bars in (bb, bs):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{b.get_height():.1f}%",
                        ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x, order, fontsize=9)
        ax.set_title(name)
    axes[0].set_ylabel("% delle risposte non identiche")
    top = max(max(S[g]["fmt_base"], S[g]["fmt_stress"], S[g]["cor_base"], S[g]["cor_stress"]) for g in order)
    axes[0].set_ylim(0, max(10, top * 1.25))
    axes[1].legend(loc="upper right", frameon=True)
    fig.suptitle(f"Formato e correttezza delle risposte non identiche ({bench.upper()})")
    fig.savefig(out_dir / "3_formato_correttezza.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_transitions(order, S, keys, bench, out_dir):
    parts = [("invariata", "Correttezza invariata", "#B8C2CC"),
             ("giusta_sbagliata", "Giusta in baseline → sbagliata", "#C8463D"),
             ("sbagliata_giusta", "Sbagliata in baseline → giusta", "#3C8D5A")]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    for k, lab, col in parts:
        vals = np.array([S[g][k] for g in order])
        ax.barh(y, vals, left=left, color=col, label=lab, edgecolor="white", linewidth=0.5)
        for yi, v, l in zip(y, vals, left):
            if v >= 2.5:
                ax.text(l + v / 2, yi, f"{v:.1f}%", ha="center", va="center", fontsize=9,
                        color="black" if k == "invariata" else "white")
        left += vals
    ax.set_yticks(y, order)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% delle risposte non identiche")
    ax.set_title(f"Effetto sulla correttezza ({keys[1]}) delle risposte non identiche ({bench.upper()})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    ax.grid(axis="y", visible=False)
    fig.savefig(out_dir / "4_transizioni.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_family_n(order, groups, bench, out_dir):
    cfgs = [g for g in order if g != "Totale"]
    fams = sorted({r["stressor_class"] for g in cfgs for r in groups[g]})
    fig, axes = plt.subplots(1, len(cfgs), figsize=(4.2 * len(cfgs), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    im = None
    for ax, g in zip(axes, cfgs):
        acc = defaultdict(lambda: [0, 0])
        for r in groups[g]:
            a = acc[(r["stressor_class"], r["N_workers"])]
            a[0] += r["identical_to_base"] == "1"
            a[1] += 1
        M = np.array([[pct(*acc[(f, n)]) if acc[(f, n)][1] else np.nan for n in N_ORDER] for f in fams])
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for i in range(len(fams)):
            for j in range(len(N_ORDER)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=9,
                            color="white" if M[i, j] > 60 else "black")
        ax.set_xticks(range(len(N_ORDER)), N_ORDER)
        ax.set_yticks(range(len(fams)), fams)
        ax.set_title(g)
        ax.grid(False)
    fig.colorbar(im, ax=axes.tolist(), label="% identiche alla baseline", shrink=0.9)
    fig.suptitle(f"Risposte identiche alla baseline per famiglia di stressor e N ({bench.upper()})")
    fig.savefig(out_dir / "5_identiche_famiglia_N.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(order, S, keys, out_dir):
    fmt_k, cor_k = keys
    with open(out_dir / "quality_identity_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["configurazione", "n_risposte", "identiche_pct", "non_identiche_pct", "n_non_identiche"]
                   + [f"bert_{lab}_pct" for _, _, lab in BINS[1:]]
                   + [f"{fmt_k}_stress_pct", f"{fmt_k}_base_pct", f"{cor_k}_stress_pct", f"{cor_k}_base_pct",
                      "giusta_sbagliata_pct", "sbagliata_giusta_pct", "invariata_pct"])
        for g in order:
            s = S[g]
            w.writerow([g, s["n"], round(s["ident_pct"], 2), round(s["diff_pct"], 2), s["n_diff"]]
                       + [round(s["bert"][lab], 2) for _, _, lab in BINS[1:]]
                       + [round(s[k], 2) for k in ("fmt_stress", "fmt_base", "cor_stress", "cor_base",
                                                    "giusta_sbagliata", "sbagliata_giusta", "invariata")])


def main():
    ap = argparse.ArgumentParser(description="Grafici delle metriche qualitative (risposte identiche, BERTScore, correttezza).")
    ap.add_argument("--tasks", type=Path, default=PROJECT_ROOT / "figures" / "quality_reports" / "mmlu" / "quality_tasks.csv",
                    help="quality_tasks.csv prodotto da evaluate_quality.py")
    ap.add_argument("--out", type=Path, default=None, help="Cartella di output (default: <cartella di --tasks>/plots)")
    args = ap.parse_args()

    out_dir = args.out or args.tasks.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    bench, stress = load(args.tasks)
    if not stress:
        raise SystemExit("[ERRORE] Nessuna risposta sotto stressor con confronto baseline nel file")
    order, groups, S, keys = summarize(bench, stress)

    plot_identity(order, S, bench, out_dir)
    plot_bert(order, S, bench, out_dir)
    plot_fmt_cor(order, S, keys, bench, out_dir)
    plot_transitions(order, S, keys, bench, out_dir)
    plot_family_n(order, groups, bench, out_dir)
    write_summary(order, S, keys, out_dir)

    print(f"[OK] Grafici e riepilogo in {out_dir}")
    for g in order:
        s = S[g]
        print(f"  {g:<12} n={s['n']:>6}  identiche {s['ident_pct']:5.1f}%  non identiche {s['diff_pct']:5.1f}%  "
              f"{keys[1]} stress/base {s['cor_stress']:.1f}%/{s['cor_base']:.1f}%")


if __name__ == "__main__":
    main()
