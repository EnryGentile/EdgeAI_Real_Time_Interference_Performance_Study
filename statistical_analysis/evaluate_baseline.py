#!/usr/bin/env python3
"""
evaluate_baseline.py - Accuratezza della baseline rispetto alla ground truth.

Legge i file responses_baseline_*.json (AGX e Nano) e valuta ogni risposta contro la risposta attesa.

MMLU
  - pred_explicit   lettera data esplicitamente come risposta: la risposta inizia con la lettera
                    ("B", "B.", "B)", "(B)", "D\\n\\nExplanation...") oppure contiene "answer is B" / "Answer: B".
                    Le risposte che ragionano senza arrivare a una lettera restano senza risposta.
  - answered        1 se è stata trovata una lettera esplicita
  - correct         1 se la lettera esplicita coincide con la ground truth  (metrica principale)
  - correct_first_letter  regola precedente (prima A/B/C/D isolata nel testo), solo per confronto:
                    conta anche l'articolo inglese "A" e le opzioni ripetute dal prompt
  - format_ok       1 se la risposta è solo la lettera
Spider
  - sql_valid, exec_match, exact_match  come in evaluate_quality.py

Le ripetizioni della baseline usano seed fisso e temperatura 0: lo script verifica che diano la stessa
risposta e calcola gli intervalli di confidenza al 95% con un bootstrap sui prompt (non sulle ripetizioni,
che non sono campioni indipendenti).

Output (in --out):
  baseline_tasks.csv              una riga per task
  baseline_summary.csv            una riga per piattaforma / modello / device / benchmark
  1_accuratezza_mmlu.png          accuratezza con IC 95%, livello del caso (25%) e regola "prima lettera"
  2_esito_risposte_mmlu.png       corretta / sbagliata / nessuna lettera esplicita
  3_domande_mmlu.png              esito domanda per domanda
  4_spider.png                    metriche Spider (se presenti)

Esempio:
  python statistical_analysis/evaluate_baseline.py
  python statistical_analysis/evaluate_baseline.py --results resultAGX/mmlu/baseline --out figures/quality_reports/baseline_agx
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_quality import (mmlu_extract, mmlu_first_letter, mmlu_format_ok, normalize_sql,  # noqa: E402
                              results_match, run_query, spider_extract_sql)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
RE_FILE = re.compile(r"^responses_baseline_([a-z0-9\-]+)_(cpu|gpu)_(mmlu|spider)_rep(\d+)$", re.I)
B_BOOT = 4000
C_OK, C_BAD, C_NONE = "#3C8D5A", "#C8463D", "#B8C2CC"

def platform_of(path: Path) -> str:
    return "Orin Nano" if "nano" in str(path).lower() else "AGX Orin"


def boot_ci(per_prompt: np.ndarray, rng) -> tuple:
    """IC 95% bootstrap della media, ricampionando i prompt."""
    if len(per_prompt) == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(per_prompt), size=(B_BOOT, len(per_prompt)))
    m = per_prompt[idx].mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def evaluate(files, db_dir: Path):
    rows = []
    gold_cache = {}
    for path in files:
        m = RE_FILE.match(path.stem)
        if not m:
            continue
        model, device, bench, rep = m.group(1).upper(), m.group(2).upper(), m.group(3).lower(), int(m.group(4))
        for t in json.loads(path.read_text(encoding="utf-8")):
            if t.get("status", "ok") != "ok":
                continue
            resp, gt = t.get("model_response", "") or "", (t.get("ground_truth", "") or "").strip()
            row = {"platform": platform_of(path), "model_id": model, "device": device, "benchmark": bench,
                   "repetition": rep, "sample_id": int(t.get("sample_id") or t.get("task_index") or 0),
                   "ground_truth": gt, "model_response": resp}
            if bench == "mmlu":
                pred = mmlu_extract(resp)
                first = mmlu_first_letter(resp)
                row.update({"pred_explicit": pred, "answered": int(bool(pred)),
                            "correct": int(pred == gt.upper()),
                            "pred_first_letter": first, "correct_first_letter": int(first == gt.upper()),
                            "format_ok": int(mmlu_format_ok(resp))})
            else:
                prompt = t.get("prompt", "")
                dm = re.search(r"database (?:SQLite )?'([^']+)'", prompt, re.I)
                db_id = dm.group(1) if dm else ""
                pred_sql = spider_extract_sql(resp)
                row.update({"db_id": db_id, "pred_sql": pred_sql,
                            "exact_match": int(bool(pred_sql) and normalize_sql(pred_sql) == normalize_sql(gt))})
                db_file = db_dir / db_id / f"{db_id}.sqlite"
                if db_id and db_file.exists():
                    if (db_id, gt) not in gold_cache:
                        gold_cache[(db_id, gt)] = run_query(db_file, gt)
                    gold_ok, gold_rows = gold_cache[(db_id, gt)]
                    ok, pred_rows = run_query(db_file, pred_sql)
                    row["sql_valid"] = int(ok)
                    row["exec_match"] = int(ok and gold_ok and results_match(pred_rows, gold_rows,
                                                                            ordered="order by" in gt.lower()))
            rows.append(row)
    return rows


def summarize(rows):
    rng = np.random.default_rng(42)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["benchmark"], r["platform"], r["model_id"], r["device"])].append(r)
    metrics = {"mmlu": ["correct", "answered", "correct_first_letter", "format_ok"],
               "spider": ["sql_valid", "exec_match", "exact_match"]}
    summary = []
    for key in sorted(groups, key=lambda k: (k[0], k[1] != "AGX Orin", k[2], k[3])):
        bench, plat, model, dev = key
        rs = groups[key]
        by_prompt = defaultdict(list)
        for r in rs:
            by_prompt[r["sample_id"]].append(r)
        stable = sum(len({x["model_response"].strip() for x in v}) == 1 for v in by_prompt.values())
        s = {"benchmark": bench, "platform": plat, "model_id": model, "device": dev,
             "n_prompts": len(by_prompt), "n_reps": len({r["repetition"] for r in rs}), "n_tasks": len(rs),
             "prompts_stable_pct": round(100 * stable / len(by_prompt), 1)}
        for mt in metrics[bench]:
            vals = [[x[mt] for x in v if mt in x] for v in by_prompt.values()]
            pp = np.array([np.mean(v) for v in vals if v])
            if not len(pp):
                continue
            lo, hi = boot_ci(pp, rng)
            s[f"{mt}_pct"] = round(100 * pp.mean(), 1)
            s[f"{mt}_ci_lo"], s[f"{mt}_ci_hi"] = round(100 * lo, 1), round(100 * hi, 1)
        if bench == "mmlu":
            ans = [r for r in rs if r["answered"]]
            s["correct_given_answered_pct"] = round(100 * np.mean([r["correct"] for r in ans]), 1) if ans else ""
        summary.append(s)
    return summary


def label(s):
    return f"{s['platform']}\n{s['model_id']} {s['device']}"


def plot_accuracy(S, out_dir):
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(S) + 2), 4.6))
    x = np.arange(len(S))
    acc = np.array([s["correct_pct"] for s in S])
    err = np.array([[s["correct_pct"] - s["correct_ci_lo"], s["correct_ci_hi"] - s["correct_pct"]] for s in S]).T
    bars = ax.bar(x, acc, color="#2E6DB4", yerr=err, capsize=4, ecolor="#4A5868", label="Accuratezza (lettera esplicita)")
    ax.scatter(x, [s["correct_first_letter_pct"] for s in S], marker="D", color="#E07B39", zorder=3, s=36,
               label="Regola precedente 'prima lettera'")
    ax.axhline(25, color="#888", ls="--", lw=1)
    ax.text(len(S) - 0.5, 25.8, "caso (25%)", ha="right", va="bottom", fontsize=9, color="#666")
    for b, s in zip(bars, S):
        ax.text(b.get_x() + b.get_width() / 2 + 0.06, s["correct_ci_hi"] + 1.5, f"{s['correct_pct']:.0f}%",
                ha="left", va="bottom", fontsize=9)
    ax.set_xticks(x, [label(s) + f"\n({s['n_prompts']} domande)" for s in S], fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% delle domande")
    ax.set_title("Accuratezza della baseline su MMLU rispetto alla ground truth (IC 95%)")
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    fig.savefig(out_dir / "1_accuratezza_mmlu.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_outcome(S, out_dir):
    fig, ax = plt.subplots(figsize=(9, 0.75 * len(S) + 1.6))
    y = np.arange(len(S))[::-1]
    ok = np.array([s["correct_pct"] for s in S])
    ans = np.array([s["answered_pct"] for s in S])
    parts = [(ok, "Lettera corretta", C_OK, "white"), (ans - ok, "Lettera sbagliata", C_BAD, "white"),
             (100 - ans, "Nessuna lettera esplicita", C_NONE, "black")]
    left = np.zeros(len(S))
    for vals, lab, col, tc in parts:
        ax.barh(y, vals, left=left, color=col, label=lab, edgecolor="white", linewidth=0.5)
        for yi, v, l in zip(y, vals, left):
            if v >= 4:
                ax.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=9, color=tc)
        left += vals
    ax.set_yticks(y, [label(s).replace("\n", " · ") for s in S], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% delle domande")
    ax.set_title("Esito delle risposte della baseline su MMLU", pad=28)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    ax.grid(axis="y", visible=False)
    fig.savefig(out_dir / "2_esito_risposte_mmlu.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_questions(rows, S, out_dir):
    keys = [(s["platform"], s["model_id"], s["device"]) for s in S]
    samples = sorted({r["sample_id"] for r in rows})
    M = np.full((len(keys), len(samples)), np.nan)
    gt = {}
    for i, k in enumerate(keys):
        by = defaultdict(list)
        for r in rows:
            if (r["platform"], r["model_id"], r["device"]) == k:
                by[r["sample_id"]].append(2 if r["correct"] else (1 if r["answered"] else 0))
                gt[r["sample_id"]] = r["ground_truth"]
        for j, sid in enumerate(samples):
            if by[sid]:
                M[i, j] = max(set(by[sid]), key=by[sid].count)   # esito più frequente fra le ripetizioni
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * len(samples) + 3), 0.55 * len(keys) + 1.8))
    ax.imshow(M, cmap=ListedColormap([C_NONE, C_BAD, C_OK]), vmin=-0.5, vmax=2.5, aspect="auto")
    ax.set_yticks(range(len(keys)), [f"{p} · {m} {d}" for p, m, d in keys], fontsize=9)
    ax.set_xticks(range(len(samples)), [f"{sid}\n{gt.get(sid, '')}" for sid in samples], fontsize=7.5)
    ax.set_xlabel("Domanda (sample_id) e risposta attesa")
    ax.set_xticks(np.arange(-0.5, len(samples)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(keys)), minor=True)
    ax.grid(which="minor", color="white", lw=1)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_OK, C_BAD, C_NONE)]
    ax.legend(handles, ["Corretta", "Sbagliata", "Nessuna lettera esplicita"], loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    ax.set_title("Esito della baseline su MMLU, domanda per domanda", pad=28)
    fig.savefig(out_dir / "3_domande_mmlu.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_spider(S, out_dir):
    mts = [m for m in ("sql_valid", "exec_match", "exact_match") if f"{m}_pct" in S[0]]
    if not mts:
        return
    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(S) + 2), 4.2))
    x = np.arange(len(S))
    w = 0.8 / len(mts)
    for i, (m, col) in enumerate(zip(mts, ["#9DB4CF", "#2E6DB4", "#1B3F6B"])):
        v = np.array([s[f"{m}_pct"] for s in S])
        e = np.array([[s[f"{m}_pct"] - s[f"{m}_ci_lo"], s[f"{m}_ci_hi"] - s[f"{m}_pct"]] for s in S]).T
        bs = ax.bar(x + (i - (len(mts) - 1) / 2) * w, v, w, yerr=e, capsize=3, color=col, ecolor="#4A5868", label=m)
        for b in bs:
            ax.text(b.get_x() + b.get_width() / 2, 1.5, f"{b.get_height():.0f}%", ha="center", va="bottom",
                    color="white" if col != "#9DB4CF" else "black", fontsize=8.5)
    ax.set_xticks(x, [label(s) + f"\n({s['n_prompts']} domande)" for s in S], fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% delle domande")
    ax.set_title("Baseline su Spider rispetto alla query attesa (IC 95%)")
    ax.legend(frameon=True, fontsize=9)
    fig.savefig(out_dir / "4_spider.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Accuratezza della baseline rispetto alla ground truth (MMLU, Spider).")
    ap.add_argument("--results", type=Path, nargs="+",
                    default=[PROJECT_ROOT / "resultAGX", PROJECT_ROOT / "resultNano"],
                    help="Cartelle in cui cercare (ricorsivamente) responses_baseline_*.json")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "figures" / "quality_reports" / "baseline")
    ap.add_argument("--db-dir", type=Path, default=PROJECT_ROOT / "datasets" / "spider_db",
                    help="Cartella con i database Spider (<db_id>/<db_id>.sqlite)")
    args = ap.parse_args()

    files = sorted({p for d in args.results for p in d.rglob("responses_baseline_*.json")
                    if "fallback_partial" not in p.name})
    if not files:
        raise SystemExit("[ERRORE] Nessun file responses_baseline_*.json trovato")
    rows = evaluate(files, args.db_dir)
    summary = summarize(rows)
    args.out.mkdir(parents=True, exist_ok=True)

    cols = ["platform", "model_id", "device", "benchmark", "repetition", "sample_id", "ground_truth",
            "pred_explicit", "answered", "correct", "pred_first_letter", "correct_first_letter", "format_ok",
            "db_id", "pred_sql", "sql_valid", "exec_match", "exact_match", "model_response"]
    with open(args.out / "baseline_tasks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    fields = list(dict.fromkeys(k for s in summary for k in s))
    with open(args.out / "baseline_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)

    mm = [s for s in summary if s["benchmark"] == "mmlu"]
    sp = [s for s in summary if s["benchmark"] == "spider"]
    if mm:
        plot_accuracy(mm, args.out)
        plot_outcome(mm, args.out)
        plot_questions([r for r in rows if r["benchmark"] == "mmlu"], mm, args.out)
    if sp:
        plot_spider(sp, args.out)

    print(f"[OK] {len(rows)} task da {len(files)} file -> {args.out}")
    for s in summary:
        if s["benchmark"] == "mmlu":
            print(f"  MMLU   {s['platform']:<9} {s['model_id']:<6} {s['device']}  reps={s['n_reps']:>2}  "
                  f"stabili {s['prompts_stable_pct']:.0f}%  accuratezza {s['correct_pct']:.1f}% "
                  f"[{s['correct_ci_lo']:.1f}-{s['correct_ci_hi']:.1f}]  risposte con lettera {s['answered_pct']:.0f}%  "
                  f"prima lettera {s['correct_first_letter_pct']:.1f}%")
        else:
            print(f"  Spider {s['platform']:<9} {s['model_id']:<6} {s['device']}  reps={s['n_reps']:>2}  "
                  + "  ".join(f"{m} {s[m + '_pct']:.0f}%" for m in ("sql_valid", "exec_match", "exact_match")
                              if m + "_pct" in s))


if __name__ == "__main__":
    main()
