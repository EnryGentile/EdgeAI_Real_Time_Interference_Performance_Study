#!/usr/bin/env python3
"""
plot_llm_metrics.py - Visualizzazione della degradazione delle metriche LLM (Traccia A1).

Fattori sperimentali DoE:
  - 1. Carico worker (num_workers: 25% N_tot, 50% N_tot, 75% N_tot, N_tot)
  - 2. Working-set size (size: 25% S_tot, 50% S_tot, 75% S_tot) - presente SOLO per mem e io.
  - La durata e fissa e costante al tempo totale (duration: t_tot).
"""

import os
import sys
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Percorsi principali
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures" / "llm_metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Configurazione stile grafico
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

N_ORDER = ["25% N_tot", "50% N_tot", "75% N_tot", "N_tot"]
S_ORDER = ["25% S_tot", "50% S_tot", "75% S_tot"]

METRICS = [
    ("ttft_ms", "Time To First Token (ms)", "Tempo al primo token"),
    ("per_token_lat_ms", "Per-Token Latency (ms)", "Latenza di decodifica per-token"),
    ("throughput_tok_s", "Throughput (tokens/s)", "Throughput di generazione"),
    ("jitter_ms", "Inter-Token Jitter (ms)", "Jitter inter-token")
]


def load_baseline_mean(model_tag: str, device_tag: str, bench_tag: str) -> dict:
    """Carica le medie della baseline dal file doe_*_none.csv corrispondente, se disponibile."""
    pattern = str(RESULTS_DIR / f"doe_{model_tag}_{device_tag}_{bench_tag}_none.csv")
    matches = glob.glob(pattern)
    if not matches:
        pattern = str(RESULTS_DIR / "doe_*_none.csv")
        matches = glob.glob(pattern)

    baseline_means = {}
    if matches:
        try:
            df_base = pd.read_csv(matches[0])
            for m_col, _, _ in METRICS:
                if m_col in df_base.columns:
                    baseline_means[m_col] = df_base[m_col].mean()
        except Exception:
            pass
    return baseline_means


def plot_doe_file(csv_path: Path, output_subdir: Path, target_duration: str = "t_tot"):
    """Genera i grafici di degradazione per un singolo file DoE."""
    if not csv_path.exists():
        print(f" [AVVISO] File non trovato: {csv_path}")
        return

    print(f"\n[ANALISI DoE] Elaborazione file: {csv_path.name}")
    df = pd.read_csv(csv_path)

    # 1. Filtro durata: la durata e fissa a t_tot per l'intero DoE
    if "duration" in df.columns:
        dur_vals = df["duration"].dropna().unique().tolist()
        if target_duration in dur_vals:
            df = df[df["duration"] == target_duration].copy()
            print(f" -> Filtro applicato: selezionati {len(df)} record con duration='{target_duration}'")
        else:
            print(f" -> duration='{target_duration}' non trovato, uso i record disponibili ({dur_vals})")

    if df.empty:
        print(f" [AVVISO] Nessun record trovato in {csv_path.name} con duration='{target_duration}'.")
        return

    # 2. Rilevamento dei fattori sperimentali
    # Fattore 1: num_workers (sempre presente)
    df["num_workers"] = pd.Categorical(df["num_workers"], categories=N_ORDER, ordered=True)

    # Fattore 2: size (presente solo per mem e io)
    has_size = False
    if "size" in df.columns:
        valid_sizes = [s for s in df["size"].dropna().unique() if str(s).upper() not in ("NAN", "N/A", "NONE", "")]
        if len(valid_sizes) > 1:
            has_size = True
            df["size"] = pd.Categorical(df["size"], categories=S_ORDER, ordered=True)

    # Estrazione identificativi dal nome file (es. doe_slm-1b_gpu_mmlu_cpu.csv)
    fname_stem = csv_path.stem.replace("doe_", "")
    parts = fname_stem.split("_")
    model_tag = parts[0] if len(parts) > 0 else "slm"
    device_tag = parts[1] if len(parts) > 1 else "gpu"
    bench_tag = parts[2] if len(parts) > 2 else "mmlu"
    stressor_tag = parts[3] if len(parts) > 3 else "stressor"

    baseline_means = load_baseline_mean(model_tag, device_tag, bench_tag)

    output_subdir.mkdir(parents=True, exist_ok=True)

    # 3. Generazione grafici per ciascuna metrica
    for metric_col, metric_label, metric_desc in METRICS:
        if metric_col not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(7.5, 5))

        if has_size:
            # DoE 3D: num_workers (X) e size (Hue)
            palette = sns.color_palette("viridis", n_colors=len(S_ORDER))
            sns.pointplot(
                data=df,
                x="num_workers",
                y=metric_col,
                hue="size",
                dodge=0.2,
                capsize=0.1,
                err_kws={'linewidth': 1.5},
                markers=["o", "s", "^"],
                linestyles=["-", "--", "-."],
                palette=palette,
                ax=ax
            )
            factor_info = "Fattori: N (Workers) e S (Working-set Size)"
        else:
            # DoE 2D: solo num_workers (X)
            sns.pointplot(
                data=df,
                x="num_workers",
                y=metric_col,
                capsize=0.1,
                err_kws={'linewidth': 1.5},
                color="#1f77b4",
                marker="o",
                linestyle="-",
                ax=ax,
                label=f"Stressor {stressor_tag.upper()} (t_tot)"
            )
            factor_info = "Fattore: N (Carico Workers)"

        # Linea di riferimento Baseline (se disponibile)
        if metric_col in baseline_means and pd.notna(baseline_means[metric_col]):
            b_val = baseline_means[metric_col]
            ax.axhline(b_val, color="green", linestyle=":", linewidth=2, label=f"Baseline (senza stress: {b_val:.2f})")

        ax.set_title(
            f"Degradazione {metric_label}\nStressor: {stressor_tag.upper()} | {factor_info} | Durata: t_tot",
            fontsize=12,
            fontweight="bold"
        )
        ax.set_xlabel("Livelli di Carico Worker (N)", fontsize=11, fontweight="bold")
        ax.set_ylabel(metric_label, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(title="Legenda", loc="best", frameon=True)

        fig.tight_layout()
        out_file = output_subdir / f"degradation_{fname_stem}_{metric_col}.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"   [OK] Grafico salvato: {out_file.name}")


def main():
    parser = argparse.ArgumentParser(description="Visualizzazione della degradazione delle metriche LLM (DoE).")
    parser.add_argument("--csv", type=str, default=None, help="Percorso di un file DoE CSV specifico.")
    parser.add_argument("--duration", type=str, default="t_tot", help="Durata da filtrare (default: t_tot).")
    args = parser.parse_args()

    target_files = []
    if args.csv:
        target_files.append(Path(args.csv))
    else:
        # Cerca tutti i file doe_*.csv in results/ (escluso doe_*_none.csv che e la baseline)
        found = [Path(p) for p in glob.glob(str(RESULTS_DIR / "doe_*.csv")) if "_none.csv" not in p]
        if found:
            target_files.extend(found)
        else:
            # Fallback su doe_example.csv se presente
            example_csv = PROJECT_ROOT / "doe_example.csv"
            if example_csv.exists():
                target_files.append(example_csv)

    if not target_files:
        print(" [ERRORE] Nessun file DoE CSV trovato in results/ ne nella root del progetto.")
        sys.exit(1)

    for csv_file in target_files:
        subdir = FIGURES_DIR / csv_file.stem.replace("doe_", "")
        plot_doe_file(csv_file, subdir, target_duration=args.duration)

    print(f"\n[COMPLETATO] Tutti i grafici sono stati generati in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
