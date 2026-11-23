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
AGX_DIR = PROJECT_ROOT / "resultAGX"
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


def _default_search_roots(custom_dir: str | None = None) -> list[Path]:
    """Restituisce le cartelle di ricerca ordinate per priorita (custom -> resultAGX -> results)."""
    if custom_dir:
        return [Path(custom_dir)]
    roots = []
    if AGX_DIR.exists():
        roots.append(AGX_DIR)
    if RESULTS_DIR.exists():
        roots.append(RESULTS_DIR)
    return roots or [PROJECT_ROOT]


def load_baseline_mean(
    model_tag: str,
    device_tag: str,
    bench_tag: str,
    search_roots: list[Path] | None = None,
    max_reps: int | None = 20,
) -> dict:
    """Carica le medie della baseline dal file doe_*_none.csv corrispondente, limitato alle prime max_reps ripetizioni."""
    roots = search_roots or _default_search_roots()
    target_name = f"doe_{model_tag}_{device_tag}_{bench_tag}_none.csv"
    matches: list[Path] = []
    for root in roots:
        matches.extend(root.rglob(target_name))
    if not matches:
        for root in roots:
            matches.extend(root.rglob("doe_*_none.csv"))

    baseline_means = {}
    if matches:
        try:
            df_base = pd.read_csv(matches[0])
            if max_reps is not None and max_reps > 0:
                df_base = df_base.head(max_reps)
            for m_col, _, _ in METRICS:
                if m_col in df_base.columns:
                    baseline_means[m_col] = df_base[m_col].mean()
        except Exception:
            pass
    return baseline_means


def plot_doe_file(
    csv_path: Path,
    output_subdir: Path,
    target_duration: str = "t_tot",
    search_roots: list[Path] | None = None,
    max_reps: int | None = 20,
):
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

    if max_reps is not None and max_reps > 0 and not df.empty:
        grp_cols = [c for c in ["stressor_class", "num_workers", "size"] if c in df.columns]
        if grp_cols:
            df = df.groupby(grp_cols, dropna=False, sort=False).head(max_reps).copy()
        else:
            df = df.head(max_reps).copy()
        print(f" -> Limite ripetizioni applicato: max {max_reps} rep per cella ({len(df)} record)")

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
    stressor_tag = "_".join(parts[3:]) if len(parts) > 3 else "stressor"

    baseline_means = load_baseline_mean(model_tag, device_tag, bench_tag, search_roots=search_roots, max_reps=max_reps)

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
    parser.add_argument("--data-dir", type=str, default=None, help="Cartella radice in cui cercare ricorsivamente i file doe_*.csv (default: resultAGX o results).")
    parser.add_argument("--model", type=str, default=None, help="Filtra per modello (es. slm-1b, llm-8b).")
    parser.add_argument("--device", type=str, default=None, help="Filtra per device (es. cpu, gpu).")
    parser.add_argument("--benchmark", type=str, default=None, help="Filtra per benchmark (es. mmlu, spider).")
    parser.add_argument("--duration", type=str, default="t_tot", help="Durata da filtrare (default: t_tot).")
    parser.add_argument("--max-reps", type=int, default=20, help="Numero massimo di ripetizioni (1..N) da considerare per cella e baseline (default: 20; 0=tutte).")
    args = parser.parse_args()

    search_roots = _default_search_roots(args.data_dir)
    target_files: list[Path] = []
    if args.csv:
        target_files.append(Path(args.csv))
    else:
        seen = set()
        for root in search_roots:
            for p in sorted(root.rglob("doe_*.csv")):
                name_low = p.name.lower()
                if "_none.csv" in name_low:
                    continue
                if args.model and args.model.lower() not in name_low:
                    continue
                if args.device and f"_{args.device.lower()}_" not in name_low:
                    continue
                if args.benchmark and f"_{args.benchmark.lower()}_" not in name_low:
                    continue
                if p.resolve() not in seen:
                    seen.add(p.resolve())
                    target_files.append(p)
        if not target_files:
            example_csv = PROJECT_ROOT / "CSV_MOCK" / "doe_example.csv"
            if example_csv.exists():
                target_files.append(example_csv)

    if not target_files:
        print(" [ERRORE] Nessun file DoE CSV trovato in resultAGX/ ne in results/.")
        sys.exit(1)

    for csv_file in target_files:
        subdir = FIGURES_DIR / csv_file.stem.replace("doe_", "")
        plot_doe_file(csv_file, subdir, target_duration=args.duration, search_roots=search_roots, max_reps=args.max_reps)

    print(f"\n[COMPLETATO] Tutti i grafici sono stati generati in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
