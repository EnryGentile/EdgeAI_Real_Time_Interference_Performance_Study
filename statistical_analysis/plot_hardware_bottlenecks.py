#!/usr/bin/env python3
"""
plot_hardware_bottlenecks.py - Visualizzazione dell'impatto e dei colli di bottiglia hardware (Traccia A1).

Analizza le serie temporali (vmstat + sensori termici) e il riepilogo generale (master_experiment_summary.csv).
Fattori sperimentali DoE:
  - 1. Carico worker (n_workers: N25, N50, N75, N100)
  - 2. Working-set size (size: S25, S50, S75) - presente SOLO per mem e io.
  - La durata e fissa al tempo totale (duration: t_tot).
"""

import os
import re
import sys
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Percorsi principali
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TIMESERIES_DIR = PROJECT_ROOT / "data_timeseries"
AGX_DIR = PROJECT_ROOT / "resultAGX"
FIGURES_DIR = PROJECT_ROOT / "figures" / "hardware_bottlenecks"
MASTER_CSV = AGX_DIR / "master_experiment_summary.csv" if (AGX_DIR / "master_experiment_summary.csv").exists() else (PROJECT_ROOT / "master_experiment_summary.csv")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

STRESSOR_WITH_SIZE = ["mem", "io"]
N_ORDER = ["N25", "N50", "N75", "N100"]
S_ORDER = ["S25", "S50", "S75"]

# Famiglie di metriche monitorate (vmstat + termico)
METRIC_GROUPS = [
    ("cpu", "Parametri CPU", [("us", "us (User %)"), ("sy", "sy (Kernel %)"), ("id", "id (Idle %)"), ("wa", "wa (I/O Wait %)")]),
    ("mem", "Parametri Memoria", [("free_kb", "Free (KB)"), ("buff_kb", "Buffers (KB)"), ("cache_kb", "Cache (KB)"), ("swpd_kb", "Swap (KB)")]),
    ("io", "Parametri I/O", [("bi", "bi (Blocchi In)"), ("bo", "bo (Blocchi Out)")]),
    ("procs", "Parametri Processi", [("r", "r (Ready)"), ("b", "b (Blocked)")]),
    ("system", "Parametri Sistema", [("cs", "cs (Context Switch/s)"), ("in", "in (Interrupts/s)")]),
    ("thermal", "Parametri Termici", [("cpu_temp_c", "Temp CPU (°C)"), ("gpu_temp_c", "Temp GPU (°C)")])
]


def parse_timeseries_data(
    target_duration: str = "t_tot",
    search_dir: Path | None = None,
    model_filter: str | None = None,
    device_filter: str | None = None,
    bench_filter: str | None = None,
    max_reps: int | None = 20,
) -> pd.DataFrame:
    """Carica e aggrega le time-series vmstat cercando ricorsivamente in resultAGX/ o nella cartella indicata."""
    ts_files: list[str] = []
    if search_dir is not None:
        ts_files = [str(p) for p in search_dir.rglob("*_timeseries.csv")]
    else:
        for cand in (AGX_DIR, TIMESERIES_DIR, PROJECT_ROOT / "timeseries"):
            if cand.exists():
                ts_files.extend(str(p) for p in cand.rglob("*_timeseries.csv"))

    # Esclude file di fallback parziali
    ts_files = [p for p in ts_files if not os.path.basename(p).lower().startswith("fallback_partial_")]
    if not ts_files:
        print(" [AVVISO] Nessun file time-series trovato.")
        return pd.DataFrame()

    records = []
    # Pattern completo: include modello, device e benchmark
    # es. run_slm-1b_gpu_mmlu_cpu_N25_t_tot_rep1_timeseries.csv o run_slm-1b_gpu_mmlu_mem_N25_t_tot_S25_rep1_timeseries.csv
    re_stress_full = re.compile(
        r"run_([a-zA-Z0-9\-]+)_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_([a-zA-Z0-9_\-]+?)_(N\d+)_(t_[a-z]+)(?:_(S\d+))?_rep(\d+)_timeseries\.csv",
        re.IGNORECASE
    )
    # Pattern legacy: senza modello, device e benchmark
    re_stress_legacy = re.compile(
        r"run_([a-zA-Z0-9_\-]+?)_(N\d+)_(t_[a-z]+)(?:_(S\d+))?_rep(\d+)_timeseries\.csv",
        re.IGNORECASE
    )
    re_base = re.compile(
        r"baseline_([a-zA-Z0-9\-]+)_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_rep(\d+)_timeseries\.csv",
        re.IGNORECASE
    )

    for filepath in ts_files:
        fname = os.path.basename(filepath)
        m_str_full = re_stress_full.match(fname)
        m_str_leg = re_stress_legacy.match(fname) if not m_str_full else None
        m_bas = re_base.match(fname)

        if m_str_full:
            model_val, dev_val, bench_val, stressor, n_w, dur, sz, rep = m_str_full.groups()
            sz = sz if sz else "N/A"
        elif m_str_leg:
            stressor, n_w, dur, sz, rep = m_str_leg.groups()
            sz = sz if sz else "N/A"
            model_val, dev_val, bench_val = "N/A", "N/A", "N/A"
        elif m_bas:
            if target_duration != "baseline":
                continue
            model_val, dev_val, bench_val, rep = m_bas.groups()
            stressor, n_w, dur, sz = "baseline", "N0", "baseline", "N/A"
        else:
            continue

        if model_filter and model_filter.lower() not in model_val.lower():
            continue
        if device_filter and dev_val.lower() != device_filter.lower():
            continue
        if bench_filter and bench_val.lower() != bench_filter.lower():
            continue
        if max_reps is not None and max_reps > 0 and int(rep) > max_reps:
            continue

        try:
            sub_df = pd.read_csv(filepath)
            # Rinomina colonne abbreviate se necessario per allineamento
            rename_map = {
                "swpd": "swpd_kb", "free": "free_kb", "buff": "buff_kb", "cache": "cache_kb"
            }
            sub_df.rename(columns={k: v for k, v in rename_map.items() if k in sub_df.columns and v not in sub_df.columns}, inplace=True)

            mean_vals = sub_df.mean(numeric_only=True).to_dict()
            mean_vals["model"] = model_val
            mean_vals["device"] = dev_val
            mean_vals["benchmark"] = bench_val
            mean_vals["stressor"] = stressor
            mean_vals["n_workers"] = n_w
            mean_vals["duration"] = dur
            mean_vals["size"] = sz
            mean_vals["repetition"] = rep
            records.append(mean_vals)
        except Exception as e:
            print(f" [ERRORE] Lettura fallita per {fname}: {e}")

    df_hw = pd.DataFrame(records)
    return df_hw


def plot_hardware_groups(df_hw: pd.DataFrame, target_duration: str = "t_tot"):
    """Genera i grafici per ciascuna combinazione (modello, device, benchmark) e famiglia hardware."""
    if df_hw.empty:
        return

    # Filtra la durata: DoE utilizza esclusivamente t_tot per gli stressor
    available_durations = df_hw["duration"].unique()
    if target_duration in available_durations:
        df_hw = df_hw[df_hw["duration"] == target_duration].copy()
        print(f" -> Filtro hardware applicato: duration='{target_duration}' ({len(df_hw)} record)")

    df_hw["n_workers"] = pd.Categorical(df_hw["n_workers"], categories=N_ORDER, ordered=True)

    for (m_val, d_val, b_val), df_ctx in df_hw.groupby(["model", "device", "benchmark"], dropna=False):
        stressors = [s for s in df_ctx["stressor"].unique() if s != "baseline"]
        if not stressors:
            continue

        ctx_tag = f"{str(m_val).lower()}_{str(d_val).lower()}_{str(b_val).lower()}"
        ctx_out_dir = FIGURES_DIR / ctx_tag
        ctx_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[HW PLOT] Contesto: {ctx_tag} ({len(df_ctx)} file)")

        for s in stressors:
            is_3d = s.lower() in STRESSOR_WITH_SIZE
            df_s = df_ctx[df_ctx["stressor"] == s].copy()
            sub = df_s.groupby("n_workers", observed=False).mean(numeric_only=True).reindex(N_ORDER)

            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 9.5), squeeze=False)
            has_plotted_any = False

            for g_idx, (grp_key, grp_title, col_tuples) in enumerate(METRIC_GROUPS):
                ax = axes[g_idx // 3, g_idx % 3]
                grp_has_data = False
                for col_name, label in col_tuples:
                    if col_name in sub.columns and sub[col_name].notna().any():
                        ax.plot(sub.index, sub[col_name], marker='o', linewidth=1.8, label=label)
                        has_plotted_any = True
                        grp_has_data = True

                ax.set_title(grp_title, fontsize=11, fontweight="bold")
                if g_idx // 3 == 1:
                    ax.set_xlabel("Livelli Carico Worker (N)", fontsize=10, fontweight="bold")
                if g_idx % 3 == 0:
                    ax.set_ylabel("Valore Medio", fontsize=10, fontweight="bold")
                ax.grid(True, linestyle="--", alpha=0.6)
                if grp_has_data:
                    ax.legend(loc="best", frameon=True, fontsize=8.5)

            if has_plotted_any:
                mode_note = "t_tot, media su S25/S50/S75" if is_3d else "t_tot"
                fig.suptitle(
                    f"Colli di Bottiglia Hardware — {m_val.upper()} ({d_val.upper()}) | Stressor {s.upper()} ({mode_note})",
                    fontsize=13,
                    fontweight="bold",
                )
                fig.tight_layout()
                out_path = ctx_out_dir / f"{ctx_tag}_{s}_bottleneck.png"
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
                print(f"   [OK] Grafico hardware unificato salvato: {out_path.name}")
            plt.close(fig)


def plot_master_microarchitecture(
    target_duration: str = "t_tot",
    model_filter: str | None = None,
    device_filter: str | None = None,
    bench_filter: str | None = None,
):
    """Visualizza metriche microarchitetturali e termiche da master_experiment_summary.csv filtrando per contesto."""
    if not MASTER_CSV.exists():
        return

    df_master = pd.read_csv(MASTER_CSV)
    if "N_workers" in df_master.columns:
        df_master.rename(columns={"N_workers": "num_workers"}, inplace=True)
    if "duration_mode" in df_master.columns:
        df_master.rename(columns={"duration_mode": "duration"}, inplace=True)
    if "S_size" in df_master.columns:
        df_master.rename(columns={"S_size": "size"}, inplace=True)

    if model_filter and "model_id" in df_master.columns:
        df_master = df_master[df_master["model_id"].astype(str).str.lower().str.contains(model_filter.lower(), na=False)].copy()
    if device_filter and "device" in df_master.columns:
        df_master = df_master[df_master["device"].astype(str).str.lower() == device_filter.lower()].copy()
    if bench_filter and "benchmark" in df_master.columns:
        df_master = df_master[df_master["benchmark"].astype(str).str.lower() == bench_filter.lower()].copy()

    if "duration" in df_master.columns and target_duration in df_master["duration"].values:
        df_master = df_master[df_master["duration"] == target_duration].copy()

    if df_master.empty:
        return

    # Evita di generare grafici con solo temperatura se mancano i contatori microarchitetturali perf_*
    if not any(c in df_master.columns and df_master[c].notna().any() for c in ("perf_cache_misses", "perf_branch_misses")):
        return

    ctx_cols = [c for c in ("model_id", "device", "benchmark") if c in df_master.columns]
    groups = df_master.groupby(ctx_cols, dropna=False) if ctx_cols else [((), df_master)]

    for ctx_key, df_ctx in groups:
        ctx_label = "_".join(str(x).lower() for x in (ctx_key if isinstance(ctx_key, tuple) else (ctx_key,)))
        df_ctx = df_ctx.copy()
        df_ctx["num_workers"] = pd.Categorical(df_ctx["num_workers"], categories=N_ORDER, ordered=True)

        for sc in df_ctx["stressor_class"].dropna().unique():
            sc_clean = str(sc).lower()
            if sc_clean in ("none", "baseline"):
                continue

            is_3d = sc_clean in STRESSOR_WITH_SIZE
            sizes = S_ORDER if is_3d else ["N/A"]
            ncols = len(sizes)

            fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(6.0 * ncols, 4.5), squeeze=False)
            has_data = False

            for c_idx, sz in enumerate(sizes):
                ax1 = axes[0, c_idx]
                cond = (df_ctx["stressor_class"] == sc)
                if is_3d and "size" in df_ctx.columns:
                    cond = cond & (df_ctx["size"] == sz)

                sub = df_ctx[cond].groupby("num_workers", observed=False).mean(numeric_only=True).reindex(N_ORDER)

                lines = []
                labels = []
                if "perf_cache_misses" in sub.columns and sub["perf_cache_misses"].notna().any():
                    l1 = ax1.plot(sub.index, sub["perf_cache_misses"], 'r-o', linewidth=1.8, label="Cache Misses")
                    lines.extend(l1)
                    labels.append("Cache Misses")
                    has_data = True

                if "perf_branch_misses" in sub.columns and sub["perf_branch_misses"].notna().any():
                    l2 = ax1.plot(sub.index, sub["perf_branch_misses"], 'm--^', linewidth=1.8, label="Branch Misses")
                    lines.extend(l2)
                    labels.append("Branch Misses")
                    has_data = True

                ax1.set_xlabel("Livelli Carico Worker (N)", fontsize=10, fontweight="bold")
                ax1.set_ylabel("Miss Counter", fontsize=10, fontweight="bold")
                ax1.grid(True, linestyle="--", alpha=0.6)

                if "avg_cpu_temp_c" in sub.columns and sub["avg_cpu_temp_c"].notna().any():
                    ax2 = ax1.twinx()
                    l3 = ax2.plot(sub.index, sub["avg_cpu_temp_c"], 'b-s', linewidth=1.8, label="CPU Temp (°C)")
                    lines.extend(l3)
                    labels.append("CPU Temp (°C)")
                    ax2.set_ylabel("Temperatura CPU (°C)", fontsize=10, color="blue")
                    has_data = True

                header = f"Microarchitettura: {sc_clean.upper()} (t_tot)" + (f" | Size: {sz}" if is_3d else "")
                ax1.set_title(header, fontsize=11, fontweight="bold")
                if lines:
                    ax1.legend(lines, labels, loc="best", frameon=True, fontsize=9)

            if has_data:
                prefix = f"{ctx_label}_" if ctx_label else ""
                fig.suptitle(f"Metriche Microarchitetturali - Stressor: {sc_clean.upper()}" + (f" ({ctx_label})" if ctx_label else ""), fontsize=13, fontweight="bold")
                fig.tight_layout()
                out_p = FIGURES_DIR / f"{prefix}{sc_clean}_microarchitecture_bottleneck.png"
                fig.savefig(out_p, dpi=300, bbox_inches="tight")
                print(f"   [OK] Grafico microarchitettura salvato: {out_p.name}")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualizzazione dei colli di bottiglia hardware e serie temporali.")
    parser.add_argument("--data-dir", type=str, default=None, help="Cartella radice con le serie temporali (default: resultAGX o data_timeseries).")
    parser.add_argument("--model", type=str, default=None, help="Filtra per modello (es. slm-1b, llm-8b).")
    parser.add_argument("--device", type=str, default=None, help="Filtra per device (es. cpu, gpu).")
    parser.add_argument("--benchmark", type=str, default=None, help="Filtra per benchmark (es. mmlu, spider).")
    parser.add_argument("--duration", type=str, default="t_tot", help="Durata da filtrare (default: t_tot).")
    parser.add_argument("--max-reps", type=int, default=20, help="Numero massimo di ripetizioni (1..N) da considerare (default: 20; 0=tutte).")
    args = parser.parse_args()

    search_dir = Path(args.data_dir) if args.data_dir else None
    print(f"\n[ANALISI HARDWARE] Parsing serie temporali in: {search_dir or (AGX_DIR if AGX_DIR.exists() else TIMESERIES_DIR)}")
    df_hw = parse_timeseries_data(
        target_duration=args.duration,
        search_dir=search_dir,
        model_filter=args.model,
        device_filter=args.device,
        bench_filter=args.benchmark,
        max_reps=args.max_reps,
    )
    plot_hardware_groups(df_hw, target_duration=args.duration)
    plot_master_microarchitecture(
        target_duration=args.duration,
        model_filter=args.model,
        device_filter=args.device,
        bench_filter=args.benchmark,
    )
    print(f"\n[COMPLETATO] Tutti i grafici hardware sono stati esportati in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
