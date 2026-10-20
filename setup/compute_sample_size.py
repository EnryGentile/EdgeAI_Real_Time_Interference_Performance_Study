#!/usr/bin/env python3
"""
compute_sample_size.py - Calcolo del numero ottimo di campioni (n) tramite Teorema del Limite Centrale (CLT).

Partendo dai risultati della Baseline (n_0 >= 40 > 30 campioni IID), applica la formula
con valore critico della Z-table (default: Confidenza 95% -> z = 1.96) e margine di errore
relativo massimo r (default: 5% -> r = 0.05):

    n = ceil( ( (z * s_0) / (r * x_bar_0) )^2 )

Permette di:
  1. Selezionare la baseline tramite --model, --device, --benchmark oppure passando un file --csv.
  2. Calcolare n sia per le metriche del modello (llama.cpp) sia per le statistiche di sistema (vmstat).
  3. Aggiornare opzionalmente 'sample_count' in setup/run_config.json (--update-config).
"""

import os
import sys
import csv
import json
import math
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_PATH = SCRIPT_DIR / "run_config.json"

# Valori critici bilaterali z_{1 - alpha/2} della distribuzione Normale Standard N(0,1)
Z_TABLE = {
    0.80: 1.2816,
    0.85: 1.4395,
    0.90: 1.6449,
    0.95: 1.9600,
    0.98: 2.3263,
    0.99: 2.5758,
}

# Metriche normalizzate per token (strettamente IID rispetto alla lunghezza variabile delle domande)
PRIMARY_IID_METRICS = [
    "per_token_lat_ms",
    "throughput_tok_s",
    "prompt_ms_per_tok",
]

# Tutte le metriche LLM
ALL_LLM_METRICS = [
    "per_token_lat_ms",
    "throughput_tok_s",
    "prompt_ms_per_tok",
    "ttft_ms",
    "jitter_ms",
]


def get_z_value(confidence: float) -> float:
    """Restituisce il quantile z_{1 - alpha/2} dalla Z-table per il livello di confidenza scelto."""
    conf_norm = confidence / 100.0 if confidence > 1.0 else confidence
    conf_round = round(conf_norm, 2)
    if conf_round in Z_TABLE:
        return Z_TABLE[conf_round]
    # Approssimazione inversa normale (Beasley-Springer-Moro / Acklam) se fuori dai valori standard
    from statistics import NormalDist
    return NormalDist(0, 1).inv_cdf(1.0 - (1.0 - conf_norm) / 2.0)


def compute_metric_sample_size(values: List[float], z_val: float, rel_error: float) -> Optional[Dict[str, float]]:
    """
    Calcola media campionaria (x_bar), deviazione standard campionaria (s),
    coefficiente di variazione (CV = s / x_bar) e il numero di campioni richiesto n.
    """
    clean_vals = [v for v in values if v is not None and not math.isnan(v)]
    count = len(clean_vals)
    if count < 2:
        return None

    x_bar = statistics.mean(clean_vals)
    s_std = statistics.stdev(clean_vals)

    if abs(x_bar) < 1e-12:
        return {
            "n_0": count,
            "mean": 0.0,
            "stdev": round(s_std, 4),
            "cv_pct": 0.0,
            "n_raw": 0.0,
            "n_required": 1,
        }

    cv = s_std / abs(x_bar)
    n_raw = ((z_val * s_std) / (rel_error * abs(x_bar))) ** 2
    n_req = max(1, math.ceil(n_raw))

    return {
        "n_0": count,
        "mean": round(x_bar, 4),
        "stdev": round(s_std, 4),
        "cv_pct": round(cv * 100.0, 2),
        "n_raw": round(n_raw, 4),
        "n_required": n_req,
    }


def resolve_baseline_csv(
    csv_arg: Optional[str],
    model_arg: Optional[str],
    device_arg: Optional[str],
    benchmark_arg: Optional[str],
    repetition_filter: Optional[int] = None
) -> Path:
    """Individua il file CSV della baseline da analizzare."""
    if csv_arg:
        p = Path(csv_arg)
        if not p.exists():
            print(f"[ERRORE] File CSV non trovato: {p}")
            sys.exit(1)
        return p

    # Se non specificato, prova a leggere da run_config.json
    model_id = "slm-1b"
    device = "gpu"
    bench = "mmlu"

    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            model_id = cfg.get("model_id", "SLM-1B").lower()
            device = cfg.get("device", "gpu").lower()
            bench = cfg.get("benchmark_name", "mmlu").lower()
        except Exception:
            pass

    if model_arg:
        m = model_arg.strip().lower()
        model_id = "slm-1b" if m in ("slm", "1b", "slm-1b") else "llm-8b"
    if device_arg:
        device = device_arg.strip().lower()
    if benchmark_arg:
        bench = benchmark_arg.strip().lower()

    candidate = RESULTS_DIR / f"baseline_{model_id}_{device}_{bench}_tasks.csv"
    if candidate.exists():
        return candidate

    # Fallback: cerca qualsiasi file baseline_*_tasks.csv in results/
    all_baselines = sorted(RESULTS_DIR.glob("baseline_*_tasks.csv"))
    if all_baselines:
        return all_baselines[0]

    print(
        f"[ERRORE] Nessun file di baseline trovato (atteso: {candidate}).\n"
        "Esegui prima la baseline (40 task senza stressor) con:\n"
        f"  python setup/run_experiments.py --model {model_id} --device {device} --benchmark {bench} --no-stressor --samples 40"
    )
    sys.exit(1)


def load_csv_columns(csv_path: Path, repetition_filter: Optional[int] = None) -> Tuple[int, Dict[str, List[float]]]:
    """Carica tutte le colonne numeriche dal CSV della baseline."""
    columns: Dict[str, List[float]] = {}
    row_count = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if repetition_filter is not None and "repetition" in row:
                try:
                    if int(row["repetition"]) != repetition_filter:
                        continue
                except ValueError:
                    pass

            row_count += 1
            for k, val_str in row.items():
                if k in ("task_index", "exp_id", "model_id", "device", "benchmark", "stressor_class", "stressor_method", "num_workers", "size", "duration", "repetition"):
                    continue
                try:
                    val = float(val_str)
                    columns.setdefault(k, []).append(val)
                except (ValueError, TypeError):
                    continue

    return row_count, columns


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calcolo del numero di campioni (n) dalla Baseline tramite CLT e Z-table (default: IC 95%, Errore 5%)."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Percorso al CSV della baseline (es. results/baseline_slm-1b_gpu_mmlu_tasks.csv)"
    )
    parser.add_argument("-m", "--model", type=str, default=None, help="Modello della baseline ('slm' o 'llm')")
    parser.add_argument("-d", "--device", type=str, choices=["cpu", "gpu", "CPU", "GPU"], default=None, help="Allocazione ('cpu' o 'gpu')")
    parser.add_argument("-b", "--benchmark", type=str, choices=["mmlu", "spider", "MMLU", "SPIDER"], default=None, help="Benchmark ('mmlu' o 'spider')")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Livello di confidenza (default: 0.95 per 95%% -> z = 1.96)"
    )
    parser.add_argument(
        "--error",
        type=float,
        default=0.05,
        help="Margine di errore relativo massimo (default: 0.05 per 5%%)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Numero minimo di campioni di sicurezza da garantire per run sotto stressor (default: 5)"
    )
    parser.add_argument(
        "--repetition",
        type=int,
        default=None,
        help="Filtra una specifica ripetizione della baseline (es. 1 per valutare solo i primi 40 campioni)"
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        default=False,
        help="Aggiorna automaticamente 'sample_count' in setup/run_config.json con il valore n calcolato"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    conf = args.confidence / 100.0 if args.confidence > 1.0 else args.confidence
    err = args.error / 100.0 if args.error > 1.0 else args.error
    z_val = get_z_value(conf)

    csv_path = resolve_baseline_csv(args.csv, args.model, args.device, args.benchmark, args.repetition)
    n_0, cols = load_csv_columns(csv_path, repetition_filter=args.repetition)

    print("\n" + "=" * 78)
    print(" CALCOLO DIMENSIONAMENTO CAMPIONE (CLT + Z-TABLE) PER IL PIANO FATTORIALE")
    print("=" * 78)
    print(f" * File Baseline analizzato : {csv_path}")
    print(f" * Campioni Baseline (n_0)  : {n_0} {'[OK >= 30: Teorema del Limite Centrale valido]' if n_0 >= 30 else '[WARNING < 30: consigliati almeno 30-40 campioni]'}")
    print(f" * Intervallo di Confidenza : {conf * 100:.1f}% (alpha = {1.0 - conf:.3f} -> z_{{1-alpha/2}} = {z_val:.4f})")
    print(f" * Errore Relativo Massimo  : {err * 100:.1f}% (r = {err:.4f})")
    print(f" * Formula applicata        : n = ceil( ( ({z_val:.2f} * s_0) / ({err:.2f} * x_bar_0) )^2 )")
    print("=" * 78)

    results_summary: Dict[str, Dict[str, float]] = {}
    for col_name, values in cols.items():
        res = compute_metric_sample_size(values, z_val=z_val, rel_error=err)
        if res is not None:
            results_summary[col_name] = res

    # Stampa Tabella Metriche Modello (llama.cpp)
    print("\n1. METRICHE DEL MODELLO (llama.cpp):")
    print(f" {'Metrica':<24s} | {'n_0':>5s} | {'Media (x_bar)':>14s} | {'Dev.Std (s_0)':>14s} | {'CV (%)':>8s} | {'n calcolato':>11s}")
    print("-" * 88)

    iid_n_candidates = []
    all_llm_n_candidates = []

    for m in ALL_LLM_METRICS:
        if m in results_summary:
            r = results_summary[m]
            tag = " *" if m in PRIMARY_IID_METRICS else "  "
            print(
                f" {m + tag:<24s} | {r['n_0']:5d} | {r['mean']:14.4f} | "
                f"{r['stdev']:14.4f} | {r['cv_pct']:7.2f}% | {r['n_required']:11d}"
            )
            all_llm_n_candidates.append(r["n_required"])
            if m in PRIMARY_IID_METRICS:
                iid_n_candidates.append(r["n_required"])

    # Eventuali metriche vmstat presenti nel CSV
    vmstat_cols = [c for c in results_summary.keys() if c.startswith("vmstat_") or c in ("us", "sy", "id", "wa", "r", "cs", "in", "free_kb", "cpu_temp_c", "gpu_temp_c")]
    if vmstat_cols:
        print("\n2. METRICHE DI SISTEMA PRINCIPALI (vmstat / sensori):")
        print(f" {'Metrica':<24s} | {'n_0':>5s} | {'Media (x_bar)':>14s} | {'Dev.Std (s_0)':>14s} | {'CV (%)':>8s} | {'n calcolato':>11s}")
        print("-" * 88)
        for vc in vmstat_cols:
            r = results_summary[vc]
            if r["mean"] > 0:
                print(
                    f" {vc:<24s} | {r['n_0']:5d} | {r['mean']:14.4f} | "
                    f"{r['stdev']:14.4f} | {r['cv_pct']:7.2f}% | {r['n_required']:11d}"
                )

    # Determinazione del numero di campioni consigliato per gli esperimenti con gli stressor
    if iid_n_candidates:
        recommended_raw = max(iid_n_candidates)
    elif all_llm_n_candidates:
        recommended_raw = max(all_llm_n_candidates)
    else:
        recommended_raw = max((r["n_required"] for r in results_summary.values()), default=10)

    recommended_n = max(args.min_samples, recommended_raw)

    print("\n" + "=" * 78)
    print(f" -> NUMERO DI CAMPIONI CONSIGLIATO PER GLI STRESSOR: n = {recommended_n}")
    print(f"    (Calcolato sulle metriche IID normalizzate per token [*], min_samples={args.min_samples})")
    print("=" * 78)

    if args.update_config and CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["sample_count"] = recommended_n
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            print(f"\n [OK] Aggiornato 'sample_count' = {recommended_n} in {CONFIG_PATH}")
        except Exception as e:
            print(f"\n [WARNING] Impossibile aggiornare {CONFIG_PATH}: {e}")

    print("\nComando pronto per lanciare gli esperimenti con gli stressor usando questo n:")
    print(f"  python setup/run_experiments.py --use-stressor --stressor <cpu|mem|io|cache|sched|interrupt> --samples {recommended_n}\n")


if __name__ == "__main__":
    main()
