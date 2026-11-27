"""
Crea una copia derivata dei risultati MMLU che contiene solo il primo prompt (task_index == 1) di ogni run.

Motivo: nelle campagne MMLU eseguite prima della correzione del 25/11 (bug "stressor solo sul task 1":
stress-ng abortiva quando --temp-path non esisteva) lo stressor resta attivo solo per i primi ~30 s,
cioè durante il primo task. La media sui 30 task diluisce quindi l'effetto dello stressor.

La copia contiene:
  - i CSV per-task (baseline_*_repN_tasks.csv, run_*_repN_tasks.csv) filtrati a task_index == 1;
  - i doe_<modello>_<device>_mmlu_<stressor>.csv ricostruiti dalle metriche del primo task.
I dati originali non vengono modificati.

Uso:
  python statistical_analysis/mmlu_first_prompt.py resultAGX/mmlu <cartella_output>
  python statistical_analysis/analyze_doe_significance.py --data-dir <cartella_output> --model slm-1b \
      --benchmark mmlu --out-dir figures/agx/statistical_reports/slm-1b_mmlu_primo_prompt
"""
import re
import sys
from pathlib import Path

import pandas as pd

TASK_RE = re.compile(r"^(baseline|run)_(slm-1b|llm-8b)_(cpu|gpu)_mmlu_.*rep(\d+)_tasks\.csv$")
DOE_COLS = ["stressor_class", "stressor_method", "num_workers", "size", "duration",
            "ttft_ms", "per_token_lat_ms", "throughput_tok_s", "jitter_ms"]


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)

    frames = []
    for path in src.rglob("*_tasks.csv"):
        if not TASK_RE.match(path.name):
            continue
        df = pd.read_csv(path)
        df = df[pd.to_numeric(df["task_index"], errors="coerce") == 1]
        if df.empty:
            continue
        df.to_csv(dst / path.name, index=False)
        frames.append(df)
    if not frames:
        raise SystemExit(f"[ERRORE] Nessun CSV per-task MMLU in {src}")

    tasks = pd.concat(frames, ignore_index=True)
    tasks["model"] = tasks["model_id"].str.lower()
    tasks["device"] = tasks["device"].str.lower()
    for (model, device, stressor), grp in tasks.groupby(["model", "device", "stressor_class"]):
        grp = grp.sort_values(["num_workers", "size", "repetition"])
        grp[DOE_COLS].to_csv(dst / f"doe_{model}_{device}_mmlu_{stressor}.csv", index=False)
        print(f"{model} {device} {stressor}: {len(grp)} run")


if __name__ == "__main__":
    main()
