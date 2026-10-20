#!/usr/bin/env python3
"""
setup_run.py - Istanziazione dell'ambiente di esecuzione per la Traccia A1.

Questo script riceve da riga di comando:
  1. Modello da allocare: SLM (Llama-3.2-1B) o LLM (Llama-3.1-8B)
  2. Target di allocazione: 'cpu' (-ngl 0) o 'gpu' (-ngl 99)
  3. Benchmark da utilizzare: 'mmlu' (prefill-heavy) o 'spider' (decode-heavy)
  4. Attivazione stressor: se avviare o meno gli stressor (--use-stressor / --no-stressor)
  5. Quale stressor eseguire: classe o metodo stress-ng (es. cpu, mem, io, cache, sched, interrupt)
  6. Numero di campioni (task) da eseguire: -n / --samples (default: 40 per baseline, o n da Z-table)
  7. Ripetizioni per esperimento (-r / --repetitions, default: 3) e attesa transitorio post-reboot (default: 90s)

Include blocchi try/except e meccanismi di fallback automatici (es. rigenerazione dataset corrotto,
fallback GPU -> CPU se l'acceleratore non è disponibile, scrittura atomica con backup di run_config.json).
"""

import os
import sys
import json
import shutil
import argparse
import traceback
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple

# ==============================================================================
# RISOLUZIONE DEI PERCORSI DEL PROGETTO
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if (Path.cwd() / "models").exists() or (Path.cwd() / "llama.cpp").exists():
    BASE_DIR = Path.cwd()
else:
    BASE_DIR = PROJECT_ROOT

MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"
LLAMA_CPP_DIR = BASE_DIR / "llama.cpp"
CONFIG_OUTPUT_PATH = SCRIPT_DIR / "run_config.json"
CONFIG_BACKUP_PATH = SCRIPT_DIR / "run_config.json.bak"
ERROR_LOG_PATH = SCRIPT_DIR / "setup_run_error.log"

# ==============================================================================
# DEFINIZIONE MODELLI, BENCHMARK E STRESSOR SUPPORTATI
# ==============================================================================
KNOWN_MODELS = {
    "slm": {
        "id": "SLM-1B",
        "role": "SLM (Small Language Model - 1B)",
        "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "description": "Llama-3.2-1B-Instruct quantizzato Q4_K_M"
    },
    "llm": {
        "id": "LLM-8B",
        "role": "LLM (Large Language Model - 8B)",
        "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "description": "Meta-Llama-3.1-8B-Instruct quantizzato Q4_K_M"
    }
}

MODEL_ALIASES = {
    "slm": "slm",
    "1b": "slm",
    "slm-1b": "slm",
    "llama-3.2-1b": "slm",
    "llama-3.2-1b-instruct": "slm",
    "llama-3.2": "slm",
    "llm": "llm",
    "8b": "llm",
    "llm-8b": "llm",
    "llama-3.1-8b": "llm",
    "llama-3.1-8b-instruct": "llm",
    "llama-3.1": "llm"
}

KNOWN_BENCHMARKS = {
    "mmlu": {
        "name": "MMLU",
        "type": "Prefill-heavy (Compute-bound)",
        "file": "mmlu_first_40.json",
        "fallback_file": "mmlu.json",
        "default_max_tokens": 32,
        "description": "Massive Multitask Language Understanding (Multiple Choice QA)"
    },
    "spider": {
        "name": "Spider",
        "type": "Decode-heavy (Memory-bandwidth-bound)",
        "file": "spider_first_40.json",
        "fallback_file": "spider.json",
        "default_max_tokens": 128,
        "description": "Complex Text-to-SQL Benchmark"
    }
}

KNOWN_STRESSORS = {
    "none": {
        "class": "none",
        "has_size_factor": False,
        "description": "Nessuno stressor (Baseline a riposo)"
    },
    "cpu": {
        "class": "cpu",
        "has_size_factor": False,
        "default_method": "cpu",
        "methods": ["cpu", "matrix", "fft"],
        "description": "Stressor CPU (compute-bound: cpu, matrix, fft)"
    },
    "mem": {
        "class": "mem",
        "has_size_factor": True,
        "default_method": "vm",
        "methods": ["vm", "stream", "memrate"],
        "description": "Stressor Memoria (DRAM bandwidth/allocazione: vm, stream, memrate)"
    },
    "cache": {
        "class": "cache",
        "has_size_factor": False,
        "default_method": "cache",
        "methods": ["cache", "l1cache"],
        "description": "Stressor Cache gerarchica (cache, l1cache)"
    },
    "io": {
        "class": "io",
        "has_size_factor": True,
        "default_method": "iomix",
        "methods": ["iomix", "hdd"],
        "description": "Stressor I/O e filesystem (iomix, hdd)"
    },
    "sched": {
        "class": "sched",
        "has_size_factor": False,
        "default_method": "switch",
        "methods": ["switch", "fork", "sched"],
        "description": "Stressor Scheduler/OS (switch, fork)"
    },
    "interrupt": {
        "class": "interrupt",
        "has_size_factor": False,
        "default_method": "timer",
        "methods": ["timer", "sigq"],
        "description": "Stressor Interrupt (timer, sigq)"
    }
}

STRESSOR_ALIASES = {
    "none": ("none", "none"),
    "baseline": ("none", "none"),
    "cpu": ("cpu", "cpu"),
    "matrix": ("cpu", "matrix"),
    "fft": ("cpu", "fft"),
    "mem": ("mem", "vm"),
    "memory": ("mem", "vm"),
    "vm": ("mem", "vm"),
    "stream": ("mem", "stream"),
    "memrate": ("mem", "memrate"),
    "cache": ("cache", "cache"),
    "l1cache": ("cache", "l1cache"),
    "io": ("io", "iomix"),
    "iomix": ("io", "iomix"),
    "hdd": ("io", "hdd"),
    "sched": ("sched", "switch"),
    "switch": ("sched", "switch"),
    "fork": ("sched", "fork"),
    "interrupt": ("interrupt", "timer"),
    "timer": ("interrupt", "timer"),
    "sigq": ("interrupt", "sigq"),
}


@dataclass
class RunConfig:
    """Configurazione istanziata per l'esecuzione del benchmark e del DoE."""
    model_id: str
    model_role: str
    model_path: str
    device: str
    n_gpu_layers: int
    benchmark_name: str
    benchmark_type: str
    dataset_path: str
    dataset_total_samples: int
    sample_count: int
    use_stressor: bool
    stressor_class: str
    stressor_method: str
    has_size_factor: bool
    repetitions: int
    transient_wait_sec: int
    reboot_after_run: bool
    ssh_target: Optional[str]
    ssh_dest: str
    ssh_port: int
    ssh_key: Optional[str]
    seed: int
    llama_cli_path: str
    temperature: float
    max_tokens: int
    n_threads: int
    example_cmd: List[str]


def log_runtime_error(context: str, exc: Exception):
    """Salva lo stacktrace completo in setup/setup_run_error.log."""
    try:
        ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n=== ERRORE RUNTIME [{context}] ===\n")
            f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            f.write("\n")
    except Exception:
        pass


# ==============================================================================
# FUNZIONI DI FORMATTAZIONE PROMPT CON FALLBACK DIFENSIVO
# ==============================================================================
def format_mmlu_prompt(sample: Dict[str, Any]) -> str:
    """Formatta un campione MMLU nel formato atteso dal modello con fallback in caso di chiavi mancanti."""
    try:
        if not isinstance(sample, dict):
            raise TypeError(f"Atteso dict per campione MMLU, ricevuto {type(sample)}")
        q = str(sample.get("question") or "Domanda di test MMLU").strip()
        choices = sample.get("choices", [])
        if not isinstance(choices, list) or len(choices) == 0:
            choices = ["Opzione A", "Opzione B", "Opzione C", "Opzione D"]
        choices_str = ""
        labels = ["A", "B", "C", "D"]
        for i, choice in enumerate(choices[:4]):
            choices_str += f"{labels[i]}. {choice}\n"
        return f"Domanda: {q}\nOpzioni:\n{choices_str}Rispondi solo con la lettera della risposta corretta:"
    except Exception as e:
        log_runtime_error("format_mmlu_prompt", e)
        return "Domanda: Qual è la complessità della ricerca binaria?\nOpzioni:\nA. O(log n)\nB. O(n)\nC. O(n^2)\nD. O(1)\nRispondi solo con la lettera della risposta corretta:"


def format_spider_prompt(sample: Dict[str, Any]) -> str:
    """Formatta un campione Spider (Text-to-SQL) nel formato atteso dal modello con fallback."""
    try:
        if not isinstance(sample, dict):
            raise TypeError(f"Atteso dict per campione Spider, ricevuto {type(sample)}")
        q = str(sample.get("question") or "Mostra tutti i record della tabella principale").strip()
        db_id = str(sample.get("db_id") or "database").strip()
        return f"Converti la seguente richiesta in una query SQL per il database '{db_id}'.\nRichiesta: {q}\nSQL:"
    except Exception as e:
        log_runtime_error("format_spider_prompt", e)
        return "Converti la seguente richiesta in una query SQL per il database 'telemetry'.\nRichiesta: Seleziona tutti i sensori attivi ordinati per id.\nSQL:"


# ==============================================================================
# CONTROLLI DI INTEGRITÀ (CHECKS) E FALLBACK HARDWARE/ASSET
# ==============================================================================
def find_llama_cli() -> Optional[Path]:
    """Cerca il binario llama-cli nei percorsi standard di compilazione o nel PATH."""
    candidate_paths = [
        LLAMA_CPP_DIR / "build" / "bin" / "llama-cli",
        LLAMA_CPP_DIR / "build" / "bin" / "llama-cli.exe",
        LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-cli.exe",
        LLAMA_CPP_DIR / "llama-cli",
        LLAMA_CPP_DIR / "llama-cli.exe",
    ]
    for p in candidate_paths:
        try:
            if p.exists() and os.access(p, os.X_OK if os.name != "nt" else os.F_OK):
                return p
        except Exception:
            continue

    which_path = shutil.which("llama-cli")
    if which_path:
        return Path(which_path)

    return None


def _is_gpu_available_on_system() -> Tuple[bool, str]:
    """Verifica se una GPU NVIDIA (dGPU con nvidia-smi o iGPU Tegra su Jetson Orin) è presente."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            smi_output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5
            ).strip()
            if smi_output:
                return True, f"GPU NVIDIA rilevata via nvidia-smi: {smi_output}"
        except Exception:
            pass

    # Controllo nodi device NVIDIA Jetson (Tegra iGPU con memoria unificata)
    tegra_gpu_paths = [
        "/dev/nvhost-gpu",
        "/dev/nvhost-ctrl-gpu",
        "/sys/devices/gpu.0",
        "/sys/bus/platform/drivers/nvgpu",
    ]
    for tp in tegra_gpu_paths:
        if os.path.exists(tp):
            return True, f"GPU integrata NVIDIA Jetson (Tegra) rilevata ({tp})"

    return False, "Nessun dispositivo GPU NVIDIA/Tegra rilevato."


def check_system_prerequisites(
    model_key: str,
    device: str,
    benchmark_key: str,
    use_stressor: bool,
    requested_samples: int
) -> Tuple[Path, Path, Path, int, int, str, bool]:
    """
    Esegue tutti i check preliminari con gestione errori try/except e fallback automatici:
      - Se il dataset è assente/corrotto, tenta di rigenerarlo tramite il fallback di setup_system.
      - Se device='gpu' ma nessuna GPU è rilevata, effettua fallback sicuro su device='cpu'.
    """
    errors = []
    warnings = []
    resolved_device = device
    resolved_use_stressor = use_stressor

    print("\n" + "=" * 65)
    print(" 1. VERIFICA PREREQUISITI SISTEMA E ASSET (POST setup_system.py)")
    print("=" * 65)

    # 1. Check binario llama.cpp
    llama_cli = find_llama_cli()
    if llama_cli is None:
        errors.append(
            f"Binario 'llama-cli' non trovato in {LLAMA_CPP_DIR / 'build' / 'bin'} né nel PATH.\n"
            "   -> Esegui prima 'python setup/setup_system.py' per compilare llama.cpp."
        )
    else:
        print(f" [OK] Runtime llama-cli trovato: {llama_cli}")

    # 2. Check file modello GGUF
    model_info = KNOWN_MODELS[model_key]
    model_file = model_info["file"]
    model_path = MODELS_DIR / model_file

    if not model_path.exists():
        # Fallback: se il modello richiesto manca ma l'altro modello è presente su disco, segnala
        errors.append(
            f"File modello '{model_file}' non trovato in {MODELS_DIR}.\n"
            "   -> Esegui prima 'python setup/setup_system.py' per scaricare i modelli LLaMA-3."
        )
    else:
        try:
            file_size_gb = model_path.stat().st_size / (1024 ** 3)
            if file_size_gb < 0.1:
                errors.append(
                    f"File modello {model_file} presente ma di dimensioni anomale ({file_size_gb:.2f} GB). "
                    "Potrebbe essere corrotto o un puntatore Git LFS non scaricato."
                )
            else:
                print(f" [OK] Modello {model_info['id']} presente ({file_size_gb:.2f} GB): {model_path}")
        except Exception as e:
            log_runtime_error("check_model_size", e)
            errors.append(f"Impossibile accedere alle proprietà del file modello {model_path}: {e}")

    # 3. Check dataset benchmark con fallback di rigenerazione
    bench_info = KNOWN_BENCHMARKS[benchmark_key]
    dataset_file = bench_info["file"]
    dataset_path = DATASETS_DIR / dataset_file
    if not dataset_path.exists():
        fallback_path = DATASETS_DIR / bench_info["fallback_file"]
        if fallback_path.exists():
            dataset_path = fallback_path
            warnings.append(f"File {dataset_file} non trovato, utilizzato fallback {bench_info['fallback_file']}.")
        else:
            print(f" [FALLBACK] Dataset '{dataset_file}' assente. Tentativo di generazione automatica di fallback...")
            try:
                from setup_system import _generate_fallback_mmlu, _generate_fallback_spider, _save_json_atomic
                DATASETS_DIR.mkdir(parents=True, exist_ok=True)
                fallback_data = _generate_fallback_mmlu(40) if benchmark_key == "mmlu" else _generate_fallback_spider(40)
                _save_json_atomic(dataset_path, fallback_data)
                warnings.append(f"Generato dataset locale di fallback in {dataset_path} (40 campioni).")
            except Exception as e:
                log_runtime_error("dataset_auto_fallback", e)
                errors.append(
                    f"Dataset per benchmark '{benchmark_key}' non trovato in {DATASETS_DIR} (atteso: {dataset_file}).\n"
                    "   -> Esegui prima 'python setup/setup_system.py' per generare i subset dei benchmark."
                )

    dataset_samples = []
    if dataset_path.exists():
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset_samples = json.load(f)
            if not isinstance(dataset_samples, list) or len(dataset_samples) == 0:
                raise ValueError(f"Il dataset {dataset_path.name} è vuoto o non è una lista JSON valida.")
            print(f" [OK] Dataset '{bench_info['name']}' valido ({len(dataset_samples)} campioni disponibili): {dataset_path}")
            if requested_samples > len(dataset_samples):
                warnings.append(
                    f"Richiesti {requested_samples} campioni ma il dataset ne contiene {len(dataset_samples)}. "
                    f"I campioni verranno ciclati con modulo {len(dataset_samples)}."
                )
            first_sample = dataset_samples[0]
            _ = format_mmlu_prompt(first_sample) if benchmark_key == "mmlu" else format_spider_prompt(first_sample)
            print(f" [OK] Formattazione prompt validata sul campione 1.")
        except Exception as e:
            log_runtime_error("parse_dataset_json", e)
            errors.append(f"Errore nel parsing del dataset JSON {dataset_path}: {e}")

    # 4. Check vmstat e stress-ng
    vmstat_bin = shutil.which("vmstat")
    if vmstat_bin:
        print(f" [OK] Tool di monitoraggio sistema 'vmstat' trovato: {vmstat_bin}")
    else:
        warnings.append("Tool 'vmstat' non trovato nel PATH (verrà utilizzato il campionatore di fallback interno).")

    if resolved_use_stressor:
        stress_ng_bin = shutil.which("stress-ng")
        if stress_ng_bin:
            print(f" [OK] Generatore di interferenza 'stress-ng' trovato: {stress_ng_bin}")
        else:
            warnings.append(
                "Tool 'stress-ng' non trovato nel PATH! Su Linux/Jetson installalo con: sudo apt-get install -y stress-ng"
            )

    # 5. Check device (CPU o GPU): se è richiesto GPU e fallisce, lo script deve fallire e riportare l'errore
    print("\n" + "=" * 65)
    print(" 2. VERIFICA ALLOCAZIONE HARDWARE (Target richiesto: %s)" % resolved_device.upper())
    print("=" * 65)

    n_cpu_cores = os.cpu_count() or 4
    if resolved_device == "cpu":
        print(f" [OK] Allocazione selezionata: CPU-bound (-ngl 0).")
        print(f" [INFO] Core CPU disponibili per l'inferenza: {n_cpu_cores}")
    elif resolved_device == "gpu":
        gpu_ok, gpu_msg = _is_gpu_available_on_system()
        if gpu_ok:
            print(f" [OK] Allocazione selezionata: GPU-offload completo (-ngl 99).")
            print(f" [OK] {gpu_msg}")
        else:
            errors.append(
                f"Target hardware 'gpu' selezionato, ma la verifica GPU è fallita: {gpu_msg}"
            )

    for w in warnings:
        print(f" [WARNING] {w}")

    if errors:
        print("\n" + "!" * 65)
        print(" [ERRORE BLOCCANTE] Uno o più check sono falliti:")
        for err in errors:
            print(f"  - {err}")
        print("!" * 65)
        sys.exit(1)

    return llama_cli, model_path, dataset_path, len(dataset_samples), n_cpu_cores, resolved_device, resolved_use_stressor


# ==============================================================================
# ISTANZIAZIONE DELLA CONFIGURAZIONE CON SCRITTURA ATOMICA E BACKUP
# ==============================================================================
def resolve_stressor(use_stressor: bool, stressor_input: Optional[str]):
    """Risolve la classe e il metodo dello stressor in base ai flag utente."""
    if not use_stressor:
        return False, "none", "none", False

    s_clean = (stressor_input or "cpu").strip().lower()
    if s_clean in ["none", "baseline", "off", "false"]:
        return False, "none", "none", False

    if s_clean not in STRESSOR_ALIASES:
        valid_opts = [k for k in STRESSOR_ALIASES.keys() if k not in ("none", "baseline")]
        print(f"[WARNING] Stressor '{stressor_input}' non riconosciuto. Scelte valide: {valid_opts}")
        print("[FALLBACK] Ripiego automatico sullo stressor di default 'cpu'.")
        s_clean = "cpu"

    s_class, s_method = STRESSOR_ALIASES[s_clean]
    has_size = KNOWN_STRESSORS[s_class]["has_size_factor"]
    return True, s_class, s_method, has_size


def save_config_safely(config: RunConfig):
    """Salva run_config.json in modo atomico con backup di fallback."""
    try:
        if CONFIG_OUTPUT_PATH.exists():
            shutil.copy2(CONFIG_OUTPUT_PATH, CONFIG_BACKUP_PATH)
        tmp_path = CONFIG_OUTPUT_PATH.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(CONFIG_OUTPUT_PATH)
    except Exception as e:
        log_runtime_error("save_config_safely", e)
        print(f"[ERRORE] Fallito salvataggio di {CONFIG_OUTPUT_PATH}: {e}")
        if CONFIG_BACKUP_PATH.exists():
            print(f"[FALLBACK] Ripristino configurazione precedente da {CONFIG_BACKUP_PATH}...")
            shutil.copy2(CONFIG_BACKUP_PATH, CONFIG_OUTPUT_PATH)
        raise


def instantiate_run(
    model_input: str,
    device_input: str,
    benchmark_input: str,
    use_stressor: bool = False,
    stressor_input: Optional[str] = None,
    samples: Optional[int] = None,
    repetitions: int = 3,
    transient_wait_sec: int = 90,
    reboot_after_run: bool = True,
    ssh_target: Optional[str] = None,
    ssh_dest: str = "./EdgeAI_Results",
    ssh_port: int = 22,
    ssh_key: Optional[str] = None,
    threads: Optional[int] = None,
    seed: int = 42
) -> RunConfig:
    """
    Valida gli argomenti, esegue i check e istanzia la configurazione di esecuzione.
    """
    try:
        # 1. Normalizzazione input modello
        m_clean = (model_input or "slm").strip().lower()
        if m_clean in MODEL_ALIASES:
            model_key = MODEL_ALIASES[m_clean]
        elif m_clean in KNOWN_MODELS:
            model_key = m_clean
        else:
            print(f"[WARNING] Modello '{model_input}' non riconosciuto. Fallback su 'slm' (Llama-3.2-1B).")
            model_key = "slm"

        # 2. Normalizzazione input device
        d_clean = (device_input or "gpu").strip().lower()
        if d_clean not in ["cpu", "gpu"]:
            print(f"[WARNING] Device '{device_input}' non valido. Fallback su 'cpu'.")
            d_clean = "cpu"

        # 3. Normalizzazione input benchmark
        b_clean = (benchmark_input or "mmlu").strip().lower()
        if b_clean not in KNOWN_BENCHMARKS:
            print(f"[WARNING] Benchmark '{benchmark_input}' non valido. Fallback su 'mmlu'.")
            b_clean = "mmlu"

        # 4. Risoluzione Stressor
        use_stressor_resolved, s_class, s_method, has_size_factor = resolve_stressor(use_stressor, stressor_input)

        # 5. Determinazione numero di campioni (task) e ripetizioni
        sample_count = samples if (samples is not None and samples > 0) else 40
        reps_clean = max(1, int(repetitions)) if repetitions is not None else 3
        wait_clean = max(0, int(transient_wait_sec)) if transient_wait_sec is not None else 90

        # 6. Esecuzione dei check di sistema (con eventuale fallback GPU -> CPU)
        (
            llama_cli,
            model_path,
            dataset_path,
            dataset_total,
            n_cpu_cores,
            d_clean,
            use_stressor_resolved,
        ) = check_system_prerequisites(
            model_key=model_key,
            device=d_clean,
            benchmark_key=b_clean,
            use_stressor=use_stressor_resolved,
            requested_samples=sample_count
        )

        # 7. Determinazione parametri di esecuzione
        n_gpu_layers = 99 if d_clean == "gpu" else 0
        n_threads = threads if (threads is not None and threads > 0) else n_cpu_cores
        bench_info = KNOWN_BENCHMARKS[b_clean]
        model_info = KNOWN_MODELS[model_key]
        max_tokens = bench_info["default_max_tokens"]
        temperature = 0.0

        example_cmd = [
            str(llama_cli),
            "-m", str(model_path),
            "-ngl", str(n_gpu_layers),
            "-t", str(n_threads),
            "--temp", str(temperature),
            "-s", str(seed),
            "-n", str(max_tokens),
            "--no-display-prompt"
        ]

        config = RunConfig(
            model_id=model_info["id"],
            model_role=model_info["role"],
            model_path=str(model_path),
            device=d_clean,
            n_gpu_layers=n_gpu_layers,
            benchmark_name=bench_info["name"],
            benchmark_type=bench_info["type"],
            dataset_path=str(dataset_path),
            dataset_total_samples=dataset_total,
            sample_count=sample_count,
            use_stressor=use_stressor_resolved,
            stressor_class=s_class,
            stressor_method=s_method,
            has_size_factor=has_size_factor,
            repetitions=reps_clean,
            transient_wait_sec=wait_clean,
            reboot_after_run=reboot_after_run,
            ssh_target=ssh_target,
            ssh_dest=ssh_dest,
            ssh_port=int(ssh_port) if ssh_port else 22,
            ssh_key=ssh_key,
            seed=seed,
            llama_cli_path=str(llama_cli),
            temperature=temperature,
            max_tokens=max_tokens,
            n_threads=n_threads,
            example_cmd=example_cmd
        )

        save_config_safely(config)
        return config

    except SystemExit:
        raise
    except Exception as e:
        log_runtime_error("instantiate_run", e)
        print(f"\n[ERRORE CRITICO] Errore imprevisto durante l'istanziazione della run: {e}")
        print(f"Dettagli salvati in: {ERROR_LOG_PATH}")
        sys.exit(1)


# ==============================================================================
# CLI ENTRYPOINT
# ==============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Istanziazione del setup di esecuzione per la Traccia A1 (llama.cpp + stress-ng su Jetson Orin)."
    )

    parser.add_argument(
        "pos_model",
        nargs="?",
        default=None,
        help="Modello da allocare: 'slm' (Llama-3.2-1B) o 'llm' (Meta-Llama-3.1-8B)"
    )
    parser.add_argument(
        "pos_device",
        nargs="?",
        default=None,
        choices=["cpu", "gpu", "CPU", "GPU"],
        help="Target di allocazione: 'cpu' o 'gpu'"
    )
    parser.add_argument(
        "pos_benchmark",
        nargs="?",
        default=None,
        choices=["mmlu", "spider", "MMLU", "SPIDER"],
        help="Benchmark da eseguire: 'mmlu' o 'spider'"
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Modello da allocare (es. 'slm', 'llm', '1b', '8b')"
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        choices=["cpu", "gpu", "CPU", "GPU"],
        default=None,
        help="Target di allocazione ('cpu' o 'gpu')"
    )
    parser.add_argument(
        "-b", "--benchmark",
        type=str,
        choices=["mmlu", "spider", "MMLU", "SPIDER"],
        default=None,
        help="Benchmark da utilizzare ('mmlu' o 'spider')"
    )

    stressor_group = parser.add_mutually_exclusive_group()
    stressor_group.add_argument(
        "--use-stressor",
        dest="use_stressor",
        action="store_true",
        default=None,
        help="Abilita l'esecuzione con gli stressor (stress-ng)"
    )
    stressor_group.add_argument(
        "--no-stressor",
        dest="use_stressor",
        action="store_false",
        help="Disabilita gli stressor (esegue solo la Baseline a riposo)"
    )

    parser.add_argument(
        "-s", "--stressor",
        type=str,
        default=None,
        help=(
            "Quale stressor utilizzare: 'none' (baseline), classe ('cpu', 'mem', 'io', 'cache', 'sched', 'interrupt') "
            "o metodo specifico ('matrix', 'fft', 'vm', 'stream', 'memrate', 'l1cache', 'iomix', 'hdd', 'switch', 'fork', 'timer', 'sigq')"
        )
    )
    parser.add_argument(
        "-n", "--samples",
        type=int,
        default=40,
        help="Numero di campioni (task) da eseguire per ogni esperimento (default: 40 per baseline, o n calcolato)"
    )
    parser.add_argument(
        "-r", "--repetitions",
        type=int,
        default=3,
        help="Numero di ripetizioni per ogni esperimento (default: 3)"
    )
    parser.add_argument(
        "--transient-wait",
        type=int,
        default=90,
        help="Tempo di attesa in secondi dopo il boot per far esaurire il transitorio (default: 90s per Jetson Orin)"
    )
    parser.add_argument(
        "--no-reboot",
        action="store_true",
        default=False,
        help="Disabilita il riavvio automatico al termine di ogni esperimento (utile per test/debug)"
    )
    parser.add_argument(
        "--ssh-target",
        type=str,
        default=os.environ.get("SSH_TARGET"),
        help="Target SSH sulla tua macchina a cui inviare i CSV a fine run (es. 'gabri@192.168.1.50' o 'gabri@localhost')"
    )
    parser.add_argument(
        "--ssh-dest",
        type=str,
        default=os.environ.get("SSH_DEST", "./EdgeAI_Results"),
        help="Cartella di destinazione sulla tua macchina remota per i file CSV (default: ./EdgeAI_Results)"
    )
    parser.add_argument(
        "--ssh-port",
        type=int,
        default=int(os.environ.get("SSH_PORT", "22")),
        help="Porta SSH della tua macchina (default: 22, oppure es. 2222 se usi tunnel SSH inverso -R)"
    )
    parser.add_argument(
        "--ssh-key",
        type=str,
        default=os.environ.get("SSH_KEY"),
        help="Percorso opzionale alla chiave privata SSH sulla board (es. ~/.ssh/id_ed25519)"
    )
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=None,
        help="Numero di thread CPU da dedicare a llama.cpp (default: core disponibili)"
    )

    args = parser.parse_args()

    model = args.model if args.model is not None else args.pos_model
    device = args.device if args.device is not None else args.pos_device
    benchmark = args.benchmark if args.benchmark is not None else args.pos_benchmark

    if not model or not device or not benchmark:
        parser.print_help()
        print("\n[ERRORE] È necessario specificare modello, device e benchmark.")
        print("Esempio Baseline (40 task, no stressor):")
        print("  python setup/setup_run.py --model slm --device gpu --benchmark mmlu --no-stressor --samples 40")
        print("Esempio Stressor (es. mem, n=12 campioni, 3 ripetizioni):")
        print("  python setup/setup_run.py --model slm --device gpu --benchmark mmlu --use-stressor --stressor mem --samples 12")
        sys.exit(1)

    use_stressor = args.use_stressor
    if use_stressor is None:
        if args.stressor is not None and args.stressor.strip().lower() not in ["none", "baseline", "off", "false"]:
            use_stressor = True
        else:
            use_stressor = False

    return (
        model,
        device,
        benchmark,
        use_stressor,
        args.stressor,
        args.samples,
        args.repetitions,
        args.transient_wait,
        not args.no_reboot,
        args.ssh_target,
        args.ssh_dest,
        args.ssh_port,
        args.ssh_key,
        args.threads
    )


def main():
    try:
        (
            model_arg,
            device_arg,
            benchmark_arg,
            use_stressor_arg,
            stressor_arg,
            samples_arg,
            repetitions_arg,
            transient_wait_arg,
            reboot_arg,
            ssh_target_arg,
            ssh_dest_arg,
            ssh_port_arg,
            ssh_key_arg,
            threads_arg
        ) = parse_arguments()

        print("\n" + "=" * 65)
        print(" AVVIO ISTANZIAZIONE RUN BENCHMARK (Traccia A1)")
        print("=" * 65)
        print(
            f" Parametri: Modello='{model_arg}', Device='{device_arg}', Benchmark='{benchmark_arg}', "
            f"Stressor={'ON (' + str(stressor_arg or 'cpu') + ')' if use_stressor_arg else 'OFF (Baseline)'}, "
            f"Samples={samples_arg}, Ripetizioni={repetitions_arg}"
        )

        config = instantiate_run(
            model_input=model_arg,
            device_input=device_arg,
            benchmark_input=benchmark_arg,
            use_stressor=use_stressor_arg,
            stressor_input=stressor_arg,
            samples=samples_arg,
            repetitions=repetitions_arg,
            transient_wait_sec=transient_wait_arg,
            reboot_after_run=reboot_arg,
            ssh_target=ssh_target_arg,
            ssh_dest=ssh_dest_arg,
            ssh_port=ssh_port_arg,
            ssh_key=ssh_key_arg,
            threads=threads_arg
        )

        print("\n" + "=" * 65)
        print(" [OK] ISTANZIAZIONE COMPLETATA CON SUCCESSO!")
        print("=" * 65)
        print(f" * Modello:           {config.model_role}")
        print(f" * File GGUF:         {config.model_path}")
        print(f" * Target Hardware:   {config.device.upper()} (GPU Layers: -ngl {config.n_gpu_layers})")
        print(f" * Benchmark:         {config.benchmark_name} [{config.benchmark_type}]")
        print(f" * Campioni (Task):   {config.sample_count} (su {config.dataset_total_samples} nel dataset)")
        print(f" * Modalità Stressor: {'ATTIVO' if config.use_stressor else 'DISATTIVO (Baseline)'}")
        if config.use_stressor:
            print(f" * Stressor Scelto:   Classe='{config.stressor_class}', Metodo='{config.stressor_method}' (3D Size S: {config.has_size_factor})")
        print(f" * Ripetizioni/Exp:   {config.repetitions}")
        print(f" * Attesa Transitorio:{config.transient_wait_sec} s (Reboot fine run: {config.reboot_after_run})")
        if config.ssh_target:
            print(f" * Export SSH CSV:    {config.ssh_target}:{config.ssh_dest} (porta {config.ssh_port})")
        print(f" * Runtime:           {config.llama_cli_path} (-t {config.n_threads}, seed={config.seed})")
        print(f" * File Config:       {CONFIG_OUTPUT_PATH}")
        print("=" * 65)
        print("\nPer avviare la campagna sperimentale esegui: python setup/run_experiments.py\n")

    except KeyboardInterrupt:
        print("\n[INTERRUZIONE] Istanziazione interrotta manualmente.")
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as e:
        log_runtime_error("main", e)
        print(f"\n[ERRORE FATALE] {e}. Consulta {ERROR_LOG_PATH}")
        sys.exit(1)


if __name__ == "__main__":
    main()
