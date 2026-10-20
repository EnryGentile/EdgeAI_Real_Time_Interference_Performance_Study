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

NOTA: Lo script effettua l'istanziazione e i controlli di integrità (check) dopo
l'esecuzione di setup_system.py, salvando la configurazione in setup/run_config.json.
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

# ==============================================================================
# RISOLUZIONE DEI PERCORSI DEL PROGETTO
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Rilevamento cartella base (supporta sia esecuzione da root sia da ./setup)
if (Path.cwd() / "models").exists() or (Path.cwd() / "llama.cpp").exists():
    BASE_DIR = Path.cwd()
else:
    BASE_DIR = PROJECT_ROOT

MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"
LLAMA_CPP_DIR = BASE_DIR / "llama.cpp"
CONFIG_OUTPUT_PATH = SCRIPT_DIR / "run_config.json"

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

# Mappatura classi e metodi stress-ng (come da specifica Traccia A1)
KNOWN_STRESSORS = {
    "none": {
        "class": "none",
        "has_size_factor": False,
        "description": "Nessuno stressor (Baseline a riposo)"
    },
    # Classi principali
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

# Alias per selezionare direttamente anche il singolo stressor specifico di stress-ng
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


# ==============================================================================
# FUNZIONI DI FORMATTAZIONE PROMPT
# ==============================================================================
def format_mmlu_prompt(sample: Dict[str, Any]) -> str:
    """Formatta un campione MMLU nel formato atteso dal modello."""
    q = sample.get("question", "")
    choices = sample.get("choices", [])
    choices_str = ""
    labels = ["A", "B", "C", "D"]
    for i, choice in enumerate(choices):
        if i < len(labels):
            choices_str += f"{labels[i]}. {choice}\n"
    return f"Domanda: {q}\nOpzioni:\n{choices_str}Rispondi solo con la lettera della risposta corretta:"


def format_spider_prompt(sample: Dict[str, Any]) -> str:
    """Formatta un campione Spider (Text-to-SQL) nel formato atteso dal modello."""
    q = sample.get("question", "")
    db_id = sample.get("db_id", "database")
    return f"Converti la seguente richiesta in una query SQL per il database '{db_id}'.\nRichiesta: {q}\nSQL:"


# ==============================================================================
# CONTROLLI DI INTEGRITÀ (CHECKS)
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
        if p.exists() and os.access(p, os.X_OK if os.name != "nt" else os.F_OK):
            return p

    which_path = shutil.which("llama-cli")
    if which_path:
        return Path(which_path)

    return None


def check_system_prerequisites(
    model_key: str,
    device: str,
    benchmark_key: str,
    use_stressor: bool,
    requested_samples: int
):
    """
    Esegue tutti i check preliminari per verificare che setup_system.py
    sia stato eseguito correttamente e che le risorse richieste siano pronte.
    """
    errors = []
    warnings = []

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
        errors.append(
            f"File modello '{model_file}' non trovato in {MODELS_DIR}.\n"
            "   -> Esegui prima 'python setup/setup_system.py' per scaricare i modelli LLaMA-3."
        )
    else:
        file_size_gb = model_path.stat().st_size / (1024 ** 3)
        if file_size_gb < 0.1:
            errors.append(
                f"File modello {model_file} presente ma di dimensioni anomale ({file_size_gb:.2f} GB). "
                "Potrebbe essere corrotto o un puntatore Git LFS non scaricato."
            )
        else:
            print(f" [OK] Modello {model_info['id']} presente ({file_size_gb:.2f} GB): {model_path}")

    # 3. Check dataset benchmark
    bench_info = KNOWN_BENCHMARKS[benchmark_key]
    dataset_file = bench_info["file"]
    dataset_path = DATASETS_DIR / dataset_file
    if not dataset_path.exists():
        fallback_path = DATASETS_DIR / bench_info["fallback_file"]
        if fallback_path.exists():
            dataset_path = fallback_path
            warnings.append(f"File {dataset_file} non trovato, utilizzato fallback {bench_info['fallback_file']}.")
        else:
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
                errors.append(f"Il dataset {dataset_path.name} è vuoto o non è una lista JSON valida.")
            else:
                print(f" [OK] Dataset '{bench_info['name']}' valido ({len(dataset_samples)} campioni disponibili): {dataset_path}")
                if requested_samples > len(dataset_samples):
                    warnings.append(
                        f"Richiesti {requested_samples} campioni ma il dataset ne contiene {len(dataset_samples)}. "
                        f"I campioni verranno ciclati con modulo {len(dataset_samples)}."
                    )
                first_sample = dataset_samples[0]
                if benchmark_key == "mmlu":
                    _ = format_mmlu_prompt(first_sample)
                else:
                    _ = format_spider_prompt(first_sample)
                print(f" [OK] Formattazione prompt validata sul campione 1.")
        except Exception as e:
            errors.append(f"Errore nel parsing del dataset JSON {dataset_path}: {e}")

    # 4. Check vmstat e stress-ng
    vmstat_bin = shutil.which("vmstat")
    if vmstat_bin:
        print(f" [OK] Tool di monitoraggio sistema 'vmstat' trovato: {vmstat_bin}")
    else:
        warnings.append("Tool 'vmstat' non trovato nel PATH (richiesto su Linux/Jetson per le metriche di sistema).")

    if use_stressor:
        stress_ng_bin = shutil.which("stress-ng")
        if stress_ng_bin:
            print(f" [OK] Generatore di interferenza 'stress-ng' trovato: {stress_ng_bin}")
        else:
            warnings.append(
                "Tool 'stress-ng' non trovato nel PATH. Su Jetson installalo con: sudo apt-get install -y stress-ng"
            )

    # 5. Check device (CPU o GPU)
    print("\n" + "=" * 65)
    print(" 2. VERIFICA ALLOCAZIONE HARDWARE (Target: %s)" % device.upper())
    print("=" * 65)

    n_cpu_cores = os.cpu_count() or 4
    if device == "cpu":
        print(f" [OK] Allocazione selezionata: CPU-bound (-ngl 0).")
        print(f" [INFO] Core CPU disponibili per l'inferenza: {n_cpu_cores}")
    elif device == "gpu":
        print(f" [OK] Allocazione selezionata: GPU-offload completo (-ngl 99).")
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            try:
                smi_output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
                    text=True,
                    timeout=5
                ).strip()
                print(f" [OK] GPU NVIDIA rilevata: {smi_output}")
            except Exception:
                warnings.append("nvidia-smi presente ma non ha risposto correttamente.")
        else:
            warnings.append(
                "Tool 'nvidia-smi' non trovato nel PATH. Se sei su una scheda Jetson (Tegra), "
                "la GPU condivide la memoria unificata con il SoC."
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

    return llama_cli, model_path, dataset_path, len(dataset_samples), n_cpu_cores


# ==============================================================================
# ISTANZIAZIONE DELLA CONFIGURAZIONE
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
        print(f"[ERRORE] Stressor '{stressor_input}' non riconosciuto.")
        print(f"Scelte valide: {valid_opts}")
        sys.exit(1)

    s_class, s_method = STRESSOR_ALIASES[s_clean]
    has_size = KNOWN_STRESSORS[s_class]["has_size_factor"]
    return True, s_class, s_method, has_size


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
    # 1. Normalizzazione input modello
    m_clean = model_input.strip().lower()
    if m_clean in MODEL_ALIASES:
        model_key = MODEL_ALIASES[m_clean]
    elif m_clean in KNOWN_MODELS:
        model_key = m_clean
    else:
        print(f"[ERRORE] Modello '{model_input}' non riconosciuto.")
        print(f"Scelte valide: {list(MODEL_ALIASES.keys())}")
        sys.exit(1)

    # 2. Normalizzazione input device
    d_clean = device_input.strip().lower()
    if d_clean not in ["cpu", "gpu"]:
        print(f"[ERRORE] Device '{device_input}' non valido. Specificare 'cpu' o 'gpu'.")
        sys.exit(1)

    # 3. Normalizzazione input benchmark
    b_clean = benchmark_input.strip().lower()
    if b_clean not in KNOWN_BENCHMARKS:
        print(f"[ERRORE] Benchmark '{benchmark_input}' non valido. Specificare 'mmlu' o 'spider'.")
        sys.exit(1)

    # 4. Risoluzione Stressor
    use_stressor_resolved, s_class, s_method, has_size_factor = resolve_stressor(use_stressor, stressor_input)

    # 5. Determinazione numero di campioni (task)
    sample_count = samples if (samples is not None and samples > 0) else 40

    # 6. Esecuzione dei check di sistema
    llama_cli, model_path, dataset_path, dataset_total, n_cpu_cores = check_system_prerequisites(
        model_key=model_key,
        device=d_clean,
        benchmark_key=b_clean,
        use_stressor=use_stressor_resolved,
        requested_samples=sample_count
    )

    # 7. Determinazione parametri di esecuzione (greedy decoding con seed fisso per isolare varianza)
    n_gpu_layers = 99 if d_clean == "gpu" else 0
    n_threads = threads if threads is not None else n_cpu_cores
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
        repetitions=repetitions,
        transient_wait_sec=transient_wait_sec,
        reboot_after_run=reboot_after_run,
        ssh_target=ssh_target,
        ssh_dest=ssh_dest,
        ssh_port=ssh_port,
        ssh_key=ssh_key,
        seed=seed,
        llama_cli_path=str(llama_cli),
        temperature=temperature,
        max_tokens=max_tokens,
        n_threads=n_threads,
        example_cmd=example_cmd
    )

    with open(CONFIG_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    return config


# ==============================================================================
# CLI ENTRYPOINT
# ==============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Istanziazione del setup di esecuzione per la Traccia A1 (llama.cpp + stress-ng su Jetson Orin)."
    )

    # Argomenti posizionali opzionali (permette: python setup_run.py slm cpu mmlu)
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

    # Argomenti espliciti con flag
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

    # Nuovi argomenti per Stressor e Numero di Campioni
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

    # Se l'utente passa --stressor <nome> (diverso da none) senza --no-stressor, attiva automaticamente use_stressor
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


if __name__ == "__main__":
    main()
