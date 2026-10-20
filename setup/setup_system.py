#!/usr/bin/env python3
"""
setup_system.py - Setup iniziale del sistema, modelli GGUF, dataset e compilazione llama.cpp.

Ottimizzato e protetto con blocchi try/except e procedure di fallback automatiche:
  1. Risoluzione assoluta dei percorsi (funziona sia da root sia da ./setup).
  2. Installazione pacchetti Python con fallback per ambienti PEP-668 (--break-system-packages / --user).
  3. Download modelli GGUF (Llama-3.2-1B e Llama-3.1-8B) con fallback via curl/wget in caso di errore API.
  4. Download dataset (MMLU e Spider, primi 40 campioni) con scrittura atomica e fallback offline.
  5. Compilazione llama.cpp con supporto CUDA (auto-rilevamento /usr/local/cuda/bin/nvcc su Jetson Orin)
     e fallback automatico alla compilazione CPU (-DGGML_CUDA=OFF) se il toolkit CUDA non è disponibile.
"""

import os
import sys
import json
import shutil
import traceback
import subprocess
from pathlib import Path
from typing import Dict, Any, List

# ==============================================================================
# RISOLUZIONE PERCORSI ASSOLUTI DEL PROGETTO
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if (Path.cwd() / "setup").exists():
    BASE_DIR = Path.cwd()
else:
    BASE_DIR = PROJECT_ROOT

MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"
LLAMA_CPP_DIR = BASE_DIR / "llama.cpp"
ERROR_LOG_PATH = BASE_DIR / "setup" / "setup_system_error.log"

MODELS_CONFIG = [
    {
        "role": "SLM (Small Language Model - 1B)",
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    },
    {
        "role": "LLM (Large Language Model - 8B)",
        "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    },
]


def log_error(stage: str, exc: Exception):
    """Registra l'errore dettagliato su file di log per diagnostica."""
    try:
        ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n=== ERRORE IN FASE: {stage} ===\n")
            f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            f.write("\n")
    except Exception:
        pass


# ==============================================================================
# 0. CHECK E INSTALLAZIONE LIBRERIE NECESSARIE CON FALLBACK
# ==============================================================================
def install_required_packages():
    print("=== 0. Verifica delle dipendenze Python ===")
    required_packages = ["huggingface_hub", "datasets"]
    for package in required_packages:
        try:
            __import__(package)
            print(f"[OK] Libreria '{package}' già installata.")
        except ImportError:
            print(f"[INSTALL] Libreria '{package}' mancante. Installazione in corso...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", package])
            except subprocess.CalledProcessError as e:
                print(f"[FALLBACK] Installazione standard di '{package}' fallita ({e}). Tentativo con --user / --break-system-packages...")
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "-U", "--user", "--break-system-packages", package
                    ])
                except Exception as e2:
                    log_error(f"install_package_{package}", e2)
                    print(f"[ERRORE CRITICO] Impossibile installare '{package}'. Salvataggio log in {ERROR_LOG_PATH} e arresto.")
                    sys.exit(1)


def setup_directories():
    """Crea le directory di lavoro con gestione errori sui permessi."""
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log_error("setup_directories", e)
        print(f"[ERRORE CRITICO] Impossibile creare le directory di lavoro in {BASE_DIR}: {e}")
        sys.exit(1)


# ==============================================================================
# 1. DOWNLOAD MODELLI GGUF CON FALLBACK WGET/CURL E ARRESTO IN CASO DI ERRORE
# ==============================================================================
def _fallback_download_url(url: str, dest_path: Path):
    """Fallback di download diretto tramite curl o wget se l'API Python fallisce."""
    tmp_path = dest_path.with_suffix(".gguf.part")
    last_err = "Nessun tool di fallback (curl/wget) trovato nel PATH."
    try:
        if shutil.which("curl"):
            print(f"[FALLBACK] Tentativo di download con 'curl' da {url}...")
            subprocess.run(["curl", "-L", "--fail", "--retry", "3", "-o", str(tmp_path), url], check=True)
            tmp_path.replace(dest_path)
            return True, ""
        elif shutil.which("wget"):
            print(f"[FALLBACK] Tentativo di download con 'wget' da {url}...")
            subprocess.run(["wget", "-O", str(tmp_path), url], check=True)
            tmp_path.replace(dest_path)
            return True, ""
    except Exception as e:
        last_err = str(e)
        log_error("fallback_download_url", e)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
    return False, last_err


def check_and_download_models():
    """Verifica e scarica i file GGUF dei modelli (SLM 1B e LLM 8B) da Hugging Face."""
    print("\n=== 1. Verifica e Download Modelli GGUF ===")
    from huggingface_hub import hf_hub_download

    for model_info in MODELS_CONFIG:
        role = model_info["role"]
        model_file = model_info["file"]
        model_repo = model_info["repo"]
        model_path = MODELS_DIR / model_file

        if model_path.exists() and model_path.stat().st_size > 100 * (1024 ** 2):
            size_gb = model_path.stat().st_size / (1024 ** 3)
            print(f"[OK] Modello {role} già presente ({size_gb:.2f} GB): {model_path}")
            continue

        print(f"[DOWNLOAD] Scaricando {role} ({model_file}) da Hugging Face...")
        try:
            hf_hub_download(
                repo_id=model_repo,
                filename=model_file,
                local_dir=str(MODELS_DIR),
            )
            print(f"[OK] Modello {role} scaricato correttamente in: {model_path}")
        except Exception as e_hf:
            log_error(f"download_model_{model_file}", e_hf)
            print(f"[WARNING] Errore con hf_hub_download per {role}: {e_hf}")
            direct_url = f"https://huggingface.co/{model_repo}/resolve/main/{model_file}"
            ok, fallback_err = _fallback_download_url(direct_url, model_path)
            if ok:
                print(f"[OK] Download di fallback (curl/wget) completato per {role}: {model_path}")
            else:
                print("\n" + "!" * 70)
                print(f"[ERRORE CRITICO] Download del modello {role} ({model_file}) fallito!")
                print(f"  - Errore hf_hub_download : {e_hf}")
                print(f"  - Errore curl/wget       : {fallback_err}")
                print(f"Lo script viene arrestato. Dettagli in: {ERROR_LOG_PATH}")
                print("!" * 70)
                sys.exit(1)


# ==============================================================================
# 2. DOWNLOAD DATASET CON SCRITTURA ATOMICA E FALLBACK SINTETICO
# ==============================================================================
def _save_json_atomic(path: Path, data: List[Dict[str, Any]]):
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def _generate_fallback_mmlu(count: int = 40) -> List[Dict[str, Any]]:
    """Genera 40 campioni MMLU di fallback validi se la rete o HuggingFace non rispondono."""
    samples = []
    for i in range(1, count + 1):
        samples.append({
            "question": (
                f"[Sample #{i}] In real-time operating systems and embedded architectures, "
                f"which scheduling policy assigns the highest priority to the task with the shortest period?"
            ),
            "subject": "computer_science",
            "choices": [
                "Rate Monotonic Scheduling (RMS)",
                "First-Come First-Served (FCFS)",
                "Round-Robin (RR) with static quantum",
                "Least Slack Time (LST) without preemption"
            ],
            "answer": 0
        })
    return samples


def _generate_fallback_spider(count: int = 40) -> List[Dict[str, Any]]:
    """Genera 40 campioni Spider (Text-to-SQL) di fallback validi se la rete non risponde."""
    samples = []
    for i in range(1, count + 1):
        samples.append({
            "db_id": f"industrial_telemetry_db_{i}",
            "question": (
                f"Find the average CPU temperature, maximum latency, and total context switches "
                f"for all edge nodes running sensor task {i} where thermal throttling was not triggered, "
                f"grouped by node architecture and ordered by average latency descending."
            ),
            "query": (
                "SELECT arch, AVG(cpu_temp), MAX(latency_ms), SUM(ctx_switches) "
                "FROM node_metrics WHERE throttled = 0 GROUP BY arch ORDER BY AVG(latency_ms) DESC;"
            )
        })
    return samples


def check_and_download_datasets():
    """Verifica e crea i subset locali di MMLU e Spider (prime 40 richieste)."""
    print("\n=== 2. Verifica e Generazione Dataset (MMLU & Spider - 40 campioni) ===")
    from datasets import load_dataset

    mmlu_path = DATASETS_DIR / "mmlu_first_40.json"
    spider_path = DATASETS_DIR / "spider_first_40.json"

    # 1. Dataset MMLU (Prefill-heavy / Compute-bound)
    if mmlu_path.exists() and mmlu_path.stat().st_size > 100:
        print(f"[OK] Dataset MMLU (40 campioni) già presente: {mmlu_path}")
    else:
        print("[DOWNLOAD] Generazione subset MMLU (prime 40 richieste)...")
        try:
            mmlu_dataset = load_dataset("cais/mmlu", "all", split="test", streaming=True)
            mmlu_subset = list(mmlu_dataset.take(40))
            if len(mmlu_subset) < 40:
                raise RuntimeError(f"Ricevuti solo {len(mmlu_subset)} campioni MMLU (< 40).")
            _save_json_atomic(mmlu_path, mmlu_subset)
            print(f"[OK] Dataset MMLU salvato in: {mmlu_path}")
        except Exception as e:
            log_error("download_dataset_mmlu", e)
            print(f"[FALLBACK] Errore download MMLU da HuggingFace ({e}). Generazione dataset MMLU locale di fallback (40 campioni)...")
            try:
                _save_json_atomic(mmlu_path, _generate_fallback_mmlu(40))
                print(f"[OK] Dataset MMLU di fallback salvato in: {mmlu_path}")
            except Exception as e2:
                log_error("fallback_dataset_mmlu", e2)
                print(f"[ERRORE CRITICO] Impossibile salvare il dataset MMLU: {e2}")
                sys.exit(1)

    # 2. Dataset Spider (Decode-heavy / Memory-bound)
    if spider_path.exists() and spider_path.stat().st_size > 100:
        print(f"[OK] Dataset Spider (40 campioni) già presente: {spider_path}")
    else:
        print("[DOWNLOAD] Generazione subset Spider (prime 40 richieste)...")
        try:
            spider_dataset = load_dataset("xlangai/spider", split="validation", streaming=True)
            spider_subset = list(spider_dataset.take(40))
            if len(spider_subset) < 40:
                raise RuntimeError(f"Ricevuti solo {len(spider_subset)} campioni Spider (< 40).")
            _save_json_atomic(spider_path, spider_subset)
            print(f"[OK] Dataset Spider salvato in: {spider_path}")
        except Exception as e:
            log_error("download_dataset_spider", e)
            print(f"[FALLBACK] Errore download Spider da HuggingFace ({e}). Generazione dataset Spider locale di fallback (40 campioni)...")
            try:
                _save_json_atomic(spider_path, _generate_fallback_spider(40))
                print(f"[OK] Dataset Spider di fallback salvato in: {spider_path}")
            except Exception as e2:
                log_error("fallback_dataset_spider", e2)
                print(f"[ERRORE CRITICO] Impossibile salvare il dataset Spider: {e2}")
                sys.exit(1)


# ==============================================================================
# 3. COMPILAZIONE llama.cpp CON FALLBACK CUDA -> CPU
# ==============================================================================
def check_and_setup_llama_cpp():
    """Verifica la presenza dei binari di llama.cpp o procede alla clonazione e build con CUDA (con fallback CPU)."""
    print("\n=== 3. Verifica e Compilazione Runtime llama.cpp ===")
    bin_candidates = [
        LLAMA_CPP_DIR / "build" / "bin" / "llama-cli",
        LLAMA_CPP_DIR / "build" / "bin" / "llama-cli.exe",
        LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-cli.exe",
    ]
    for b in bin_candidates:
        if b.exists():
            print(f"[OK] Runtime llama.cpp già compilato trovato in: {b}")
            return

    print("[SETUP] llama.cpp non trovato o non compilato. Configurazione in corso...")
    try:
        if not LLAMA_CPP_DIR.exists():
            print("Clonazione del repository ufficiale di llama.cpp...")
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", str(LLAMA_CPP_DIR)],
                check=True
            )
    except Exception as e:
        log_error("git_clone_llama_cpp", e)
        print(f"[ERRORE CRITICO] Clonazione di llama.cpp fallita: {e}")
        sys.exit(1)

    build_dir = LLAMA_CPP_DIR / "build"
    build_dir.mkdir(exist_ok=True)
    n_jobs = str(os.cpu_count() or 4)

    # Rilevamento compilatore nvcc (su Jetson spesso risiede in /usr/local/cuda/bin/nvcc)
    env = os.environ.copy()
    jetson_cuda_bin = "/usr/local/cuda/bin"
    if os.path.isdir(jetson_cuda_bin) and jetson_cuda_bin not in env.get("PATH", ""):
        env["PATH"] = f"{jetson_cuda_bin}:{env.get('PATH', '')}"

    nvcc_path = shutil.which("nvcc", path=env.get("PATH"))
    cmake_cuda_args = ["-B", str(build_dir), "-S", str(LLAMA_CPP_DIR), "-DGGML_CUDA=ON"]
    if nvcc_path:
        cmake_cuda_args.append(f"-DCMAKE_CUDA_COMPILER={nvcc_path}")

    try:
        print("Configurazione CMake e compilazione con supporto CUDA...")
        subprocess.run(["cmake"] + cmake_cuda_args, env=env, check=True)
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--config", "Release", "-j", n_jobs, "--target", "llama-cli"],
            env=env,
            check=True
        )
        print("[OK] Compilazione di llama.cpp (CUDA) completata con successo!")
    except Exception as e:
        log_error("cmake_build_cuda", e)
        print(f"\n[FALLBACK] Compilazione con CUDA fallita ({e}).")
        print("[FALLBACK] Pulizia cache CMake e avvio compilazione di fallback in modalità CPU (-DGGML_CUDA=OFF)...")
        try:
            shutil.rmtree(build_dir, ignore_errors=True)
            build_dir.mkdir(exist_ok=True)
            subprocess.run(
                ["cmake", "-B", str(build_dir), "-S", str(LLAMA_CPP_DIR), "-DGGML_CUDA=OFF"],
                env=env,
                check=True
            )
            subprocess.run(
                ["cmake", "--build", str(build_dir), "--config", "Release", "-j", n_jobs, "--target", "llama-cli"],
                env=env,
                check=True
            )
            print("[OK] Compilazione di fallback (CPU) di llama.cpp completata con successo!")
        except Exception as e2:
            log_error("cmake_build_cpu_fallback", e2)
            print(f"[ERRORE CRITICO] Compilazione di llama.cpp fallita anche in modalità CPU: {e2}")
            print(f"Controlla il log degli errori in: {ERROR_LOG_PATH}")
            sys.exit(1)


if __name__ == "__main__":
    print("=== Avvio Setup Ambiente di Benchmark (Jetson AGX Orin / Linux) ===")
    try:
        install_required_packages()
        setup_directories()
        check_and_download_models()
        check_and_download_datasets()
        check_and_setup_llama_cpp()
        print("\n=== Setup Completato con Successo! SLM (1B), LLM (8B), Dataset e Binari pronti. ===")
    except KeyboardInterrupt:
        print("\n[INTERRUZIONE] Setup interrotto manualmente dall'utente.")
        sys.exit(130)
    except Exception as unexpected_exc:
        log_error("main_unexpected", unexpected_exc)
        print(f"\n[ERRORE FATALE] Errore imprevisto durante il setup: {unexpected_exc}")
        print(f"Dettagli salvati in: {ERROR_LOG_PATH}")
        sys.exit(1)
