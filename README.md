# EdgeAI Real-Time Interference Performance Study (Traccia A1)

Studio sperimentale e statistico sull'interferenza da contesa hardware in tempo reale durante l'inferenza di Small/Large Language Model (**SLM/LLM**) su piattaforme embedded edge (**NVIDIA Jetson AGX Orin**).

L'obiettivo è quantificare e modellare matematicamente come la saturazione di specifiche risorse hardware (mediante `stress-ng`) degrada la reattività temporale (Time To First Token), la latenza inter-token (TPOT), il throughput e il jitter, correlandoli con i contatori hardware del kernel Linux (`vmstat`) e i sensori termici.

---

## 1. Struttura del Piano Fattoriale degli Esperimenti (DoE)

In conformità ai requisiti sperimentali, i fattori del DoE sono rigorosamente:

1. **Numero di Worker ($N$)**:
   - Livelli: `25% N_tot`, `50% N_tot`, `75% N_tot`, `N_tot` (100% dei core fisici).
2. **Working-Set Size ($S$)**:
   - Livelli: `25% S_tot`, `50% S_tot`, `75% S_tot` (percentuale della RAM o dello spazio I/O).
   - **Presente ESCLUSIVAMENTE** per gli stressor 3D (`mem`, `io`). Per gli stressor 2D (`cpu`, `cache`, `sched`, `interrupt`), la size è assente (`N/A`).
3. **Durata Temporale ($D$)**:
   - **NON è un fattore variabile**: è fissa e costante al **tempo totale ($t_{\text{tot}}$)** della richiesta di inferenza, coprendo sia la fase di *prefill* (elaborazione del prompt) che la fase di *decode* (generazione autoregressiva dei token) in modo continuativo.

### Matrice Sperimentale
- **Stressor 2D** (`cpu`, `cache`, `sched`, `interrupt`): $4 \text{ livelli } (N) \times 1 \text{ durata } (t_{\text{tot}}) \times 3 \text{ ripetizioni} = \mathbf{12 \text{ esecuzioni}}$.
- **Stressor 3D** (`mem`, `io`): $4 \text{ livelli } (N) \times 3 \text{ livelli } (S) \times 1 \text{ durata } (t_{\text{tot}}) \times 3 \text{ ripetizioni} = \mathbf{36 \text{ esecuzioni}}$.
- **Baseline** (senza stressor): 1 configurazione $\times 3 \text{ ripetizioni}$ su $n_0 = 40$ task per la stima della varianza iniziale e il dimensionamento CLT.

---

## 2. Architettura della Repository

```
├── colab_test_workflow.ipynb       # Notebook per test e debug end-to-end (GPU Tesla T4 / Pinggy SSH)
├── setup/
│   ├── setup_system.py             # Download modelli GGUF, dataset (MMLU/Spider), build llama.cpp CUDA
│   ├── setup_run.py                # Configurazione parametri run e validazione rigida hardware GPU
│   ├── compute_sample_size.py      # Calcolo dimensione campionaria n via CLT (IC 95%, errore 5%)
│   └── run_experiments.py          # Orchestratore: DoE, stress-ng, vmstat 1, watchdog 300s, SSH sync, reboot
├── statistical_analysis/
│   ├── plot_llm_metrics.py         # Grafici di degradazione metriche LLM con riferimento Baseline
│   ├── plot_hardware_bottlenecks.py# Grafici colli di bottiglia hardware (CPU, Mem, I/O, Sys, Thermal)
│   ├── importance_significance.jsl # Script JMP: ANOVA (SS/SST), Shapiro-Wilk residui, Kruskal-Wallis
│   └── distributions.jsl           # Script JMP: Distribuzioni univariate, istogrammi e normal quantile plot
├── data_timeseries/                # Serie temporali vmstat a 1 Hz (.csv con CPU/GPU temp e throttling)
├── results/                        # Risultati DoE (doe_*.csv), metriche (tasks_*.csv) e risposte (responses_*.json)
├── figures/                        # Grafici ad alta risoluzione (300 DPI) generati per la relazione
└── master_experiment_summary.csv   # Riepilogo master di tutte le sessioni sperimentali
```

---

## 3. Flusso Operativo e Guida all'Uso

### A. Preparazione e Setup Ambiente
Scarica i modelli quantizzati GGUF (`Llama-3.2-1B-Instruct-Q4_K_M.gguf`, `Llama-3.1-8B-Instruct-Q4_K_M.gguf`), prepara i primi 40 campioni dei benchmark MMLU e Spider, e compila `llama.cpp` con supporto CUDA:
```bash
python setup/setup_system.py
```

### B. Raccolta Dati Baseline e Dimensionamento Campionario
1. Esegui la baseline (senza stressor):
   ```bash
   python setup/run_experiments.py --model slm --device gpu --benchmark mmlu --no-stressor --samples 40 --repetitions 1
   ```
2. Calcola la dimensione campionaria minima $n$ richiesta secondo il Teorema del Limite Centrale:
   ```bash
   python setup/compute_sample_size.py --model slm --device gpu --benchmark mmlu --confidence 0.95 --error 0.05
   ```

### C. Esecuzione del DoE degli Stressor
Esegui la matrice sperimentale completa dello stressor selezionato (es. `cpu`). Sulla scheda fisica Jetson AGX Orin, il sistema riavvia la macchina dopo ogni run e attende un transitorio termico/OS di 90 secondi prima del run successivo:
```bash
python setup/run_experiments.py \
  --model slm \
  --device gpu \
  --benchmark mmlu \
  --use-stressor \
  --stressor cpu \
  --samples 10 \
  --repetitions 3 \
  --transient-wait 90 \
  --ssh-target "utente@ip_host" \
  --ssh-port 22 \
  --ssh-dest "percorso/destinazione/host"
```
*(Su Google Colab per debug è possibile disabilitare il riavvio con `--no-reboot` e azzerare il transitorio con `--transient-wait 0`)*.

---

## 4. Analisi Statistica e Visualizzazione

### Generazione Grafici Python
1. **Metriche di Inferenza LLM**:
   ```bash
   python statistical_analysis/plot_llm_metrics.py
   ```
   Genera le curve di degradazione per TTFT, latenza per-token, throughput e jitter in funzione del carico worker ($N$) e working-set ($S$), tracciando la linea di riferimento tratteggiata della baseline senza stress.
2. **Colli di Bottiglia Hardware**:
   ```bash
   python statistical_analysis/plot_hardware_bottlenecks.py
   ```
   Elabora le serie temporali `vmstat` a 1 Hz, visualizzando l'impatto su saturazione CPU (%us, %sy, %id, %wa), memoria libera/swap, blocchi I/O, context switch, interrupt e temperature.

### Analisi Statistica Avanzata in JMP
- Apri `statistical_analysis/importance_significance.jsl` in JMP per calcolare automaticamente:
  - **Importanza dei Fattori**: percentuale di devianza spiegata ($SS_{\text{Effect}} / SS_{\text{Total}}$).
  - **Test di Normalità dei Residui**: test di Shapiro-Wilk sui residui del modello.
  - **Biforcazione Automatica**: valutazione del valore $p$ dell'ANOVA ($F$-test) se i residui sono normali ($p \ge 0.05$), o calcolo del test non parametrico di **Kruskal-Wallis** ($\chi^2$-test) se la normalità è violata ($p < 0.05$).
- Apri `statistical_analysis/distributions.jsl` per visualizzare istogrammi, quantile plot e test di normalità per ciascun parametro di sistema.
