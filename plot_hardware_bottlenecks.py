import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

#NOTA: DA TESTARE

os.makedirs("figures/hardware_bottlenecks", exist_ok=True)

stressor_class_with_s = ["mem", "io"]

# 1. Caricamento e aggregazione Time-Series (vmstat + sensori termici)
ts_files = glob.glob("data_timeseries/*_timeseries.csv")
ts_records = []

for filepath in ts_files:
    filename = os.path.basename(filepath)
    parts = filename.replace(".csv", "").split("_")
    # Formato atteso senza S: run_<stressor>_<N>_<D>_rep<rep>_timeseries.csv
    # Formato atteso con S:    run_<stressor>_<N>_<D>_<S>_rep<rep>_timeseries.csv
    
    stressor = parts[1]
    n_workers = parts[2]
    duration = parts[3]
    
    if stressor in stressor_class_with_s:
        size = parts[4]
        rep = parts[5]
    else:
        size = "N/A"
        rep = parts[4]
    
    sub_df = pd.read_csv(filepath)
    mean_vals = sub_df.mean(numeric_only=True).to_dict()
    mean_vals["stressor"] = stressor
    mean_vals["n_workers"] = n_workers
    mean_vals["duration"] = duration
    mean_vals["size"] = size
    mean_vals["repetition"] = rep
    ts_records.append(mean_vals)

df_hw = pd.DataFrame(ts_records)

n_order = ["N25", "N50", "N75", "N100"]
d_order = ["t_prefill", "t_decode", "t_tot"]
s_order = ["S25", "S50", "S75"]

df_hw["n_workers"] = pd.Categorical(df_hw["n_workers"], categories=n_order, ordered=True)
df_hw["duration"] = pd.Categorical(df_hw["duration"], categories=d_order, ordered=True)
if "size" in df_hw.columns:
    df_hw["size"] = pd.Categorical(df_hw["size"], categories=s_order + ["N/A"], ordered=True)

# Famiglie di parametri stile Figura 2.4 - 2.9
groups = [
    ("cpu", "Parametri CPU", [("us", "us (User)"), ("sy", "sy (Kernel)"), ("id", "id (Idle)"), ("wa", "wa (I/O Wait)")]),
    ("mem", "Parametri Memoria", [("swpd", "swpd"), ("free", "free"), ("buff", "buff"), ("cache", "cache")]),
    ("io", "Parametri I/O", [("bi", "bi (In)"), ("bo", "bo (Out)")]),
    ("procs", "Parametri Processi", [("r", "r (Ready)"), ("b", "b (Blocked)")]),
    ("system", "Parametri Sistema", [("cs", "cs (Context Switch)"), ("in", "in (Interrupts)")])
]

# 2. Plotting per famiglie di parametri con gestione 2D (N, D) e 3D (N, D, S)
stressors = df_hw["stressor"].unique() if not df_hw.empty else []

for s in stressors:
    is_3d = s in stressor_class_with_s
    sizes = s_order if is_3d else ["N/A"]
    
    for grp_key, grp_title, col_tuples in groups:
        ncols = len(sizes) if is_3d else 1
        nrows = len(d_order)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.5 * nrows), sharex=True, squeeze=False)
        
        for c_idx, sz in enumerate(sizes):
            for r_idx, dur in enumerate(d_order):
                ax = axes[r_idx, c_idx]
                
                cond = (df_hw["stressor"] == s) & (df_hw["duration"] == dur)
                if is_3d:
                    cond = cond & (df_hw["size"] == sz)
                    
                sub = df_hw[cond].groupby("n_workers", observed=False).mean(numeric_only=True).reindex(n_order)
                
                for col, label in col_tuples:
                    if col in sub.columns and sub[col].notna().any():
                        ax.plot(sub.index, sub[col], marker='o', label=label)
                
                header = f"D: {dur}" + (f" | S: {sz}" if is_3d else "")
                ax.set_title(header, fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.5)
                
                if r_idx == nrows - 1:
                    ax.set_xlabel("Livelli Carico (N)")
                if c_idx == 0:
                    ax.set_ylabel("Media Valore")
                if r_idx == 0 and c_idx == ncols - 1:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        fig.suptitle(f"{grp_title} - Stressor: {s.upper()}", fontsize=13, weight='bold')
        fig.tight_layout()
        fig.savefig(f"figures/hardware_bottlenecks/{s}_{grp_key}_bottleneck.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

# 3. Microarchitettura e Valori Termici (Master Summary CSV)
if os.path.exists("master_experiment_summary.csv"):
    df_master = pd.read_csv("master_experiment_summary.csv")
    
    # Uniformazione nomi colonne se necessario
    if "N_workers" in df_master.columns:
        df_master.rename(columns={"N_workers": "num_workers"}, inplace=True)
    if "duration_mode" in df_master.columns:
        df_master.rename(columns={"duration_mode": "duration"}, inplace=True)
    if "S_size" in df_master.columns:
        df_master.rename(columns={"S_size": "size"}, inplace=True)
        
    df_master["num_workers"] = pd.Categorical(df_master["num_workers"], categories=n_order, ordered=True)
    df_master["duration"] = pd.Categorical(df_master["duration"], categories=d_order, ordered=True)
    
    for sc in df_master["stressor_class"].unique():
        sc_clean = str(sc).lower()
        is_3d = sc_clean in stressor_class_with_s
        sizes = s_order if is_3d else ["N/A"]
        
        ncols = len(sizes) if is_3d else 1
        nrows = len(d_order)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.5 * ncols, 3.5 * nrows), sharex=True, squeeze=False)
        
        for c_idx, sz in enumerate(sizes):
            for r_idx, dur in enumerate(d_order):
                ax1 = axes[r_idx, c_idx]
                ax2 = ax1.twinx()
                
                cond = (df_master["stressor_class"] == sc) & (df_master["duration"] == dur)
                if is_3d and "size" in df_master.columns:
                    cond = cond & (df_master["size"] == sz)
                    
                sub = df_master[cond].groupby("num_workers", observed=False).mean(numeric_only=True).reindex(n_order)
                
                p1, p2, p3 = None, None, None
                if "perf_cache_misses" in sub.columns and sub["perf_cache_misses"].notna().any():
                    p1 = ax1.plot(sub.index, sub["perf_cache_misses"], 'r-o', label="Cache Misses")
                if "perf_branch_misses" in sub.columns and sub["perf_branch_misses"].notna().any():
                    p2 = ax1.plot(sub.index, sub["perf_branch_misses"], 'm--^', label="Branch Misses")
                if "avg_cpu_temp_c" in sub.columns and sub["avg_cpu_temp_c"].notna().any():
                    p3 = ax2.plot(sub.index, sub["avg_cpu_temp_c"], 'b-s', label="CPU Temp (°C)")
                
                header = f"D: {dur}" + (f" | S: {sz}" if is_3d else "")
                ax1.set_title(header, fontsize=10)
                ax1.grid(True, linestyle='--', alpha=0.5)
                
                if r_idx == nrows - 1:
                    ax1.set_xlabel("Livelli Carico (N)")
                if c_idx == 0:
                    ax1.set_ylabel("Perf Misses")
                if c_idx == ncols - 1:
                    ax2.set_ylabel("Temp (°C)", color='b')
                else:
                    ax2.set_ylabel("")
                
                if r_idx == 0 and c_idx == ncols - 1:
                    plots = [p[0] for p in [p1, p2, p3] if p is not None]
                    labels = [p.get_label() for p in plots]
                    ax1.legend(plots, labels, bbox_to_anchor=(1.2, 1), loc='upper left')
        
        fig.suptitle(f"Microarchitettura e Termica - Stressor: {str(sc).upper()}", fontsize=13, weight='bold')
        fig.tight_layout()
        fig.savefig(f"figures/hardware_bottlenecks/{sc_clean}_perf_thermal.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

print("Tutti i grafici hardware (2D e 3D) generati con successo in figures/hardware_bottlenecks/")