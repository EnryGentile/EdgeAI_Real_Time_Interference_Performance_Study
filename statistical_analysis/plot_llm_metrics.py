import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configurazione stile
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs("figures/llm_metrics", exist_ok=True)

# Caricamento del dataset aggregato
df = pd.read_csv("doe_example.csv")

# Ordinamento esplicito dei livelli
n_order = ["25% N_tot", "50% N_tot", "75% N_tot", "N_tot"]
s_order = ["25% S_tot", "50% S_tot", "75% S_tot"]
d_order = ["t_prefill", "t_decode", "t_tot"]

df["num_workers"] = pd.Categorical(df["num_workers"], categories=n_order, ordered=True)
df["duration"] = pd.Categorical(df["duration"], categories=d_order, ordered=True)
if "size" in df.columns:
    df["size"] = pd.Categorical(df["size"], categories=s_order, ordered=True)

metrics = [
    ("ttft_ms", "Time To First Token (ms)"),
    ("per_token_lat_ms", "Per-Token Latency (ms)"),
    ("throughput_tok_s", "Throughput (tokens/s)"),
    ("jitter_ms", "Inter-Token Jitter (ms)")
]

# Generazione grafici per ciascuna metrica
for metric_col, metric_label in metrics:
    has_size = "size" in df.columns and df["size"].nunique() > 1
    
    g = sns.catplot(
        data=df,
        x="num_workers",
        y=metric_col,
        hue="duration",
        col="size" if has_size else None,
        kind="point",
        capsize=0.1,
        err_kws={'linewidth': 1.5},
        markers=["o", "s", "^"],
        linestyles=["-", "--", "-."],
        height=4.5,
        aspect=1.2
    )
    
    g.set_axis_labels("Carico Worker (N)", metric_label)
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle(f"Degradazione {metric_label} al variare dei Fattori DoE", fontsize=14, weight='bold')
    
    output_path = f"figures/llm_metrics/degradation_{metric_col}.png"
    g.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

print("Grafici metriche LLM generati con successo in figures/llm_metrics/")