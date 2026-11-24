#!/usr/bin/env python3
"""
evaluate_quality.py - Metriche qualitative sulle risposte del modello (baseline e stressor).

Legge i file results/responses_<exp_id>.json prodotti da run_experiments.py e calcola, per task:

  MMLU
    - mmlu_pred            lettera data esplicitamente come risposta: la risposta inizia con la lettera
                           ("B", "B.", "D\\n\\nExplanation...") oppure contiene "answer is B" / "Answer: B";
                           vuota se il modello ragiona senza arrivare a una lettera
    - correct              1 se la lettera coincide con la risposta attesa (0 se non c'è una lettera)
    - format_ok            1 se la risposta è solo la lettera (es. "B" o "B.")
  Spider (text-to-SQL)
    - pred_sql             query estratta dalla risposta (blocco ```sql``` o primo SELECT/WITH)
    - sql_valid            1 se la query viene eseguita senza errori sul database SQLite
    - exec_match           1 se il risultato coincide con quello della query attesa
                           (execution accuracy; ordine delle righe considerato solo se la query
                           attesa contiene ORDER BY)
    - exact_match          1 se la query coincide con quella attesa a meno di spazi/maiuscole/';'
  Comuni
    - truncated            1 se la generazione si è fermata per limite di token (finish_reason=length)
    - bert_f1_gt           BERTScore F1 tra risposta e risposta attesa (solo Spider: query vs query)
    - identical_to_base    1 se la risposta è identica a quella della baseline sullo stesso prompt
    - bert_f1_vs_base      BERTScore F1 tra la risposta e quella della baseline sullo stesso prompt:
                           misura quanto l'interferenza cambia il contenuto della risposta
                           (con temperatura 0 e seed fisso, senza interferenza ci si aspetta 1.0)

Le colonne BERTScore richiedono:  pip install bert-score
Il modello predefinito è multilingue (risposte in inglese/italiano e SQL): --bert-model per cambiarlo,
--no-bert per saltare il calcolo.

Adattamenti per questa versione del framework (llama-cli):
  - i file responses_*.json sono cercati ricorsivamente (es. resultAGX/mmlu/{baseline,stressor}/<modello>/<device>/)
  - modello, device, benchmark, stressor, N e S sono ricavati dal nome del file se mancano nel master
  - 'truncated' è una stima: output_tokens >= 80% di --max-tokens-mmlu / --max-tokens-spider (llama-cli non
    riporta finish_reason e output_tokens conta i blocchi letti da stdout, non i token esatti)
  - la baseline di riferimento per identical_to_base / bert_f1_vs_base è la prima ripetizione della baseline
  - viene salvato anche task_index, per poter separare il primo task di ogni run

Output:
  <out>/quality_tasks.csv     una riga per task
  <out>/quality_summary.csv   una riga per esperimento (medie delle colonne sopra)

Esempio:
  python statistical_analysis/evaluate_quality.py --results resultAGX/mmlu --out figures/quality_reports/mmlu
"""

import argparse
import csv
import json
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BERT_MODEL = "bert-base-multilingual-cased"


# ------------------------------------------------------------------------------
# MMLU
# ------------------------------------------------------------------------------
# Lettera data come risposta: in apertura (seguita da . ) : o fine riga), oppure dopo "answer is" / "Answer:"
# (copre anche le risposte che ripetono il prompt e poi rispondono dopo "...correct answer:")
RE_MMLU_LEAD = re.compile(r"^\s*\(?([ABCD])\)?\s*(?:[.):]|\n|$)")
RE_MMLU_ANSWER = re.compile(r"\b(?:answer|risposta)\s*(?:is|è|:)\s*\(?\**([ABCD])\b", re.I)


def mmlu_extract(response: str) -> str:
    """Lettera data esplicitamente come risposta; stringa vuota se il modello non arriva a una lettera."""
    text = response or ""
    m = RE_MMLU_LEAD.match(text) or RE_MMLU_ANSWER.search(text)
    return m.group(1).upper() if m else ""


def mmlu_first_letter(response: str) -> str:
    """Regola precedente (solo per confronto): prima A/B/C/D isolata nel testo. Conta anche l'articolo
    inglese "A" e le opzioni ripetute dal prompt."""
    letters = re.findall(r"\b([ABCD])\b", response or "")
    return letters[0] if letters else ""


def mmlu_format_ok(response: str) -> bool:
    return bool(re.fullmatch(r"\s*[ABCD][.)]?\s*", response or ""))


# ------------------------------------------------------------------------------
# SPIDER
# ------------------------------------------------------------------------------
def spider_extract_sql(response: str) -> str:
    """Estrae la query: prima dal blocco ```sql```, altrimenti dal primo SELECT/WITH fino a ';' o fine."""
    text = response or ""
    m = re.search(r"```(?:sql|sqlite)?\s*(.*?)```", text, re.S | re.I)
    if m:
        text = m.group(1)
    else:
        m = re.search(r"\b(SELECT|WITH)\b", text, re.I)
        if not m:
            return ""
        text = text[m.start():]
        # una query senza blocco di codice finisce al primo ';' o alla prima riga vuota
        text = re.split(r";|\n\s*\n", text, maxsplit=1)[0]
    return text.strip().rstrip(";").strip()


def normalize_sql(q: str) -> str:
    q = (q or "").strip().rstrip(";").lower().replace('"', "'")
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"\s*([(),=<>])\s*", r"\1", q)
    return q


def run_query(db_file: Path, query: str, timeout_s: float = 10.0) -> Tuple[bool, Any]:
    """Esegue la query in sola lettura con un limite di tempo. Restituisce (ok, righe | errore)."""
    if not query:
        return False, "query vuota"
    conn = sqlite3.connect(f"file:{db_file.as_posix()}?mode=ro", uri=True)
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    try:
        return True, conn.execute(query).fetchall()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        conn.close()


def _norm_row(row) -> tuple:
    out = []
    for v in row:
        if isinstance(v, float):
            v = round(v, 6)
        elif isinstance(v, str):
            v = v.strip()
        out.append(v)
    return tuple(out)


def results_match(pred_rows, gold_rows, ordered: bool) -> bool:
    p = [_norm_row(r) for r in pred_rows]
    g = [_norm_row(r) for r in gold_rows]
    return p == g if ordered else Counter(p) == Counter(g)


# ------------------------------------------------------------------------------
# BERTSCORE
# ------------------------------------------------------------------------------
class BertScorer:
    def __init__(self, model_type: str):
        try:
            from bert_score import BERTScorer  # type: ignore
        except ImportError:
            raise SystemExit(
                "[ERRORE] Pacchetto 'bert-score' non installato: pip install bert-score\n"
                "         (oppure usa --no-bert per calcolare solo le altre metriche)"
            )
        print(f"[BERTScore] Caricamento modello '{model_type}'...")
        self.scorer = BERTScorer(model_type=model_type, lang=None)

    def f1(self, cands: List[str], refs: List[str]) -> List[Optional[float]]:
        """F1 per coppia; None dove una delle due stringhe è vuota.
        Le coppie ripetute vengono calcolate una sola volta (con decoding greedy le risposte si ripetono)."""
        out: List[Optional[float]] = [None] * len(cands)
        unique = list(dict.fromkeys((c, r) for c, r in zip(cands, refs) if c.strip() and r.strip()))
        scores: Dict[Tuple[str, str], float] = {}
        if unique:
            print(f"[BERTScore] {len(unique)} coppie distinte su {len(cands)}")
            _, _, f = self.scorer.score([c for c, _ in unique], [r for _, r in unique], batch_size=32)
            scores = {p: round(float(v), 4) for p, v in zip(unique, f.tolist())}
        for i, (c, r) in enumerate(zip(cands, refs)):
            if (c, r) in scores:
                out[i] = scores[(c, r)]
        return out


# ------------------------------------------------------------------------------
# LETTURA ESPERIMENTI
# ------------------------------------------------------------------------------
def load_experiment_index(master_csv: Optional[Path]) -> Dict[str, Dict[str, str]]:
    """exp_id -> {model_id, device, benchmark, stressor_class, N_workers, S_size} dal riepilogo master."""
    index: Dict[str, Dict[str, str]] = {}
    if master_csv and master_csv.exists():
        with open(master_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                index[r["exp_id"]] = r
    return index


RE_BASE = re.compile(r"^baseline_([a-z0-9\-]+)_(cpu|gpu)_(mmlu|spider)_rep(\d+)$", re.I)
RE_RUN = re.compile(r"^run_([a-z0-9\-]+)_(cpu|gpu)_(mmlu|spider)_([a-z]+)_(N\d+)_(t_[a-z]+)(?:_(S\d+))?_rep(\d+)$", re.I)


def info_from_exp_id(exp_id: str) -> Dict[str, str]:
    """Ricava modello, device, benchmark, stressor, N, S e ripetizione dal nome dell'esperimento."""
    m = RE_BASE.match(exp_id)
    if m:
        return {"model_id": m.group(1), "device": m.group(2), "benchmark": m.group(3), "stressor_class": "none",
                "N_workers": "N0", "S_size": "N/A", "repetition": m.group(4)}
    m = RE_RUN.match(exp_id)
    if m:
        return {"model_id": m.group(1), "device": m.group(2), "benchmark": m.group(3), "stressor_class": m.group(4),
                "N_workers": m.group(5), "S_size": m.group(7) or "N/A", "repetition": m.group(8)}
    return {}


def baseline_key(info: Dict[str, str]) -> Tuple[str, str, str]:
    return (info.get("model_id", "").lower(), info.get("device", "").lower(), info.get("benchmark", "").lower())


def main():
    ap = argparse.ArgumentParser(description="Metriche qualitative (MMLU, Spider, BERTScore) sulle risposte del modello.")
    ap.add_argument("--results", type=Path, default=PROJECT_ROOT / "results", help="Cartella con responses_*.json")
    ap.add_argument("--master", type=Path, default=None,
                    help="master_experiment_summary.csv (default: <results>/../master_experiment_summary.csv)")
    ap.add_argument("--db-dir", type=Path, default=PROJECT_ROOT / "datasets" / "spider_db",
                    help="Cartella con i database Spider (<db_id>/<db_id>.sqlite)")
    ap.add_argument("--spider-dataset", type=Path, default=PROJECT_ROOT / "datasets" / "spider_first_40.json",
                    help="Dataset Spider locale: serve a ricavare il db_id di ogni campione")
    ap.add_argument("--out", type=Path, default=None, help="Cartella di output (default: --results)")
    ap.add_argument("--bert-model", default=DEFAULT_BERT_MODEL, help=f"Modello BERTScore (default: {DEFAULT_BERT_MODEL})")
    ap.add_argument("--no-bert", action="store_true", help="Non calcola le metriche BERTScore")
    ap.add_argument("--max-tokens-mmlu", type=int, default=64, help="Limite di token usato per MMLU (stima di 'truncated')")
    ap.add_argument("--max-tokens-spider", type=int, default=256, help="Limite di token usato per Spider (stima di 'truncated')")
    args = ap.parse_args()

    master = args.master or (args.results.parent / "master_experiment_summary.csv")
    out_dir = args.out or args.results
    exp_index = load_experiment_index(master)
    token_limit = {"mmlu": args.max_tokens_mmlu, "spider": args.max_tokens_spider}

    # db_id di ogni campione Spider (le risposte salvano il sample_id ma non il database)
    spider_db_by_question: Dict[str, str] = {}
    if args.spider_dataset.exists():
        for s in json.loads(args.spider_dataset.read_text(encoding="utf-8")):
            spider_db_by_question[s.get("question", "").strip()] = s.get("db_id", "")

    # Ricerca ricorsiva: in questa versione i risultati sono divisi in sottocartelle
    files = sorted(p for p in args.results.rglob("responses_*.json") if "fallback_partial" not in p.name)
    if not files:
        raise SystemExit(f"[ERRORE] Nessun file responses_*.json in {args.results}")

    rows: List[Dict[str, Any]] = []
    gold_cache: Dict[Tuple[str, str], Tuple[bool, Any]] = {}
    for path in files:
        exp_id = path.stem[len("responses_"):]
        info = dict(info_from_exp_id(exp_id))
        info.update({k: v for k, v in exp_index.get(exp_id, {}).items() if v})
        for t in json.loads(path.read_text(encoding="utf-8")):
            bench = str(t.get("benchmark", info.get("benchmark", ""))).lower()
            resp = t.get("model_response", "") or ""
            gt = t.get("ground_truth", "") or ""
            if "finish_reason" in t:
                truncated = int(t.get("finish_reason") == "length")
            else:
                # output_tokens qui conta i blocchi letti da stdout, non i token esatti: si considera
                # troncata una risposta che arriva ad almeno l'80% del limite (stima)
                truncated = int(int(t.get("output_tokens") or 0) >= 0.8 * token_limit.get(bench, 10 ** 9))
            row: Dict[str, Any] = {
                "exp_id": exp_id,
                "model_id": info.get("model_id", ""),
                "device": info.get("device", ""),
                "benchmark": bench,
                "stressor_class": info.get("stressor_class", ""),
                "N_workers": info.get("N_workers", ""),
                "S_size": info.get("S_size", ""),
                "repetition": info.get("repetition", ""),
                "task_index": t.get("task_index", ""),
                "sample_id": t.get("sample_id"),
                "status": t.get("status", "ok"),
                "truncated": truncated,
                "model_response": resp,
                "ground_truth": gt,
            }
            if row["status"] != "ok":
                rows.append(row)
                continue

            if bench == "mmlu":
                pred = mmlu_extract(resp)
                row.update({"mmlu_pred": pred, "correct": int(pred == gt.strip().upper()),
                            "format_ok": int(mmlu_format_ok(resp))})
            elif bench == "spider":
                prompt = t.get("prompt", "")
                m = re.search(r"(?:Richiesta|Request):\s*(.*?)\n", prompt + "\n")
                question = m.group(1).strip() if m else ""
                db_id = spider_db_by_question.get(question, "")
                if not db_id:
                    m = re.search(r"database (?:SQLite )?'([^']+)'", prompt, re.I)
                    db_id = m.group(1) if m else ""
                pred_sql = spider_extract_sql(resp)
                row.update({"db_id": db_id, "pred_sql": pred_sql,
                            "exact_match": int(bool(pred_sql) and normalize_sql(pred_sql) == normalize_sql(gt))})
                db_file = args.db_dir / db_id / f"{db_id}.sqlite"
                if db_id and db_file.exists():
                    if (db_id, gt) not in gold_cache:
                        gold_cache[(db_id, gt)] = run_query(db_file, gt)
                    gold_ok, gold_rows = gold_cache[(db_id, gt)]
                    ok, pred_rows = run_query(db_file, pred_sql)
                    row["sql_valid"] = int(ok)
                    row["exec_match"] = int(ok and gold_ok and results_match(
                        pred_rows, gold_rows, ordered="order by" in gt.lower()))
                    if not ok:
                        row["sql_error"] = str(pred_rows)[:200]
            rows.append(row)

    # Confronto con la baseline sullo stesso prompt (stesso modello, device e benchmark)
    # (riferimento: la ripetizione di baseline con il numero più basso, per ogni prompt)
    baseline_resp: Dict[Tuple[Tuple[str, str, str], Any], str] = {}
    base_rows = [r for r in rows if r["exp_id"].startswith("baseline_") and r["status"] == "ok"]
    for r in sorted(base_rows, key=lambda r: -int(r.get("repetition") or 0)):
        baseline_resp[(baseline_key(r), r["sample_id"])] = r["model_response"]
    for r in rows:
        if r["status"] != "ok" or r["exp_id"].startswith("baseline_"):
            continue
        base = baseline_resp.get((baseline_key(r), r["sample_id"]))
        if base is not None:
            r["identical_to_base"] = int(r["model_response"].strip() == base.strip())
            r["_base"] = base

    # BERTScore
    if not args.no_bert:
        scorer = BertScorer(args.bert_model)
        spider_rows = [r for r in rows if r["status"] == "ok" and r["benchmark"] == "spider"]
        if spider_rows:
            cands = [r.get("pred_sql") or r["model_response"] for r in spider_rows]
            for r, f1 in zip(spider_rows, scorer.f1(cands, [r["ground_truth"] for r in spider_rows])):
                r["bert_f1_gt"] = f1
        cmp_rows = [r for r in rows if "_base" in r]
        if cmp_rows:
            for r, f1 in zip(cmp_rows, scorer.f1([r["model_response"] for r in cmp_rows], [r["_base"] for r in cmp_rows])):
                r["bert_f1_vs_base"] = f1
    for r in rows:
        r.pop("_base", None)

    # Scrittura per-task
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["exp_id", "model_id", "device", "benchmark", "stressor_class", "N_workers", "S_size", "repetition",
               "task_index", "sample_id",
               "status", "truncated", "mmlu_pred", "correct", "format_ok", "db_id", "pred_sql", "sql_valid",
               "exec_match", "exact_match", "sql_error", "bert_f1_gt", "identical_to_base", "bert_f1_vs_base",
               "ground_truth", "model_response"]
    with open(out_dir / "quality_tasks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # Riepilogo per esperimento
    metrics = ["correct", "format_ok", "sql_valid", "exec_match", "exact_match", "truncated",
               "bert_f1_gt", "identical_to_base", "bert_f1_vs_base"]
    summary = []
    for exp_id in dict.fromkeys(r["exp_id"] for r in rows):
        sel = [r for r in rows if r["exp_id"] == exp_id]
        ok = [r for r in sel if r["status"] == "ok"]
        s = {k: sel[0][k] for k in ("exp_id", "model_id", "device", "benchmark", "stressor_class", "N_workers", "S_size")}
        s.update({"n_tasks": len(sel), "n_ok": len(ok)})
        for m in metrics:
            vals = [r[m] for r in ok if r.get(m) not in (None, "")]
            s[m] = round(statistics.mean(vals), 4) if vals else ""
        summary.append(s)
    with open(out_dir / "quality_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"[OK] {len(rows)} task valutati -> {out_dir / 'quality_tasks.csv'}")
    print(f"[OK] Riepilogo per esperimento -> {out_dir / 'quality_summary.csv'}\n")
    shown = ["exp_id", "n_ok"] + [m for m in metrics if any(s[m] != "" for s in summary)]
    print(" | ".join(shown))
    for s in summary:
        print(" | ".join(str(s[k]) for k in shown))


if __name__ == "__main__":
    sys.exit(main())
