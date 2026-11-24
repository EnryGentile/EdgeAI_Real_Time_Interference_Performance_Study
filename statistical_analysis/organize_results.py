"""Riorganizza una cartella di risultati copiata dalla board nella struttura di resultAGX.

Input (come esce dalla board):
    <root>/results/*            file piatti (tasks, responses, doe, ...)
    <root>/data_timeseries/*    serie temporali

Output:
    <root>/<benchmark>/<baseline|stressor>/<llm|slm>/<cpu|gpu>/<file>

Regole:
    baseline_*, responses_baseline_*, doe_*_none.csv   -> baseline
    run_*, responses_run_*, tasks_run_*, doe_*_<fam>  -> stressor
I file che non seguono lo schema (log, ecc.) restano dove sono.
Se il file esiste gia' nella destinazione viene sostituito (la copia dalla board
e' la piu' recente), quindi lo script si puo' rilanciare dopo ogni nuovo scp.

Uso:
    python statistical_analysis/organize_results.py --root resultNano [--dry-run]
"""
import argparse
import os
import re
from collections import Counter
from pathlib import Path

PATTERN = re.compile(
    r"^(?:responses_|tasks_)?(baseline|run|doe)_(llm|slm)-\w+?_(cpu|gpu)_(mmlu|spider)(?:_(.*))?$"
)
SOURCE_DIRS = ("results", "data_timeseries")


def destination(root: Path, name: str):
    m = PATTERN.match(name)
    if not m:
        return None
    kind, model, device, bench, rest = m.groups()
    if kind == "doe":
        family = Path(rest or "").stem
        phase = "baseline" if family == "none" else "stressor"
    else:
        phase = "baseline" if kind == "baseline" else "stressor"
    return root / bench / phase / model / device / name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="resultNano", help="cartella risultati (default: resultNano)")
    ap.add_argument("--dry-run", action="store_true", help="mostra cosa farebbe senza spostare")
    args = ap.parse_args()

    root = Path(args.root)
    moved, skipped = Counter(), []
    for sub in SOURCE_DIRS:
        src_dir = root / sub
        if not src_dir.is_dir():
            continue
        for f in sorted(src_dir.iterdir()):
            if not f.is_file():
                continue
            dst = destination(root, f.name)
            if dst is None:
                skipped.append(f)
                continue
            moved[str(dst.parent.relative_to(root))] += 1
            if not args.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(f, dst)

    for folder, n in sorted(moved.items()):
        print(f"{n:6d}  {folder}")
    print(f"{sum(moved.values()):6d}  totale {'da spostare' if args.dry_run else 'spostati'}")
    for f in skipped:
        print(f"  lasciato: {f}")

    if not args.dry_run:
        for sub in SOURCE_DIRS:
            d = root / sub
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()


if __name__ == "__main__":
    main()
