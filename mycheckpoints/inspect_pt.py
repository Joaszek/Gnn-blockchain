#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import csv
import torch
from typing import Any, Dict, List

LINE = "-" * 72

def fmt_float(x):
    try:
        return f"{float(x):.6f}"
    except Exception:
        return str(x)

def print_metrics_block(title: str, m: Dict[str, Any]):
    print(f"\n{title}")
    print(LINE)
    keys = ["acc", "precision_macro", "recall_macro", "f1_macro", "auroc"]
    for k in keys:
        if k in m:
            print(f"{k:>18}: {fmt_float(m[k])}")
    # Confusion matrix (optional)
    cm = m.get("confusion_matrix")
    if cm is not None:
        try:
            r, c = len(cm), len(cm[0]) if cm else 0
            print(f"{'confusion_matrix':>18}: {r}x{c}")
        except Exception:
            print(f"{'confusion_matrix':>18}: (available)")

def tail_csv(path: str, n: int = 5) -> List[List[str]]:
    rows = []
    try:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception:
        return []
    if not rows:
        return []
    header, data = rows[0], rows[1:]
    data_tail = data[-n:] if len(data) > n else data
    return [header] + data_tail

def inspect_checkpoint(filepath):
    """Ładuje i drukuje zawartość checkpointu (.pt) oraz sidecar JSON/CSV."""
    if not filepath.endswith(".pt"):
        print(f"Błąd: Plik '{filepath}' nie jest plikiem .pt")
        return

    print(f"--- Ładowanie checkpointu: {filepath} ---")
    try:
        ckpt = torch.load(filepath, map_location=torch.device("cpu"))
        print("Plik załadowany pomyślnie.")

        # Główne klucze
        print("\nKlucze znalezione w pliku:")
        print(list(ckpt.keys()))

        # Epoka
        if "epoch" in ckpt:
            print(f"\nEpoka: {ckpt['epoch']}")

        # EXTRA (metrics + history)
        extra = ckpt.get("extra", {})
        metrics = extra.get("metrics")
        history = extra.get("history")

        # Obsługa metryk: płaskie albo {"train":..., "val":...}
        if metrics:
            if isinstance(metrics, dict) and "train" in metrics and "val" in metrics:
                print_metrics_block("METRYKI (train)", metrics["train"])
                print_metrics_block("METRYKI (val)", metrics["val"])
            else:
                print_metrics_block("METRYKI (this epoch)", metrics)

        # Historia (ostatnie 5)
        if isinstance(history, list) and history:
            print(f"\nHistoria metryk (ostatnie {min(5, len(history))}):")
            print(LINE)
            for row in history[-5:]:
                ep = row.get("epoch")
                # Obsługa obu wariantów (płaski vs train/val)
                if "train" in row and "val" in row:
                    tf1 = row["train"].get("f1_macro")
                    vf1 = row["val"].get("f1_macro")
                    vacc = row["val"].get("acc")
                    print(f"epoch={ep}, train_f1={fmt_float(tf1)}, val_f1={fmt_float(vf1)}, val_acc={fmt_float(vacc)}")
                else:
                    f1 = row.get("f1_macro")
                    acc = row.get("acc")
                    print(f"epoch={ep}, f1={fmt_float(f1)}, acc={fmt_float(acc)}")

        # Model state (krótka zajawka)
        if "model_state" in ckpt:
            ms = ckpt["model_state"]
            print("\n--- Stan Modelu (model_state) ---")
            print(f"Typ: {type(ms)}")
            try:
                print(f"Liczba warstw/wag: {len(ms)}")
                for i, (k, v) in enumerate(ms.items()):
                    if i >= 5:
                        print(f"  ... i {len(ms) - 5} więcej.")
                        break
                    shape = list(v.shape) if hasattr(v, "shape") else "-"
                    print(f"  Klucz: {k:<30} | Kształt: {shape}")
            except Exception:
                pass

        # Optimizer (opcjonalny)
        if "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
            print("\n--- Stan Optymalizatora (optimizer_state) ---")
            try:
                print(f"Klucze: {list(ckpt['optimizer_state'].keys())}")
            except Exception:
                print("optimizer_state: (obecny)")

        # Sidecars w tym samym katalogu
        ckpt_dir = os.path.dirname(os.path.abspath(filepath))
        latest_json = os.path.join(ckpt_dir, "metrics_latest.json")
        history_csv = os.path.join(ckpt_dir, "metrics_history.csv")

        if os.path.exists(latest_json):
            print(f"\n{LINE}\nSidecar: metrics_latest.json")
            try:
                with open(latest_json, "r") as f:
                    mj = json.load(f)
                # pokaż podstawowe pola
                ep = mj.get("epoch")
                print(f"epoch={ep}")
                for k in ["acc", "precision_macro", "recall_macro", "f1_macro", "auroc"]:
                    if k in mj:
                        print(f"{k:>18}: {fmt_float(mj[k])}")
            except Exception as e:
                print(f"(nie udało się odczytać metrics_latest.json: {e})")

        if os.path.exists(history_csv):
            print(f"\n{LINE}\nSidecar: metrics_history.csv (ostatnie 5)")
            try:
                tail = tail_csv(history_csv, n=5)
                if tail:
                    # ładny wydruk CSV
                    header, *rows = tail
                    print(" | ".join(header))
                    for r in rows:
                        print(" | ".join(r))
                else:
                    print("(plik pusty)")
            except Exception as e:
                print(f"(nie udało się odczytać metrics_history.csv: {e})")

    except Exception as e:
        print(f"\nBŁĄD: Nie można załadować pliku {filepath}.")
        print(f"Szczegóły: {e}")
        print("Upewnij się, że masz zainstalowany PyTorch (`pip install torch`).")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Użycie: python3 inspect_ckpt.py <ścieżka_do_pliku.pt>")
        sys.exit(1)
    inspect_checkpoint(sys.argv[1])
