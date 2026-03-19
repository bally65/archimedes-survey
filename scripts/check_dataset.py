#!/usr/bin/env python3
"""
check_dataset.py
================
驗證 cscan_dataset 資料集品質。

Usage:
    python3 check_dataset.py data/cscan_dataset/
"""

import json
import os
import sys

import numpy as np


LABELS   = ["C", "B", "O", "M"]
EXPECTED_SHAPE = (25, 25, 40)


def check_dir(label_dir: str, label: str) -> dict:
    npy_files = sorted(f for f in os.listdir(label_dir) if f.endswith(".npy"))
    results = {"label": label, "count": 0, "errors": [], "snr_values": []}

    for fname in npy_files:
        stem = fname[:-4]
        npy_path  = os.path.join(label_dir, fname)
        json_path = os.path.join(label_dir, stem + ".json")

        # Check JSON companion
        if not os.path.exists(json_path):
            results["errors"].append(f"{fname}: missing .json companion")
            continue

        # Load volume
        try:
            vol = np.load(npy_path).astype(np.float32)
        except Exception as e:
            results["errors"].append(f"{fname}: load error: {e}")
            continue

        # Shape check
        if vol.shape != EXPECTED_SHAPE:
            results["errors"].append(
                f"{fname}: shape {vol.shape} != expected {EXPECTED_SHAPE}"
            )
            continue

        # NaN / Inf check
        if not np.isfinite(vol).all():
            results["errors"].append(f"{fname}: contains NaN or Inf")
            continue

        # All-zero check
        if vol.max() == 0.0:
            results["errors"].append(f"{fname}: all-zero volume")
            continue

        # SNR estimate
        nonzero = vol[vol > 0.01]
        if len(nonzero) > 0:
            snr_proxy = float(vol.max() / nonzero.mean())
            results["snr_values"].append(snr_proxy)

        results["count"] += 1

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: check_dataset.py <dataset_root>")
        sys.exit(1)

    root = sys.argv[1]
    if not os.path.isdir(root):
        print(f"ERROR: {root} is not a directory")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Dataset Quality Check: {root}")
    print(f"{'='*60}")

    total_ok    = 0
    total_error = 0
    label_stats = {}

    for label in LABELS:
        label_dir = os.path.join(root, label)
        if not os.path.isdir(label_dir):
            print(f"\n[{label}]  directory not found — 0 samples")
            label_stats[label] = {"count": 0, "mean_snr": None}
            continue

        r = check_dir(label_dir, label)
        total_ok    += r["count"]
        total_error += len(r["errors"])

        mean_snr = (float(np.mean(r["snr_values"]))
                    if r["snr_values"] else None)
        label_stats[label] = {"count": r["count"], "mean_snr": mean_snr}

        snr_str = f"mean SNR proxy = {mean_snr:.2f}" if mean_snr else "no data"
        print(f"\n[{label}]  {r['count']} OK samples    {snr_str}")
        for err in r["errors"]:
            print(f"       ERROR: {err}")

    print(f"\n{'─'*60}")
    print(f"  Total valid samples : {total_ok}")
    print(f"  Total errors        : {total_error}")

    # SNR comparison
    b_snr = label_stats["B"]["mean_snr"]
    c_snr = label_stats["C"]["mean_snr"]
    if b_snr and c_snr:
        delta = b_snr - c_snr
        marker = "OK" if delta > 0 else "WARN (B SNR should > C SNR)"
        print(f"  B/C SNR delta       : {delta:+.2f}  [{marker}]")

    # Minimum count warnings
    MIN = {"C": 50, "B": 50, "O": 20, "M": 20}
    for label, req in MIN.items():
        cnt = label_stats[label]["count"]
        if cnt < req:
            print(f"  WARN [{label}]: only {cnt}/{req} required samples")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
