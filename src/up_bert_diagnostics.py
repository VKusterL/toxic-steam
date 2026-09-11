#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
up_bert_diagnostics.py - lightweight diagnostics for up_bert_user.py runs

Generates CSVs and SVGs without depending on pandas/matplotlib. By default it reads the new
fold_*_loss.json logs and metrics from results/replication_bert_undersampled.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]


def rolling_mean(vals, window=25):
    out = []
    for i in range(len(vals)):
        j = max(0, i - window + 1)
        out.append(mean(vals[j:i + 1]))
    return out


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def svg_line_chart(path, series, title, xlabel, ylabel, width=980, height=560, y_min=None, y_max=None):
    ml, mr, mt, mb = 70, 24, 46, 64
    pw, ph = width - ml - mr, height - mt - mb
    xs = [x for _, pts in series for x, _ in pts]
    ys = [y for _, pts in series for _, y in pts]
    if not xs or not ys:
        return
    if y_min is None:
        y_min = min(ys)
    if y_max is None:
        y_max = max(ys)
    if y_max == y_min:
        y_max += 1
    x_min, x_max = min(xs), max(xs)
    if x_max == x_min:
        x_max += 1

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c",
              "#0891b2", "#4f46e5", "#be123c", "#65a30d", "#0f766e"]

    def x_pos(x):
        return ml + (x - x_min) / (x_max - x_min) * pw

    def y_pos(y):
        return mt + (y_max - y) / (y_max - y_min) * ph

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
    ]
    for i in range(6):
        yy = mt + i * ph / 5
        val = y_max - i * (y_max - y_min) / 5
        lines.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml + pw}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{ml - 10}" y="{yy + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#374151">{val:.3f}</text>')
    for i in range(6):
        xx = ml + i * pw / 5
        val = x_min + i * (x_max - x_min) / 5
        lines.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt + ph}" stroke="#f3f4f6"/>')
        lines.append(f'<text x="{xx:.1f}" y="{mt + ph + 22}" text-anchor="middle" font-family="Arial" font-size="11" fill="#374151">{val:.0f}</text>')
    lines += [
        f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="#111827"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#111827"/>',
        f'<text x="{ml + pw / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="13">{xlabel}</text>',
        f'<text transform="translate(18 {mt + ph / 2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">{ylabel}</text>',
    ]
    for idx, (name, pts) in enumerate(series):
        color = colors[idx % len(colors)]
        points = " ".join(f"{x_pos(x):.1f},{y_pos(y):.1f}" for x, y in pts)
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>')
        lx = ml + 8 + (idx % 3) * 180
        ly = mt + 12 + (idx // 3) * 18
        lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 18}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{lx + 24}" y="{ly + 4}" font-family="Arial" font-size="12" fill="#111827">{name}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def load_loss(loss_dir):
    summary, long_rows = [], []
    for path in sorted(loss_dir.glob("fold_*_loss.json")):
        fold = int(path.stem.split("_")[1])
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data:
            continue
        times = [float(r[0]) for r in data]
        steps = [int(r[1]) for r in data]
        losses = [float(r[2]) for r in data]
        smooth = rolling_mean(losses, 25)
        n = len(losses)
        first = losses[:min(50, n)]
        last = losses[-min(50, n):]
        summary.append({
            "fold": fold,
            "n_steps": n,
            "duration_min": (times[-1] - times[0]) / 60 if len(times) > 1 else 0,
            "loss_first50_mean": mean(first),
            "loss_last50_mean": mean(last),
            "loss_delta_first_last": mean(first) - mean(last),
            "loss_min": min(losses),
            "loss_max": max(losses),
            "loss_final": losses[-1],
            "loss_final_smooth25": smooth[-1],
            "loss_last50_sd": stdev(last) if len(last) > 1 else 0,
        })
        for step, loss, smoothed in zip(steps, losses, smooth):
            long_rows.append({"fold": fold, "step": step, "loss": loss, "loss_smooth25": smoothed})
    return sorted(summary, key=lambda r: r["fold"]), long_rows


def load_metrics(path):
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for key, val in row.items():
                try:
                    parsed[key] = float(val)
                except ValueError:
                    parsed[key] = val
            rows.append(parsed)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Diagnostics for user-level BERT")
    ap.add_argument("--loss-dir", default=str(ROOT / "results" / "replication_bert_undersampled"))
    ap.add_argument("--metrics", default=str(ROOT / "results" / "replication_bert_undersampled" / "bert_user_metrics.csv"))
    ap.add_argument("--out", default=str(ROOT / "fine-tuning_balanced" / "diagnostics_retrain_latest"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    summary, long_rows = load_loss(Path(args.loss_dir))
    if summary:
        write_csv(out_dir / "training_loss_summary.csv", summary, list(summary[0].keys()))
        write_csv(out_dir / "training_loss_long.csv", long_rows, ["fold", "step", "loss", "loss_smooth25"])
        folds = sorted({r["fold"] for r in long_rows})
        svg_line_chart(
            out_dir / "training_loss_curves.svg",
            [(f"fold {f}", [(r["step"], r["loss_smooth25"]) for r in long_rows if r["fold"] == f]) for f in folds],
            "Balanced fine-tuning: training loss per fold",
            "step",
            "train loss",
        )
        svg_line_chart(
            out_dir / "training_loss_summary.svg",
            [
                ("first 50 steps", [(r["fold"], r["loss_first50_mean"]) for r in summary]),
                ("last 50 steps", [(r["fold"], r["loss_last50_mean"]) for r in summary]),
                ("minimum", [(r["fold"], r["loss_min"]) for r in summary]),
            ],
            "Loss summary per fold",
            "fold",
            "loss",
        )

    metrics = load_metrics(Path(args.metrics))
    if metrics:
        write_csv(out_dir / "bert_undersampled_metrics_copy.csv", metrics, list(metrics[0].keys()))
        fields = [f for f in ["auc_pr", "auc_roc", "f1_macro", "kappa"] if f in metrics[0]]
        svg_line_chart(
            out_dir / "bert_metrics_by_fold.svg",
            [(f, [(r["fold"], r[f]) for r in metrics]) for f in fields],
            "BERT undersampled: metrics per fold at real prevalence",
            "fold",
            "metric",
            y_min=0.5,
            y_max=1.0,
        )
    print(f"diagnostics written to: {out_dir}")


if __name__ == "__main__":
    main()
