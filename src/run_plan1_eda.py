from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_EDA = ROOT / "out" / "eda"
OUT_PROCESSED = ROOT / "out" / "processed"
FIG_EDA = ROOT / "fig" / "eda"

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
COLS_BASE = ["unit_id", "cycle"] + [f"op_setting_{i}" for i in range(1, 4)] + [
    f"sensor_{i}" for i in range(1, 22)
]
FEATURE_COLS = [f"op_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]


def ensure_dirs() -> None:
    OUT_EDA.mkdir(parents=True, exist_ok=True)
    OUT_PROCESSED.mkdir(parents=True, exist_ok=True)
    FIG_EDA.mkdir(parents=True, exist_ok=True)


def load_split(split: str, ds: str) -> pd.DataFrame:
    file_path = DATA_DIR / f"{split}_{ds}.txt"
    df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, :26].copy()
    df.columns = COLS_BASE
    df.insert(0, "dataset", ds)
    return df


def load_rul(ds: str) -> pd.DataFrame:
    file_path = DATA_DIR / f"RUL_{ds}.txt"
    df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, [0]].copy()
    df.columns = ["rul_true"]
    df.insert(0, "unit_id", np.arange(1, len(df) + 1))
    df.insert(0, "dataset", ds)
    return df


def phase1_inventory(
    train: Dict[str, pd.DataFrame], test: Dict[str, pd.DataFrame], rul: Dict[str, pd.DataFrame]
) -> None:
    rows = []
    issues: List[str] = []
    for ds in DATASETS:
        tr = train[ds]
        te = test[ds]
        ru = rul[ds]

        rows.extend(
            [
                {"dataset": ds, "split": "train", "rows": len(tr), "cols": tr.shape[1], "units": tr["unit_id"].nunique()},
                {"dataset": ds, "split": "test", "rows": len(te), "cols": te.shape[1], "units": te["unit_id"].nunique()},
                {"dataset": ds, "split": "rul", "rows": len(ru), "cols": ru.shape[1], "units": ru["unit_id"].nunique()},
            ]
        )

        if tr.shape[1] != 27:
            issues.append(f"{ds}: train columns expected 27, found {tr.shape[1]}")
        if te.shape[1] != 27:
            issues.append(f"{ds}: test columns expected 27, found {te.shape[1]}")
        if ru.shape[1] != 3:
            issues.append(f"{ds}: rul columns expected 3, found {ru.shape[1]}")

        if te["unit_id"].nunique() != len(ru):
            issues.append(
                f"{ds}: mismatch test units ({te['unit_id'].nunique()}) vs rul labels ({len(ru)})"
            )

        if tr.isna().sum().sum() > 0 or te.isna().sum().sum() > 0:
            issues.append(f"{ds}: unexpected NaN detected in train/test")

    pd.DataFrame(rows).to_csv(OUT_EDA / "01_inventory_summary.csv", index=False)

    numeric_summary = []
    for ds in DATASETS:
        all_df = pd.concat([train[ds], test[ds]], ignore_index=True)
        for col in FEATURE_COLS:
            numeric_summary.append(
                {
                    "dataset": ds,
                    "column": col,
                    "min": float(all_df[col].min()),
                    "max": float(all_df[col].max()),
                    "mean": float(all_df[col].mean()),
                    "std": float(all_df[col].std(ddof=0)),
                }
            )
    pd.DataFrame(numeric_summary).to_csv(OUT_EDA / "01_numeric_ranges.csv", index=False)

    with (OUT_EDA / "01_schema_report.txt").open("w", encoding="utf-8") as f:
        f.write("Schema report\n")
        f.write("====================\n")
        if issues:
            f.write("Issues found:\n")
            for item in issues:
                f.write(f"- {item}\n")
        else:
            f.write("No blocking schema issues detected.\n")
        f.write("\nExpected format:\n")
        f.write("- train/test: dataset + 26 columns (27 total)\n")
        f.write("- rul: dataset, unit_id, rul_true\n")


def phase2_stats(train: Dict[str, pd.DataFrame]) -> None:
    train_all = pd.concat([train[ds] for ds in DATASETS], ignore_index=True)
    stats_global = train_all[FEATURE_COLS].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    stats_global.reset_index().rename(columns={"index": "feature"}).to_csv(
        OUT_EDA / "02_stats_global.csv", index=False
    )

    per_ds = []
    for ds in DATASETS:
        d = train[ds]
        for col in FEATURE_COLS:
            per_ds.append(
                {
                    "dataset": ds,
                    "feature": col,
                    "mean": float(d[col].mean()),
                    "std": float(d[col].std(ddof=0)),
                    "min": float(d[col].min()),
                    "p25": float(d[col].quantile(0.25)),
                    "p50": float(d[col].quantile(0.50)),
                    "p75": float(d[col].quantile(0.75)),
                    "max": float(d[col].max()),
                    "is_near_constant": bool(d[col].std(ddof=0) < 1e-6),
                }
            )
    pd.DataFrame(per_ds).to_csv(OUT_EDA / "02_stats_by_dataset.csv", index=False)

    # Correlation heatmap per dataset for sensors only.
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    for ds in DATASETS:
        corr = train[ds][sensor_cols].corr(method="pearson")
        plt.figure(figsize=(10, 8))
        plt.imshow(corr.values, aspect="auto")
        plt.colorbar(label="Pearson correlation")
        plt.xticks(range(len(sensor_cols)), sensor_cols, rotation=90, fontsize=6)
        plt.yticks(range(len(sensor_cols)), sensor_cols, fontsize=6)
        plt.title(f"Sensor Correlation Heatmap - {ds}")
        plt.tight_layout()
        plt.savefig(FIG_EDA / f"correlation_heatmap_{ds.lower()}.png", dpi=160)
        plt.close()


def phase3_temporal(train: Dict[str, pd.DataFrame]) -> None:
    seq_rows = []
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    for ds in DATASETS:
        d = train[ds]
        seq_len = d.groupby("unit_id")["cycle"].max().reset_index(name="max_cycle")
        seq_len["dataset"] = ds
        seq_rows.append(seq_len)

        # Top 3 variable sensors per dataset.
        top_sensors = d[sensor_cols].std(ddof=0).sort_values(ascending=False).head(3).index.tolist()
        trend = d.groupby("cycle")[top_sensors].mean().reset_index()
        plt.figure(figsize=(9, 5))
        for c in top_sensors:
            plt.plot(trend["cycle"], trend[c], label=c)
        plt.title(f"Average Sensor Trends by Cycle - {ds}")
        plt.xlabel("cycle")
        plt.ylabel("avg sensor value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_EDA / f"sensor_trends_{ds.lower()}.png", dpi=160)
        plt.close()

        # Unit examples: first 4 units for one top sensor.
        chosen = top_sensors[0]
        units = sorted(d["unit_id"].unique())[:4]
        plt.figure(figsize=(9, 5))
        for u in units:
            du = d[d["unit_id"] == u]
            plt.plot(du["cycle"], du[chosen], label=f"unit_{u}")
        plt.title(f"Unit Example Trajectories ({chosen}) - {ds}")
        plt.xlabel("cycle")
        plt.ylabel(chosen)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_EDA / f"unit_examples_{ds.lower()}.png", dpi=160)
        plt.close()

    pd.concat(seq_rows, ignore_index=True).to_csv(OUT_EDA / "03_sequence_lengths.csv", index=False)


def build_train_with_rul(train: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for ds in DATASETS:
        d = train[ds].copy()
        max_cycle = d.groupby("unit_id")["cycle"].transform("max")
        d["rul_linear"] = max_cycle - d["cycle"]
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def phase4_rul(train_with_rul: pd.DataFrame, rul: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, int]:
    test_rul = pd.concat([rul[ds] for ds in DATASETS], ignore_index=True)
    candidate_cap = 130
    train_with_rul["rul_capped"] = train_with_rul["rul_linear"].clip(upper=candidate_cap)

    dist_rows = []
    for source, series in [
        ("train_rul_linear", train_with_rul["rul_linear"]),
        ("train_rul_capped", train_with_rul["rul_capped"]),
        ("test_rul_true", test_rul["rul_true"]),
    ]:
        dist_rows.append(
            {
                "source": source,
                "count": int(series.shape[0]),
                "min": float(series.min()),
                "p25": float(series.quantile(0.25)),
                "p50": float(series.quantile(0.50)),
                "p75": float(series.quantile(0.75)),
                "p95": float(series.quantile(0.95)),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
            }
        )
    pd.DataFrame(dist_rows).to_csv(OUT_EDA / "04_rul_distribution.csv", index=False)

    plt.figure(figsize=(10, 5))
    bins = 60
    plt.hist(train_with_rul["rul_linear"], bins=bins, alpha=0.4, label="train_rul_linear")
    plt.hist(train_with_rul["rul_capped"], bins=bins, alpha=0.4, label=f"train_rul_capped_{candidate_cap}")
    plt.hist(test_rul["rul_true"], bins=bins, alpha=0.4, label="test_rul_true")
    plt.legend()
    plt.title("RUL Distributions")
    plt.xlabel("RUL")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(FIG_EDA / "rul_histograms.png", dpi=160)
    plt.close()

    with (OUT_EDA / "04_target_definition.txt").open("w", encoding="utf-8") as f:
        f.write("Target definition\n")
        f.write("====================\n")
        f.write("Primary regression target: rul_capped\n")
        f.write(f"Cap value selected: {candidate_cap}\n")
        f.write("Secondary target for analysis: rul_linear\n")
        f.write("Suggested risk bands:\n")
        f.write("- critical: RUL <= 20\n")
        f.write("- warning: 20 < RUL <= 60\n")
        f.write("- healthy: RUL > 60\n")

    return train_with_rul, candidate_cap


def phase5_preprocess(train_with_rul: pd.DataFrame, test: Dict[str, pd.DataFrame], rul: Dict[str, pd.DataFrame], cap: int) -> None:
    # Feature selection: remove near-constant columns.
    stds = train_with_rul[FEATURE_COLS].std(ddof=0)
    selected = [c for c in FEATURE_COLS if stds[c] >= 1e-6]
    dropped = [c for c in FEATURE_COLS if c not in selected]

    means = train_with_rul[selected].mean()
    std_safe = train_with_rul[selected].std(ddof=0).replace(0, 1.0)

    # Split by unit_id within each dataset for validation.
    train_parts = []
    valid_parts = []
    for ds in DATASETS:
        d = train_with_rul[train_with_rul["dataset"] == ds].copy()
        units = sorted(d["unit_id"].unique())
        cut = max(1, int(len(units) * 0.8))
        train_units = set(units[:cut])
        valid_units = set(units[cut:])
        train_parts.append(d[d["unit_id"].isin(train_units)])
        valid_parts.append(d[d["unit_id"].isin(valid_units)])

    train_split = pd.concat(train_parts, ignore_index=True)
    valid_split = pd.concat(valid_parts, ignore_index=True)

    for frame in [train_split, valid_split]:
        for c in selected:
            frame[f"{c}_z"] = (frame[c] - means[c]) / std_safe[c]
        frame["target_rul_capped"] = frame["rul_linear"].clip(upper=cap)
        frame["target_rul_linear"] = frame["rul_linear"]

    # Prepare test with end-of-unit marker and true RUL on final cycle.
    test_all = pd.concat([test[ds] for ds in DATASETS], ignore_index=True)
    test_all["is_last_cycle"] = test_all.groupby(["dataset", "unit_id"])["cycle"].transform("max") == test_all["cycle"]
    test_all["rul_true_at_end"] = np.nan
    rul_all = pd.concat([rul[ds] for ds in DATASETS], ignore_index=True)
    key_to_rul = {(row["dataset"], int(row["unit_id"])): float(row["rul_true"]) for _, row in rul_all.iterrows()}
    mask = test_all["is_last_cycle"]
    test_all.loc[mask, "rul_true_at_end"] = test_all.loc[mask, ["dataset", "unit_id"]].apply(
        lambda r: key_to_rul[(r["dataset"], int(r["unit_id"]))], axis=1
    )
    for c in selected:
        test_all[f"{c}_z"] = (test_all[c] - means[c]) / std_safe[c]

    with (OUT_EDA / "05_feature_list.txt").open("w", encoding="utf-8") as f:
        f.write("Selected input features\n")
        f.write("====================\n")
        for c in selected:
            f.write(f"- {c}\n")
        f.write("\nDropped near-constant features\n")
        f.write("====================\n")
        for c in dropped:
            f.write(f"- {c}\n")

    config = {
        "normalization": "global_zscore_from_train",
        "selected_features": selected,
        "dropped_features": dropped,
        "target": {"primary": "target_rul_capped", "secondary": "target_rul_linear", "cap": cap},
        "split_strategy": "by_unit_id_within_dataset_80_20",
    }
    with (OUT_EDA / "05_preprocessing_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    train_split.to_parquet(OUT_PROCESSED / "train_processed.parquet", index=False)
    valid_split.to_parquet(OUT_PROCESSED / "valid_processed.parquet", index=False)
    test_all.to_parquet(OUT_PROCESSED / "test_processed.parquet", index=False)


def phase6_close(train_with_rul: pd.DataFrame) -> None:
    # Simple findings from computed statistics.
    near_constant = (train_with_rul[FEATURE_COLS].std(ddof=0) < 1e-6).sum()
    rul_mean = float(train_with_rul["rul_linear"].mean())
    rul_p95 = float(train_with_rul["rul_linear"].quantile(0.95))

    with (OUT_EDA / "06_findings_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Findings Summary\n\n")
        f.write(f"- Near-constant features detected: {near_constant}\n")
        f.write(f"- Train RUL mean: {rul_mean:.2f}\n")
        f.write(f"- Train RUL p95: {rul_p95:.2f}\n")
        f.write("- Multi-dataset setting (FD001-FD004) introduces operating-condition heterogeneity.\n")

    with (OUT_EDA / "06_risks_and_actions.md").open("w", encoding="utf-8") as f:
        f.write("# Risks And Actions\n\n")
        f.write("- Risk: distribution shift across subsets. Action: evaluate per-dataset and mixed training.\n")
        f.write("- Risk: noisy sensors. Action: robust scaling and feature selection.\n")
        f.write("- Risk: temporal leakage. Action: split strictly by unit_id and preserve cycle order.\n")

    with (OUT_EDA / "06_baseline_plan.md").open("w", encoding="utf-8") as f:
        f.write("# Baseline Modeling Plan\n\n")
        f.write("1. Start with gradient boosting on aggregated temporal stats.\n")
        f.write("2. Train LSTM/GRU on sequence windows as temporal baseline.\n")
        f.write("3. Compare with capped-RUL objective and report RMSE/MAE by dataset.\n")


def main() -> None:
    ensure_dirs()
    train = {ds: load_split("train", ds) for ds in DATASETS}
    test = {ds: load_split("test", ds) for ds in DATASETS}
    rul = {ds: load_rul(ds) for ds in DATASETS}

    phase1_inventory(train, test, rul)
    phase2_stats(train)
    phase3_temporal(train)
    train_with_rul = build_train_with_rul(train)
    train_with_rul, cap = phase4_rul(train_with_rul, rul)
    phase5_preprocess(train_with_rul, test, rul, cap)
    phase6_close(train_with_rul)

    print("Plan 1 (EDA) completed.")
    print(f"EDA outputs: {OUT_EDA}")
    print(f"Figures: {FIG_EDA}")
    print(f"Processed data: {OUT_PROCESSED}")


if __name__ == "__main__":
    main()
