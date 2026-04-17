from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .common import OUT_PREDICTIVE_LAYER, by_dataset_metrics, by_rul_band_metrics, evaluate_latency, metrics_frame


def _load_preds() -> dict[str, pd.DataFrame]:
    files = {
        "rf": OUT_PREDICTIVE_LAYER / "02_valid_predictions_rf.parquet",
        "gb": OUT_PREDICTIVE_LAYER / "02_valid_predictions_gb.parquet",
        "lstm": OUT_PREDICTIVE_LAYER / "02_valid_predictions_lstm.parquet",
        "gru": OUT_PREDICTIVE_LAYER / "02_valid_predictions_gru.parquet",
    }
    return {k: pd.read_parquet(v) for k, v in files.items()}


def _compute_metrics(preds: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_rows = []
    by_ds_rows = []
    by_band_rows = []
    for name, df in preds.items():
        m = metrics_frame(df["y_true"].values, df["y_pred"].values)
        global_rows.append({"model_name": name, **m, "count": int(len(df))})
        ds = by_dataset_metrics(df)
        ds["model_name"] = name
        by_ds_rows.append(ds)
        bb = by_rul_band_metrics(df)
        bb["model_name"] = name
        by_band_rows.append(bb)

    return (
        pd.DataFrame(global_rows),
        pd.concat(by_ds_rows, ignore_index=True),
        pd.concat(by_band_rows, ignore_index=True),
    )


def _compute_latency() -> pd.DataFrame:
    rows = []
    models = {
        "rf": joblib.load(OUT_PREDICTIVE_LAYER / "models" / "rf_model.joblib"),
        "gb": joblib.load(OUT_PREDICTIVE_LAYER / "models" / "gb_model.joblib"),
        "lstm": joblib.load(OUT_PREDICTIVE_LAYER / "models" / "lstm_model.joblib"),
        "gru": joblib.load(OUT_PREDICTIVE_LAYER / "models" / "gru_model.joblib"),
    }
    # Tabular features count from rf.
    d_tab = models["rf"].n_features_in_
    d_seq = models["lstm"].n_features_in_
    x_tab = np.random.randn(1024, d_tab).astype(np.float32)
    x_seq_flat = np.random.randn(512, d_seq).astype(np.float32)
    rows.append({"model_name": "rf", **evaluate_latency(models["rf"].predict, x_tab)})
    rows.append({"model_name": "gb", **evaluate_latency(models["gb"].predict, x_tab)})
    rows.append({"model_name": "lstm", **evaluate_latency(models["lstm"].predict, x_seq_flat)})
    rows.append({"model_name": "gru", **evaluate_latency(models["gru"].predict, x_seq_flat)})
    return pd.DataFrame(rows)


def evaluate_all() -> str:
    preds = _load_preds()
    m_global, m_ds, m_band = _compute_metrics(preds)
    latency = _compute_latency()

    m_global.to_csv(OUT_PREDICTIVE_LAYER / "03_metrics_global_by_model.csv", index=False)
    m_ds.to_csv(OUT_PREDICTIVE_LAYER / "03_metrics_by_dataset_by_model.csv", index=False)
    m_band.to_csv(OUT_PREDICTIVE_LAYER / "03_error_by_rul_band_by_model.csv", index=False)
    latency.to_csv(OUT_PREDICTIVE_LAYER / "03_latency_by_model.csv", index=False)

    merged = m_global.merge(latency[["model_name", "latency_ms_p95"]], on="model_name", how="left")
    merged = merged.sort_values(by=["rmse", "latency_ms_p95"], ascending=[True, True]).reset_index(drop=True)
    champion = merged.iloc[0]["model_name"]

    (OUT_PREDICTIVE_LAYER / "03_champion_decision_record.md").write_text(
        "\n".join(
            [
                "# Champion Decision Record",
                "",
                "Selection rule: minimum RMSE global, tie-break by p95 latency.",
                f"- Champion: **{champion}**",
                "",
                "## Ranking",
                "```text",
                merged.to_string(index=False),
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    Path(OUT_PREDICTIVE_LAYER / "champion.json").write_text(json.dumps({"champion": champion}, indent=2), encoding="utf-8")
    return champion


if __name__ == "__main__":
    print("Champion:", evaluate_all())


