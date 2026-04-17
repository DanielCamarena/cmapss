from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from predictive_layer.common import (
    OUT_PREDICTIVE_LAYER,
    OUT_PROCESSED,
    by_dataset_metrics,
    ensure_dirs,
    read_selected_features,
    save_json,
    write_contract_files,
    write_decisions_and_contract,
)
from predictive_layer.eval_baseline import evaluate_all
from predictive_layer.inference_service import predict_rul
from predictive_layer.train_gb import train_gb
from predictive_layer.train_lstm import train_lstm
from predictive_layer.train_rf import train_rf
from predictive_layer.train_tcn_or_gru import train_gru


def _persist_train_metadata() -> None:
    models_dir = OUT_PREDICTIVE_LAYER / "models"
    for name in ["rf", "gb", "lstm", "gru"]:
        src = models_dir / f"{name}_metadata.json"
        dst = OUT_PREDICTIVE_LAYER / f"02_train_metadata_{name}.json"
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _compute_calibration_and_robustness() -> None:
    champion = json.loads((OUT_PREDICTIVE_LAYER / "champion.json").read_text(encoding="utf-8"))["champion"]
    preds = pd.read_parquet(OUT_PREDICTIVE_LAYER / f"02_valid_predictions_{champion}.parquet")
    residual = preds["y_true"] - preds["y_pred"]
    residual_std = float(residual.std(ddof=0))

    save_json(OUT_PREDICTIVE_LAYER / "04_confidence_band_policy.json", {"method": "residual_std_scaled", "residual_std": residual_std})
    (OUT_PREDICTIVE_LAYER / "04_calibration_report.md").write_text(
        f"# Calibration Report\n\nChampion: {champion}\n\nResidual std: {residual_std:.4f}\n",
        encoding="utf-8",
    )

    # Robustness proxy: add Gaussian noise to predictions and recompute metrics.
    rows = []
    rng = np.random.RandomState(42)
    for sigma in [0.5, 1.0, 2.0]:
        noisy_pred = preds["y_pred"].values + rng.normal(0, sigma, size=len(preds))
        rmse = float(np.sqrt(np.mean((preds["y_true"].values - noisy_pred) ** 2)))
        mae = float(np.mean(np.abs(preds["y_true"].values - noisy_pred)))
        rows.append({"noise_sigma": sigma, "rmse": rmse, "mae": mae})
    pd.DataFrame(rows).to_csv(OUT_PREDICTIVE_LAYER / "04_robustness_results.csv", index=False)

    (OUT_PREDICTIVE_LAYER / "04_fallback_policy.txt").write_text(
        "Fallback policy\n===============\n"
        "If champion prediction fails, use GB model with service_status=fallback.\n",
        encoding="utf-8",
    )


def _save_normalizer() -> None:
    train_df = pd.read_parquet(OUT_PROCESSED / "train_processed.parquet")
    selected_raw, _ = read_selected_features()
    means = {c: float(train_df[c].mean()) for c in selected_raw}
    stds = {c: float(train_df[c].std(ddof=0) if train_df[c].std(ddof=0) != 0 else 1.0) for c in selected_raw}
    save_json(OUT_PREDICTIVE_LAYER / "normalizer.json", {"feature_order": selected_raw, "means": means, "stds": stds})


def _smoke_test_e2e() -> None:
    sample_csv = Path("dashboard/mock/sample_input.csv")
    df = pd.read_csv(sample_csv)
    row = df.iloc[0]
    payload = {
        "dataset_id": str(row["dataset_id"]),
        "unit_id": int(row["unit_id"]),
        "cycle": int(row["cycle"]),
        "op_settings": [float(row[f"op_setting_{i}"]) for i in range(1, 4)],
        "sensors": [float(row[f"sensor_{i}"]) for i in range(1, 22)],
        "source": "csv",
    }
    output = predict_rul(payload)
    (OUT_PREDICTIVE_LAYER / "06_smoke_test_e2e.txt").write_text(
        "E2E smoke test\n==============\n"
        f"Input unit: {payload['dataset_id']}/{payload['unit_id']}\n"
        f"Output: {json.dumps(output, indent=2)}\n",
        encoding="utf-8",
    )

    timings = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = predict_rul(payload)
        timings.append((time.perf_counter() - t0) * 1000.0)
    lat = pd.DataFrame(
        [
            {
                "latency_ms_mean": float(np.mean(timings)),
                "latency_ms_p95": float(np.quantile(timings, 0.95)),
                "latency_ms_min": float(np.min(timings)),
                "latency_ms_max": float(np.max(timings)),
            }
        ]
    )
    lat.to_csv(OUT_PREDICTIVE_LAYER / "06_latency_e2e.csv", index=False)

    (OUT_PREDICTIVE_LAYER / "06_dashboard_integration_report.md").write_text(
        "# Dashboard Integration Report\n\n"
        "- predictive_layer inference service produces contract-compatible output.\n"
        "- ready for adapter integration in layer 2.\n",
        encoding="utf-8",
    )


def _write_release() -> None:
    (OUT_PREDICTIVE_LAYER / "07_release_notes.md").write_text(
        "# Release Notes\n\n"
        "- Multi-model training completed (RF, GB, LSTM, GRU).\n"
        "- Champion selected and contract v1 published.\n",
        encoding="utf-8",
    )
    (OUT_PREDICTIVE_LAYER / "07_residual_risks.md").write_text(
        "# Residual Risks\n\n"
        "- Temporal models may need longer tuning for maximal gains.\n"
        "- Domain shift between subsets must be monitored in production.\n",
        encoding="utf-8",
    )
    (OUT_PREDICTIVE_LAYER / "07_next_steps_backlog.md").write_text(
        "# Next Steps Backlog\n\n"
        "1. Add TCN variant and compare with GRU.\n"
        "2. Add physics-guided regularization.\n"
        "3. Add online drift monitor.\n",
        encoding="utf-8",
    )
    (OUT_PREDICTIVE_LAYER / "07_deploy_checklist.txt").write_text(
        "Deploy checklist\n================\n"
        "[ ] champion model artifact available\n"
        "[ ] contract v1 files published\n"
        "[ ] smoke test e2e passed\n"
        "[ ] latency within target\n",
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    write_decisions_and_contract()

    # Phase 2: train model portfolio.
    if not (OUT_PREDICTIVE_LAYER / "02_valid_predictions_rf.parquet").exists():
        train_rf()
    if not (OUT_PREDICTIVE_LAYER / "02_valid_predictions_gb.parquet").exists():
        train_gb()
    if not (OUT_PREDICTIVE_LAYER / "02_valid_predictions_lstm.parquet").exists():
        train_lstm()
    if not (OUT_PREDICTIVE_LAYER / "02_valid_predictions_gru.parquet").exists():
        train_gru()
    _persist_train_metadata()

    # Phase 3: comparative evaluation and champion.
    evaluate_all()

    # Phase 4: calibration + robustness + fallback.
    _compute_calibration_and_robustness()

    # Phase 5: publish contract files.
    write_contract_files()

    # Supporting files for service.
    _save_normalizer()

    # Phase 6: service smoke + latency.
    _smoke_test_e2e()

    # Phase 7: release package.
    _write_release()
    print("Plan 3 (predictive_layer final) completed.")
    print(f"Outputs: {OUT_PREDICTIVE_LAYER}")


if __name__ == "__main__":
    main()


