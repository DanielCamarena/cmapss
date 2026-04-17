from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .common import load_train_valid_test, read_selected_features


def _make_windows(df: pd.DataFrame, feature_cols: List[str], target_col: str, window: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x_list = []
    y_list = []
    meta = []

    for (_, unit_id), d in df.groupby(["dataset", "unit_id"], sort=False):
        d = d.sort_values("cycle")
        x = d[feature_cols].values.astype(np.float32)
        y = d[target_col].values.astype(np.float32)
        ds = d["dataset"].values
        cyc = d["cycle"].values

        if len(d) < window:
            continue

        for i in range(window - 1, len(d)):
            x_list.append(x[i - window + 1 : i + 1])
            y_list.append(y[i])
            meta.append((ds[i], int(unit_id), int(cyc[i])))

    x_arr = np.stack(x_list, axis=0) if x_list else np.empty((0, window, len(feature_cols)), dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.float32)
    meta_df = pd.DataFrame(meta, columns=["dataset", "unit_id", "cycle"])
    return x_arr, y_arr, meta_df


def build_temporal_data(target_col: str = "target_rul_capped", window: int = 30, max_train_samples: int = 30000):
    train_df, valid_df, _ = load_train_valid_test()
    _, feature_cols_z = read_selected_features()

    x_train, y_train, _ = _make_windows(train_df, feature_cols_z, target_col, window)
    x_valid, y_valid, valid_meta = _make_windows(valid_df, feature_cols_z, target_col, window)

    if len(x_train) > max_train_samples:
        idx = np.random.RandomState(42).choice(len(x_train), size=max_train_samples, replace=False)
        x_train = x_train[idx]
        y_train = y_train[idx]

    return x_train, y_train, x_valid, y_valid, valid_meta, feature_cols_z

