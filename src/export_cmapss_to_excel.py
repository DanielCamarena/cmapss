from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out" / "excel"

DATASETS = ("FD001", "FD002", "FD003", "FD004")

COLUMNS = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def _load_main_split(split: str, dataset: str) -> pd.DataFrame:
    file_path = DATA_DIR / f"{split}_{dataset}.txt"
    df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, :26].copy()
    df.columns = COLUMNS
    df.insert(0, "dataset", dataset)
    return df


def _load_rul(dataset: str) -> pd.DataFrame:
    file_path = DATA_DIR / f"RUL_{dataset}.txt"
    df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, [0]].copy()
    df.columns = ["rul_true"]
    df.insert(0, "unit_id", range(1, len(df) + 1))
    df.insert(0, "dataset", dataset)
    return df


def build_combined_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_frames = [_load_main_split("train", ds) for ds in DATASETS]
    test_frames = [_load_main_split("test", ds) for ds in DATASETS]
    rul_frames = [_load_rul(ds) for ds in DATASETS]
    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)
    rul_df = pd.concat(rul_frames, ignore_index=True)
    return train_df, test_df, rul_df


def export_excel(train_df: pd.DataFrame, test_df: pd.DataFrame, rul_df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_excel(OUT_DIR / "train.xlsx", index=False)
    test_df.to_excel(OUT_DIR / "test.xlsx", index=False)
    rul_df.to_excel(OUT_DIR / "rul.xlsx", index=False)


def print_summary(train_df: pd.DataFrame, test_df: pd.DataFrame, rul_df: pd.DataFrame) -> None:
    print("Generated Excel files in:", OUT_DIR)
    print(f"train rows: {len(train_df):,} | columns: {train_df.shape[1]}")
    print(f"test rows:  {len(test_df):,} | columns: {test_df.shape[1]}")
    print(f"rul rows:   {len(rul_df):,} | columns: {rul_df.shape[1]}")
    for ds in DATASETS:
        train_units = train_df.loc[train_df["dataset"] == ds, "unit_id"].nunique()
        test_units = test_df.loc[test_df["dataset"] == ds, "unit_id"].nunique()
        rul_units = rul_df.loc[rul_df["dataset"] == ds, "unit_id"].nunique()
        print(f"{ds} -> train units: {train_units}, test units: {test_units}, rul units: {rul_units}")


def main() -> None:
    train_df, test_df, rul_df = build_combined_frames()
    export_excel(train_df, test_df, rul_df)
    print_summary(train_df, test_df, rul_df)


if __name__ == "__main__":
    main()
