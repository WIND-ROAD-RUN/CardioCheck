import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


UNIFIED_COLUMNS = [
    "dataset_source",
    "age",
    "sex",
    "height_cm",
    "weight_kg",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "total_cholesterol",
    "cholesterol_level",
    "fasting_glucose",
    "glucose_level",
    "heart_rate",
    "smoker",
    "alcohol",
    "active",
    "diabetes_history",
    "hypertension_history",
    "family_history_cvd",
    "target",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 CardioCheck 统一字段数据集")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="原始数据目录")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="输出目录")
    return parser.parse_args()


def to_binary(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Float64")


def build_cardiovascular(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["dataset_source"] = "cardiovascular"
    # cardio_train 的 age 是天数。
    out["age"] = pd.to_numeric(df.get("age"), errors="coerce") / 365.25
    out["sex"] = (pd.to_numeric(df.get("gender"), errors="coerce") == 2).astype("Float64")
    out["height_cm"] = pd.to_numeric(df.get("height"), errors="coerce")
    out["weight_kg"] = pd.to_numeric(df.get("weight"), errors="coerce")
    out["bmi"] = out["weight_kg"] / ((out["height_cm"] / 100.0) ** 2)
    out["systolic_bp"] = pd.to_numeric(df.get("ap_hi"), errors="coerce")
    out["diastolic_bp"] = pd.to_numeric(df.get("ap_lo"), errors="coerce")
    out["total_cholesterol"] = np.nan
    out["cholesterol_level"] = pd.to_numeric(df.get("cholesterol"), errors="coerce")
    out["fasting_glucose"] = np.nan
    out["glucose_level"] = pd.to_numeric(df.get("gluc"), errors="coerce")
    out["heart_rate"] = np.nan
    out["smoker"] = to_binary(df.get("smoke"))
    out["alcohol"] = to_binary(df.get("alco"))
    out["active"] = to_binary(df.get("active"))
    out["diabetes_history"] = np.nan
    out["hypertension_history"] = np.nan
    out["family_history_cvd"] = np.nan
    out["target"] = to_binary(df.get("cardio"))
    return out


def build_framingham(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["dataset_source"] = "framingham"
    out["age"] = pd.to_numeric(df.get("age"), errors="coerce")
    out["sex"] = to_binary(df.get("male"))
    out["height_cm"] = np.nan
    out["weight_kg"] = np.nan
    out["bmi"] = pd.to_numeric(df.get("BMI"), errors="coerce")
    out["systolic_bp"] = pd.to_numeric(df.get("sysBP"), errors="coerce")
    out["diastolic_bp"] = pd.to_numeric(df.get("diaBP"), errors="coerce")
    out["total_cholesterol"] = pd.to_numeric(df.get("totChol"), errors="coerce")
    out["cholesterol_level"] = np.nan
    out["fasting_glucose"] = pd.to_numeric(df.get("glucose"), errors="coerce")
    out["glucose_level"] = np.nan
    out["heart_rate"] = pd.to_numeric(df.get("heartRate"), errors="coerce")
    out["smoker"] = to_binary(df.get("currentSmoker"))
    out["alcohol"] = np.nan
    out["active"] = np.nan
    out["diabetes_history"] = to_binary(df.get("diabetes"))
    out["hypertension_history"] = to_binary(df.get("prevalentHyp"))
    out["family_history_cvd"] = np.nan
    out["target"] = to_binary(df.get("TenYearCHD"))
    return out


def build_uci(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["dataset_source"] = "uci_cleveland"
    out["age"] = pd.to_numeric(df.get("age"), errors="coerce")
    out["sex"] = to_binary(df.get("sex"))
    out["height_cm"] = np.nan
    out["weight_kg"] = np.nan
    out["bmi"] = np.nan
    out["systolic_bp"] = pd.to_numeric(df.get("trestbps"), errors="coerce")
    out["diastolic_bp"] = np.nan
    out["total_cholesterol"] = pd.to_numeric(df.get("chol"), errors="coerce")
    out["cholesterol_level"] = np.nan
    out["fasting_glucose"] = np.nan
    out["glucose_level"] = to_binary(df.get("fbs"))
    out["heart_rate"] = pd.to_numeric(df.get("thalach"), errors="coerce")
    out["smoker"] = np.nan
    out["alcohol"] = np.nan
    out["active"] = np.nan
    out["diabetes_history"] = np.nan
    out["hypertension_history"] = np.nan
    out["family_history_cvd"] = np.nan
    out["target"] = to_binary(df.get("target"))
    return out


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    for col in ["age", "systolic_bp", "diastolic_bp", "bmi", "heart_rate", "total_cholesterol", "fasting_glucose"]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned.loc[(cleaned["age"] < 18) | (cleaned["age"] > 100), "age"] = np.nan
    cleaned.loc[(cleaned["systolic_bp"] < 70) | (cleaned["systolic_bp"] > 260), "systolic_bp"] = np.nan
    cleaned.loc[(cleaned["diastolic_bp"] < 40) | (cleaned["diastolic_bp"] > 180), "diastolic_bp"] = np.nan
    cleaned.loc[(cleaned["bmi"] < 12) | (cleaned["bmi"] > 70), "bmi"] = np.nan
    cleaned.loc[(cleaned["heart_rate"] < 30) | (cleaned["heart_rate"] > 240), "heart_rate"] = np.nan
    cleaned.loc[(cleaned["total_cholesterol"] < 80) | (cleaned["total_cholesterol"] > 700), "total_cholesterol"] = np.nan
    cleaned.loc[(cleaned["fasting_glucose"] < 40) | (cleaned["fasting_glucose"] > 600), "fasting_glucose"] = np.nan

    # 用身高体重回填 BMI。
    bmi_calc = cleaned["weight_kg"] / ((cleaned["height_cm"] / 100.0) ** 2)
    cleaned["bmi"] = cleaned["bmi"].fillna(bmi_calc)
    cleaned.loc[(cleaned["bmi"] < 12) | (cleaned["bmi"] > 70), "bmi"] = np.nan

    for binary_col in [
        "sex",
        "smoker",
        "alcohol",
        "active",
        "diabetes_history",
        "hypertension_history",
        "family_history_cvd",
        "target",
    ]:
        cleaned[binary_col] = pd.to_numeric(cleaned[binary_col], errors="coerce")

    cleaned = cleaned[UNIFIED_COLUMNS]
    cleaned = cleaned.dropna(subset=["target"])
    cleaned["target"] = cleaned["target"].astype(int)
    return cleaned


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[UNIFIED_COLUMNS]


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cardio_raw = pd.read_csv(raw_dir / "cardiovascular_disease.csv")
    framingham_raw = pd.read_csv(raw_dir / "framingham.csv")
    uci_raw = pd.read_csv(raw_dir / "heart_uci_cleveland.csv")

    cardio = basic_cleaning(ensure_columns(build_cardiovascular(cardio_raw)))
    framingham = basic_cleaning(ensure_columns(build_framingham(framingham_raw)))
    uci = basic_cleaning(ensure_columns(build_uci(uci_raw)))

    merged = pd.concat([cardio, framingham, uci], ignore_index=True)

    save_dataset(cardio, output_dir / "cardio_unified_cardiovascular.csv")
    save_dataset(framingham, output_dir / "cardio_unified_framingham.csv")
    save_dataset(uci, output_dir / "cardio_unified_uci.csv")
    save_dataset(merged, output_dir / "cardio_unified_merged.csv")

    profile = {
        "output_files": {
            "cardiovascular": str(output_dir / "cardio_unified_cardiovascular.csv"),
            "framingham": str(output_dir / "cardio_unified_framingham.csv"),
            "uci": str(output_dir / "cardio_unified_uci.csv"),
            "merged": str(output_dir / "cardio_unified_merged.csv"),
        },
        "row_count": {
            "cardiovascular": int(len(cardio)),
            "framingham": int(len(framingham)),
            "uci": int(len(uci)),
            "merged": int(len(merged)),
        },
        "target_rate": {
            "cardiovascular": float(cardio["target"].mean()),
            "framingham": float(framingham["target"].mean()),
            "uci": float(uci["target"].mean()),
            "merged": float(merged["target"].mean()),
        },
        "columns": UNIFIED_COLUMNS,
    }

    profile_path = output_dir / "unified_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

    print("统一字段数据已生成:")
    print(f"- merged: {output_dir / 'cardio_unified_merged.csv'}")
    print(f"- profile: {profile_path}")


if __name__ == "__main__":
    main()
