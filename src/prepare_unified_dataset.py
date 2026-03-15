import argparse
import json
from pathlib import Path
from typing import Any

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
    parser.add_argument(
        "--mapping-config",
        type=str,
        default="config/dataset_mapping.json",
        help="字段映射配置文件路径",
    )
    return parser.parse_args()


def to_binary(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Float64")


def transform_identity(series: pd.Series) -> pd.Series:
    return series


def transform_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def transform_age_days_to_years(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") / 365.25


def transform_gender_2_to_male_binary(series: pd.Series) -> pd.Series:
    return (pd.to_numeric(series, errors="coerce") == 2).astype("Float64")


TRANSFORMS = {
    "identity": transform_identity,
    "to_numeric": transform_to_numeric,
    "binary": to_binary,
    "age_days_to_years": transform_age_days_to_years,
    "gender_2_to_male_binary": transform_gender_2_to_male_binary,
}


def load_mapping_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"映射配置不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def apply_rule(raw_df: pd.DataFrame, rule: dict[str, Any]) -> pd.Series | Any:
    rule_type = rule.get("type")
    if rule_type == "constant":
        return rule.get("value", np.nan)

    if rule_type == "column":
        source_col = rule.get("source")
        if not source_col:
            return np.nan
        series = raw_df[source_col] if source_col in raw_df.columns else pd.Series(np.nan, index=raw_df.index)
        transform_name = rule.get("transform", "identity")
        transform_fn = TRANSFORMS.get(transform_name)
        if transform_fn is None:
            raise ValueError(f"不支持的 transform: {transform_name}")
        return transform_fn(series)

    raise ValueError(f"不支持的 rule type: {rule_type}")


def build_from_mapping(raw_df: pd.DataFrame, dataset_cfg: dict[str, Any]) -> pd.DataFrame:
    out = pd.DataFrame(index=raw_df.index)
    mappings: dict[str, dict[str, Any]] = dataset_cfg.get("mappings", {})

    for target_col in UNIFIED_COLUMNS:
        rule = mappings.get(target_col)
        if not rule:
            out[target_col] = np.nan
            continue

        result = apply_rule(raw_df, rule)
        out[target_col] = result

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
    mapping_config_path = Path(args.mapping_config)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_mapping_config(mapping_config_path)
    datasets_cfg: list[dict[str, Any]] = config.get("datasets", [])
    if not datasets_cfg:
        raise ValueError("映射配置中未找到 datasets 定义")

    unified_frames: dict[str, pd.DataFrame] = {}
    output_files: dict[str, str] = {}

    for dataset_cfg in datasets_cfg:
        dataset_name = str(dataset_cfg.get("name", "unknown"))
        input_file = dataset_cfg.get("input_file")
        output_file = dataset_cfg.get("output_file")
        if not input_file or not output_file:
            raise ValueError(f"数据集配置缺少 input_file/output_file: {dataset_name}")

        raw_df = pd.read_csv(raw_dir / input_file)
        unified_df = build_from_mapping(raw_df, dataset_cfg)
        cleaned_df = basic_cleaning(ensure_columns(unified_df))
        unified_frames[dataset_name] = cleaned_df

        output_path = output_dir / output_file
        save_dataset(cleaned_df, output_path)
        output_files[dataset_name] = str(output_path)

    merged = pd.concat(list(unified_frames.values()), ignore_index=True)
    merged_path = output_dir / "cardio_unified_merged.csv"
    save_dataset(merged, merged_path)
    output_files["merged"] = str(merged_path)

    profile = {
        "mapping_config": str(mapping_config_path),
        "output_files": output_files,
        "row_count": {name: int(len(df)) for name, df in unified_frames.items()} | {"merged": int(len(merged))},
        "target_rate": {name: float(df["target"].mean()) for name, df in unified_frames.items()}
        | {"merged": float(merged["target"].mean())},
        "columns": UNIFIED_COLUMNS,
    }

    profile_path = output_dir / "unified_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

    print("统一字段数据已生成:")
    for name, path in output_files.items():
        print(f"- {name}: {path}")
    print(f"- profile: {profile_path}")


if __name__ == "__main__":
    main()
