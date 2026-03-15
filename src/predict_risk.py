import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


DEFAULT_MODEL_PATH = "model/cardio_risk_multisource_ensemble.joblib"
DEFAULT_METADATA_PATH = "model/cardio_risk_multisource_ensemble_meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CardioCheck 风险预测接口")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-json",
        type=str,
        help="输入 JSON 文件路径，支持单条对象或对象数组",
    )
    input_group.add_argument(
        "--input-csv",
        type=str,
        help="输入 CSV 文件路径，适用于批量预测",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="训练好的模型路径",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=DEFAULT_METADATA_PATH,
        help="训练元数据路径，用于确定特征顺序",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="可选，预测输出 JSON 保存路径",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="可选，预测输出 CSV 保存路径",
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=0.33,
        help="low 与 medium 的分界阈值，默认 0.33",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.66,
        help="medium 与 high 的分界阈值，默认 0.66",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"输入 JSON 不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"输入 CSV 不存在: {path}")
    frame = pd.read_csv(path)
    return frame.to_dict(orient="records")


def load_feature_names(metadata_path: Path, model: Any) -> list[str]:
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        features = metadata.get("features")
        if isinstance(features, list) and features:
            return [str(f) for f in features]

    if hasattr(model, "feature_names_in_"):
        return [str(f) for f in model.feature_names_in_]

    raise ValueError("无法从元数据或模型中获取特征列表")


def normalize_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    raise ValueError("输入 JSON 必须是对象或对象数组")


def risk_level(probability: float, low_threshold: float, high_threshold: float) -> str:
    if probability < low_threshold:
        return "low"
    if probability < high_threshold:
        return "medium"
    return "high"


def risk_message(level: str) -> str:
    if level == "low":
        return "低风险：建议保持当前生活方式并定期体检。"
    if level == "medium":
        return "中风险：建议关注血压、血糖、血脂并进行生活方式干预。"
    return "高风险：建议尽快进行专业心血管评估与医生随访。"


def build_feature_frame(records: list[dict[str, Any]], features: list[str]) -> pd.DataFrame:
    normalized_rows: list[dict[str, Any]] = []

    for row in records:
        item = dict(row)
        item.setdefault("dataset_source", "user_input")
        normalized_rows.append(item)

    frame = pd.DataFrame(normalized_rows)

    for col in features:
        if col not in frame.columns:
            frame[col] = np.nan

    return frame[features]


def validate_thresholds(low_threshold: float, high_threshold: float) -> None:
    if not (0.0 <= low_threshold <= 1.0 and 0.0 <= high_threshold <= 1.0):
        raise ValueError("阈值必须在 [0, 1] 范围内")
    if low_threshold >= high_threshold:
        raise ValueError("low-threshold 必须小于 high-threshold")


def build_output(
    records: list[dict[str, Any]],
    probabilities: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    for idx, proba in enumerate(probabilities):
        p = float(proba)
        level = risk_level(p, low_threshold=low_threshold, high_threshold=high_threshold)
        results.append(
            {
                "index": idx,
                "risk_probability": round(p, 6),
                "risk_level": level,
                "risk_message": risk_message(level),
                "input": records[idx],
            }
        )

    return {
        "count": len(results),
        "thresholds": {
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
        },
        "results": results,
    }


def build_csv_output(output: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in output.get("results", []):
        rows.append(
            {
                "index": item["index"],
                "risk_probability": item["risk_probability"],
                "risk_level": item["risk_level"],
                "risk_message": item["risk_message"],
            }
        )
    return pd.DataFrame(rows)


def load_model_and_features(model_path: Path, metadata_path: Path) -> tuple[Any, list[str]]:
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    model = joblib.load(model_path)
    features = load_feature_names(metadata_path, model)
    return model, features


def predict_from_records(
    records: list[dict[str, Any]],
    model: Any,
    features: list[str],
    low_threshold: float,
    high_threshold: float,
) -> dict[str, Any]:
    validate_thresholds(low_threshold, high_threshold)
    X = build_feature_frame(records, features)
    probabilities = model.predict_proba(X)[:, 1]
    return build_output(records, probabilities, low_threshold=low_threshold, high_threshold=high_threshold)


def load_input_records(input_json: str | None, input_csv: str | None) -> list[dict[str, Any]]:
    if input_json:
        payload = load_json(Path(input_json))
        return normalize_records(payload)
    if input_csv:
        return load_csv(Path(input_csv))
    raise ValueError("必须提供 --input-json 或 --input-csv")


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    metadata_path = Path(args.metadata_path)
    records = load_input_records(args.input_json, args.input_csv)

    model, features = load_model_and_features(model_path, metadata_path)
    output = predict_from_records(
        records,
        model=model,
        features=features,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )

    output_text = json.dumps(output, ensure_ascii=False, indent=2)
    print(output_text)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
        print(f"\n预测结果已保存: {output_path}")

    if args.output_csv:
        output_csv_path = Path(args.output_csv)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_frame = build_csv_output(output)
        csv_frame.to_csv(output_csv_path, index=False)
        print(f"预测结果 CSV 已保存: {output_csv_path}")


if __name__ == "__main__":
    main()
