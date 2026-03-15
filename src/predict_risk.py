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
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="输入 JSON 文件路径，支持单条对象或对象数组",
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
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"输入 JSON 不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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


def risk_level(probability: float) -> str:
    if probability < 0.33:
        return "low"
    if probability < 0.66:
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


def build_output(records: list[dict[str, Any]], probabilities: np.ndarray) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    for idx, proba in enumerate(probabilities):
        p = float(proba)
        level = risk_level(p)
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
        "results": results,
    }


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_json)
    model_path = Path(args.model_path)
    metadata_path = Path(args.metadata_path)

    payload = load_json(input_path)
    records = normalize_records(payload)

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model = joblib.load(model_path)
    features = load_feature_names(metadata_path, model)
    X = build_feature_frame(records, features)

    probabilities = model.predict_proba(X)[:, 1]
    output = build_output(records, probabilities)

    output_text = json.dumps(output, ensure_ascii=False, indent=2)
    print(output_text)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
        print(f"\n预测结果已保存: {output_path}")


if __name__ == "__main__":
    main()
