import argparse
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from predict_risk import load_model_and_features, predict_from_records


DEFAULT_MODEL_PATH = "model/cardio_risk_multisource_ensemble.joblib"
DEFAULT_METADATA_PATH = "model/cardio_risk_multisource_ensemble_meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CardioCheck 最小 Web 预测原型")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务监听地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="模型路径")
    parser.add_argument("--metadata-path", type=str, default=DEFAULT_METADATA_PATH, help="模型元数据路径")
    parser.add_argument("--low-threshold", type=float, default=0.33, help="low 与 medium 阈值")
    parser.add_argument("--high-threshold", type=float, default=0.66, help="medium 与 high 阈值")
    parser.add_argument("--debug", action="store_true", help="是否开启 debug")
    return parser.parse_args()


def create_app(
    model_path: Path,
    metadata_path: Path,
    low_threshold: float,
    high_threshold: float,
) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    model, features = load_model_and_features(model_path=model_path, metadata_path=metadata_path)

    def validate_record(payload: dict[str, Any]) -> list[str]:
        errors: list[str] = []

        def as_float(key: str) -> float | None:
            value = payload.get(key)
            if value is None or value == "":
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                errors.append(f"{key} 必须是数值")
                return None

        age = as_float("age")
        sex = as_float("sex")
        sbp = as_float("systolic_bp")
        dbp = as_float("diastolic_bp")
        bmi = as_float("bmi")

        if age is not None and not (18 <= age <= 100):
            errors.append("age 需在 18 到 100 之间")
        if sex is not None and sex not in (0.0, 1.0):
            errors.append("sex 仅支持 0 或 1")
        if sbp is not None and not (70 <= sbp <= 260):
            errors.append("systolic_bp 建议在 70 到 260 之间")
        if dbp is not None and not (40 <= dbp <= 180):
            errors.append("diastolic_bp 建议在 40 到 180 之间")
        if bmi is not None and not (12 <= bmi <= 70):
            errors.append("bmi 建议在 12 到 70 之间")

        for binary_key in ["smoker", "alcohol", "active", "diabetes_history", "hypertension_history"]:
            val = as_float(binary_key)
            if val is not None and val not in (0.0, 1.0):
                errors.append(f"{binary_key} 仅支持 0 或 1")

        if payload.get("age") is None:
            errors.append("age 为必填项")
        if payload.get("sex") is None:
            errors.append("sex 为必填项")
        if payload.get("systolic_bp") is None:
            errors.append("systolic_bp 为必填项")

        return errors

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )

    @app.post("/api/predict")
    def api_predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "请求体必须是 JSON 对象"}), 400

        errors = validate_record(payload)
        if errors:
            return jsonify({"error": "输入校验失败", "details": errors}), 400

        try:
            output = predict_from_records(
                [payload],
                model=model,
                features=features,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify(output["results"][0])

    @app.post("/api/predict-batch")
    def api_predict_batch():
        payload = request.get_json(silent=True)
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            return jsonify({"error": "请求体必须是 JSON 对象数组"}), 400

        all_errors: list[dict[str, Any]] = []
        for idx, item in enumerate(payload):
            errors = validate_record(item)
            if errors:
                all_errors.append({"index": idx, "errors": errors})

        if all_errors:
            return jsonify({"error": "批量输入校验失败", "details": all_errors}), 400

        try:
            output = predict_from_records(
                payload,
                model=model,
                features=features,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify(output)

    return app


def main() -> None:
    args = parse_args()
    app = create_app(
        model_path=Path(args.model_path),
        metadata_path=Path(args.metadata_path),
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
