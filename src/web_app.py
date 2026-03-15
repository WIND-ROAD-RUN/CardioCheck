import argparse
from pathlib import Path

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
