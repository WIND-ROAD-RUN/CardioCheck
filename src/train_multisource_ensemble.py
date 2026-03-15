import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 CardioCheck 多数据集集成模型")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/cardio_unified_merged.csv",
        help="统一后的训练数据路径",
    )
    parser.add_argument("--target", type=str, default="target", help="标签列名")
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例")
    parser.add_argument(
        "--output-model",
        type=str,
        default="model/cardio_risk_multisource_ensemble.joblib",
        help="模型输出路径",
    )
    parser.add_argument(
        "--output-metadata",
        type=str,
        default="model/cardio_risk_multisource_ensemble_meta.json",
        help="模型元数据输出路径",
    )
    return parser.parse_args()


def load_data(path: str, target_col: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"找不到标签列: {target_col}")
    if "dataset_source" not in df.columns:
        raise ValueError("训练数据需要 dataset_source 列用于来源评估")

    y = df[target_col].astype(int)
    source = df["dataset_source"].astype(str)
    X = df.drop(columns=[target_col])
    return X, y, source


def build_pipeline(X_train: pd.DataFrame) -> Pipeline:
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X_train.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_split=6,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    gb = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    )

    lr = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
        voting="soft",
        weights=[2, 2, 1],
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", ensemble),
        ]
    )


def print_metrics(title: str, y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }

    print(f"\n=== {title} ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("\n混淆矩阵:")
    print(confusion_matrix(y_true, y_pred))
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    return metrics


def evaluate_by_source(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    source: pd.Series,
) -> dict:
    result: dict = {}
    print("\n=== 分数据源评估 ===")
    for name in sorted(source.unique()):
        mask = source == name
        if int(mask.sum()) < 10:
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]
        y_pr = y_proba[mask]

        # 子数据源可能出现单类别测试样本，AUC 需兜底。
        try:
            auc = float(roc_auc_score(y_t, y_pr))
        except ValueError:
            auc = float("nan")

        row = {
            "size": int(mask.sum()),
            "accuracy": float(accuracy_score(y_t, y_p)),
            "precision": float(precision_score(y_t, y_p, zero_division=0)),
            "recall": float(recall_score(y_t, y_p, zero_division=0)),
            "f1": float(f1_score(y_t, y_p, zero_division=0)),
            "roc_auc": auc,
        }
        result[name] = row
        print(
            f"- {name}: n={row['size']}, acc={row['accuracy']:.4f}, "
            f"f1={row['f1']:.4f}, auc={row['roc_auc'] if not np.isnan(row['roc_auc']) else 'nan'}"
        )
    return result


def main() -> None:
    args = parse_args()
    X, y, source = load_data(args.data, args.target)

    X_train, X_test, y_train, y_test, _, source_test = train_test_split(
        X,
        y,
        source,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_pipeline(X_train)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    overall_metrics = print_metrics("总体评估", y_test, y_pred, y_proba)
    source_metrics = evaluate_by_source(y_test, y_pred, y_proba, source_test)

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_model)
    print(f"\n模型已保存: {output_model}")

    metadata = {
        "data_path": args.data,
        "target": args.target,
        "features": list(X.columns),
        "test_size": args.test_size,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "overall_metrics": overall_metrics,
        "source_metrics": source_metrics,
    }
    output_metadata = Path(args.output_metadata)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"模型元数据已保存: {output_metadata}")


if __name__ == "__main__":
    main()