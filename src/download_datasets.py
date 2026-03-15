import argparse
import io
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from sklearn.datasets import fetch_openml


UCI_CLEVELAND_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)

UCI_CLEVELAND_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

OPENML_NAME_CANDIDATES = {
    "framingham": [
        "framingham",
        "Framingham",
        "framingham_wdbc",
    ],
    "cardiovascular-disease": [
        "cardiovascular-disease",
        "Cardiovascular-Disease-dataset",
        "cardio_train",
        "cardiovascular disease",
    ],
}

FRAMINGHAM_FALLBACK_URLS = [
    "https://raw.githubusercontent.com/GauravPadawe/Framingham-Heart-Study/master/framingham.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 CardioCheck 原型所需数据集")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=["uci-cleveland", "framingham", "cardiovascular-disease", "all"],
        help="指定要下载的数据集，可重复传入；不传时默认下载 all",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="下载输出目录，默认为 data/raw",
    )
    parser.add_argument(
        "--cardio-url",
        type=str,
        default=None,
        help="Cardiovascular Disease Dataset 的 CSV 镜像地址，可覆盖自动检索",
    )
    parser.add_argument(
        "--framingham-url",
        type=str,
        default="https://raw.githubusercontent.com/GauravPadawe/Framingham-Heart-Study/master/framingham.csv",
        help="Framingham 数据集的 CSV 镜像地址，可覆盖自动检索",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP 请求超时时间，单位秒，默认 60",
    )
    return parser.parse_args()


def request_text(url: str, timeout: int) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    return response.text


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def download_uci_cleveland(output_dir: Path, timeout: int) -> dict:
    raw_text = request_text(UCI_CLEVELAND_URL, timeout)
    df = pd.read_csv(
        io.StringIO(raw_text),
        header=None,
        names=UCI_CLEVELAND_COLUMNS,
        na_values="?",
    )
    df["target"] = (df["target"] > 0).astype(int)
    output_path = output_dir / "heart_uci_cleveland.csv"
    save_dataframe(df, output_path)
    return {
        "dataset": "uci-cleveland",
        "rows": int(len(df)),
        "columns": list(df.columns),
        "path": str(output_path),
        "source": UCI_CLEVELAND_URL,
    }


def fetch_csv_from_url(url: str, output_path: Path, timeout: int) -> dict:
    raw_text = request_text(url, timeout)
    raw_text = raw_text.lstrip("\ufeff")
    try:
        df = pd.read_csv(io.StringIO(raw_text), sep=",")
    except Exception:
        df = pd.read_csv(
            io.StringIO(raw_text),
            sep=",",
            engine="python",
            on_bad_lines="skip",
        )
    save_dataframe(df, output_path)
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "path": str(output_path),
        "source": url,
    }


def iter_openml_candidates(dataset_names: Iterable[str], timeout: int) -> tuple[int, str] | None:
    for dataset_name in dataset_names:
        api_url = (
            "https://www.openml.org/api/v1/json/data/list/data_name/"
            f"{requests.utils.quote(dataset_name)}"
            "/limit/20"
        )
        response = requests.get(api_url, timeout=timeout)
        if response.status_code != 200:
            continue
        payload = response.json()
        datasets = payload.get("data", {}).get("dataset", [])
        if isinstance(datasets, dict):
            datasets = [datasets]
        if not datasets:
            continue

        exact_matches = [
            item
            for item in datasets
            if str(item.get("name", "")).strip().lower() == dataset_name.strip().lower()
        ]
        pool = exact_matches or datasets
        active_pool = [item for item in pool if str(item.get("status", "")).lower() == "active"]
        chosen_pool = active_pool or pool
        chosen_pool.sort(key=lambda item: int(item.get("version", 0)), reverse=True)
        selected = chosen_pool[0]
        did = selected.get("did")
        name = selected.get("name")
        if did and name:
            return int(did), str(name)
    return None


def download_openml_dataset(
    dataset_key: str,
    output_filename: str,
    output_dir: Path,
    timeout: int,
) -> dict:
    candidates = OPENML_NAME_CANDIDATES[dataset_key]
    match = iter_openml_candidates(candidates, timeout)
    if match is None:
        raise RuntimeError(
            f"无法在 OpenML 自动定位数据集 {dataset_key}。"
            "请使用对应的 --*-url 参数传入可下载的 CSV 地址。"
        )

    data_id, data_name = match
    bunch = fetch_openml(data_id=data_id, as_frame=True)
    frame = bunch.frame.copy()
    output_path = output_dir / output_filename
    save_dataframe(frame, output_path)
    return {
        "dataset": dataset_key,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "path": str(output_path),
        "source": f"openml:{data_name}:{data_id}",
    }


def download_framingham(output_dir: Path, timeout: int, source_url: str | None) -> dict:
    output_path = output_dir / "framingham.csv"
    if source_url:
        result = fetch_csv_from_url(source_url, output_path, timeout)
        result["dataset"] = "framingham"
        return result

    try:
        return download_openml_dataset("framingham", "framingham.csv", output_dir, timeout)
    except Exception:
        for fallback_url in FRAMINGHAM_FALLBACK_URLS:
            try:
                result = fetch_csv_from_url(fallback_url, output_path, timeout)
                result["dataset"] = "framingham"
                return result
            except Exception:
                continue

    raise RuntimeError(
        "无法通过 OpenML 或默认镜像下载 framingham。"
        "请使用 --framingham-url 传入可下载的 CSV 地址。"
    )


def download_cardiovascular(output_dir: Path, timeout: int, source_url: str | None) -> dict:
    output_path = output_dir / "cardiovascular_disease.csv"
    if source_url:
        result = fetch_csv_from_url(source_url, output_path, timeout)
        result["dataset"] = "cardiovascular-disease"
        return result
    return download_openml_dataset(
        "cardiovascular-disease",
        "cardiovascular_disease.csv",
        output_dir,
        timeout,
    )


def expand_requested_datasets(values: list[str] | None) -> list[str]:
    if not values or "all" in values:
        return ["cardiovascular-disease", "framingham", "uci-cleveland"]
    return values


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    requested = expand_requested_datasets(args.dataset)

    downloaders = {
        "uci-cleveland": lambda: download_uci_cleveland(output_dir, args.timeout),
        "framingham": lambda: download_framingham(output_dir, args.timeout, args.framingham_url),
        "cardiovascular-disease": lambda: download_cardiovascular(
            output_dir,
            args.timeout,
            args.cardio_url,
        ),
    }

    results: list[dict] = []
    failures: list[dict] = []

    for dataset in requested:
        try:
            result = downloaders[dataset]()
            results.append(result)
            print(
                f"[OK] {dataset}: rows={result['rows']}, cols={len(result['columns'])}, "
                f"saved={result['path']}"
            )
        except Exception as exc:
            failures.append({"dataset": dataset, "error": str(exc)})
            print(f"[FAILED] {dataset}: {exc}")

    manifest = {
        "downloaded": results,
        "failed": failures,
    }
    manifest_path = output_dir / "download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n下载清单已写入: {manifest_path}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()