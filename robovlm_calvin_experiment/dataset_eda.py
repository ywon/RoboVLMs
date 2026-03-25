import os
import csv
import json
import argparse
from pathlib import Path

import numpy as np


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def safe_stat_numeric(arr: np.ndarray):
    try:
        if arr.size == 0:
            return {"min": None, "max": None, "mean": None}
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
        }
    except Exception:
        return {"min": None, "max": None, "mean": None}


def inspect_npz_file(npz_path: Path, allow_pickle: bool = False):
    result = {
        "file": str(npz_path),
        "file_size_bytes": npz_path.stat().st_size,
        "arrays": [],
        "error": None,
    }

    try:
        with np.load(npz_path, allow_pickle=allow_pickle) as data:
            keys = list(data.keys())

            for key in keys:
                try:
                    arr = data[key]

                    item = {
                        "key": key,
                        "shape": tuple(arr.shape) if hasattr(arr, "shape") else None,
                        "dtype": str(arr.dtype) if hasattr(arr, "dtype") else str(type(arr)),
                        "ndim": int(arr.ndim) if hasattr(arr, "ndim") else None,
                        "size": int(arr.size) if hasattr(arr, "size") else None,
                    }

                    if hasattr(arr, "nbytes"):
                        item["nbytes"] = int(arr.nbytes)
                    else:
                        item["nbytes"] = None

                    if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.number):
                        stats = safe_stat_numeric(arr)
                        item.update(stats)
                    else:
                        item["min"] = None
                        item["max"] = None
                        item["mean"] = None

                    if isinstance(arr, np.ndarray) and arr.dtype == object:
                        preview = None
                        try:
                            if arr.size > 0:
                                preview = repr(arr.flat[0])[:200]
                        except Exception:
                            preview = None
                        item["object_preview"] = preview
                    else:
                        item["object_preview"] = None

                    result["arrays"].append(item)

                except Exception as e:
                    result["arrays"].append(
                        {
                            "key": key,
                            "shape": None,
                            "dtype": None,
                            "ndim": None,
                            "size": None,
                            "nbytes": None,
                            "min": None,
                            "max": None,
                            "mean": None,
                            "object_preview": None,
                            "array_error": str(e),
                        }
                    )

    except Exception as e:
        result["error"] = str(e)

    return result


def collect_npz_files(root: Path):
    return sorted(root.rglob("*.npz"))


def summarize_results(results):
    summary = {
        "num_files": len(results),
        "num_ok_files": sum(r["error"] is None for r in results),
        "num_error_files": sum(r["error"] is not None for r in results),
        "total_file_size_bytes": sum(r["file_size_bytes"] for r in results),
        "key_frequency": {},
        "shape_examples": {},
        "dtype_examples": {},
    }

    for r in results:
        if r["error"] is not None:
            continue

        for arr in r["arrays"]:
            key = arr["key"]
            summary["key_frequency"][key] = summary["key_frequency"].get(key, 0) + 1

            if key not in summary["shape_examples"]:
                summary["shape_examples"][key] = str(arr["shape"])
            if key not in summary["dtype_examples"]:
                summary["dtype_examples"][key] = str(arr["dtype"])

    return summary


def print_human_readable(results, summary, max_files_to_show=20):
    print("=" * 100)
    print("NPZ DATASET INSPECTION SUMMARY")
    print("=" * 100)
    print(f"Total files found       : {summary['num_files']}")
    print(f"Successfully opened     : {summary['num_ok_files']}")
    print(f"Failed to open          : {summary['num_error_files']}")
    print(f"Total file size         : {format_bytes(summary['total_file_size_bytes'])}")
    print()

    print("-" * 100)
    print("Key frequency across files")
    print("-" * 100)
    if summary["key_frequency"]:
        for key, freq in sorted(summary["key_frequency"].items(), key=lambda x: (-x[1], x[0])):
            shape_ex = summary["shape_examples"].get(key)
            dtype_ex = summary["dtype_examples"].get(key)
            print(f"{key:20s} | files={freq:5d} | example_shape={shape_ex:20s} | example_dtype={dtype_ex}")
    else:
        print("No readable arrays found.")
    print()

    print("-" * 100)
    print(f"Per-file details (showing up to {max_files_to_show} files)")
    print("-" * 100)

    for i, r in enumerate(results[:max_files_to_show]):
        print(f"[{i+1}] {r['file']}")
        print(f"    file size: {format_bytes(r['file_size_bytes'])}")

        if r["error"] is not None:
            print(f"    ERROR: {r['error']}")
            print()
            continue

        for arr in r["arrays"]:
            if "array_error" in arr:
                print(f"    - key={arr['key']}: ERROR while reading array: {arr['array_error']}")
                continue

            line = (
                f"    - key={arr['key']}, shape={arr['shape']}, dtype={arr['dtype']}, "
                f"ndim={arr['ndim']}, size={arr['size']}, nbytes={arr['nbytes']}"
            )

            if arr["min"] is not None:
                line += f", min={arr['min']:.6f}, max={arr['max']:.6f}, mean={arr['mean']:.6f}"

            print(line)

            if arr["object_preview"] is not None:
                print(f"      object preview: {arr['object_preview']}")

        print()

    if len(results) > max_files_to_show:
        print(f"... {len(results) - max_files_to_show} more files omitted")
        print()


def save_json(results, summary, json_path: Path):
    payload = {
        "summary": summary,
        "results": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_csv(results, csv_path: Path):
    rows = []
    for r in results:
        if r["error"] is not None:
            rows.append(
                {
                    "file": r["file"],
                    "file_size_bytes": r["file_size_bytes"],
                    "key": None,
                    "shape": None,
                    "dtype": None,
                    "ndim": None,
                    "size": None,
                    "nbytes": None,
                    "min": None,
                    "max": None,
                    "mean": None,
                    "error": r["error"],
                }
            )
            continue

        for arr in r["arrays"]:
            rows.append(
                {
                    "file": r["file"],
                    "file_size_bytes": r["file_size_bytes"],
                    "key": arr.get("key"),
                    "shape": arr.get("shape"),
                    "dtype": arr.get("dtype"),
                    "ndim": arr.get("ndim"),
                    "size": arr.get("size"),
                    "nbytes": arr.get("nbytes"),
                    "min": arr.get("min"),
                    "max": arr.get("max"),
                    "mean": arr.get("mean"),
                    "error": arr.get("array_error"),
                }
            )

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "file_size_bytes",
                "key",
                "shape",
                "dtype",
                "ndim",
                "size",
                "nbytes",
                "min",
                "max",
                "mean",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Inspect many .npz files in a dataset folder.")
    parser.add_argument("--root", type=str, required=True, help="Root folder containing .npz files")
    parser.add_argument("--sample", type=int, default=20, help="Number of files to inspect (ignored if --all)")
    parser.add_argument("--all", action="store_true", help="Inspect all .npz files")
    parser.add_argument("--allow-pickle", action="store_true", help="Use allow_pickle=True in np.load")
    parser.add_argument("--save-json", type=str, default=None, help="Optional path to save JSON summary")
    parser.add_argument("--save-csv", type=str, default=None, help="Optional path to save CSV summary")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root path does not exist: {root}")
        return

    npz_files = collect_npz_files(root)
    if not npz_files:
        print(f"No .npz files found under: {root}")
        return

    print(f"Found {len(npz_files)} .npz files under {root}")

    if args.all:
        target_files = npz_files
    else:
        target_files = npz_files[: args.sample]

    print(f"Inspecting {len(target_files)} files...")
    print()

    results = []
    for idx, npz_path in enumerate(target_files, 1):
        print(f"[{idx}/{len(target_files)}] {npz_path}")
        results.append(inspect_npz_file(npz_path, allow_pickle=args.allow_pickle))

    summary = summarize_results(results)
    print()
    print_human_readable(results, summary, max_files_to_show=min(20, len(results)))

    if args.save_json:
        save_json(results, summary, Path(args.save_json))
        print(f"Saved JSON summary to: {args.save_json}")

    if args.save_csv:
        save_csv(results, Path(args.save_csv))
        print(f"Saved CSV summary to: {args.save_csv}")


if __name__ == "__main__":
    main()