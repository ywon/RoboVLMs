import os
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np


def load_auto_lang_ann(path):
    data = np.load(path, allow_pickle=True).item()
    anns = data["language"]["ann"]
    tasks = data["language"]["task"]
    embs = data["language"]["emb"]
    indx = data["info"]["indx"]
    return anns, tasks, embs, indx


def load_ep_meta(root):
    ep_start_end = np.load(Path(root) / "ep_start_end_ids.npy", allow_pickle=True)
    ep_lens = np.load(Path(root) / "ep_lens.npy", allow_pickle=True)
    if isinstance(ep_lens, np.ndarray) and ep_lens.shape == ():
        ep_lens = ep_lens.item()
    return ep_start_end, ep_lens


def summarize_annotations(split_name, anns, tasks, indx, window_size):
    print("\n" + "=" * 100)
    print(f"[{split_name}] LANGUAGE / TASK SUMMARY")
    print("=" * 100)

    print(f"num annotations: {len(anns)}")
    print(f"num unique tasks: {len(set(tasks))}")
    print(f"unique tasks: {sorted(set(tasks))}")
    print()

    task_counter = Counter(tasks)
    print("task annotation counts:")
    for task, cnt in sorted(task_counter.items()):
        print(f"  {task:30s}: {cnt}")

    print("\nannotation table:")
    print(f"{'idx':>3} | {'task':30s} | {'start':>8} | {'end':>8} | {'len':>5} | ann")
    print("-" * 100)

    window_counts = defaultdict(int)

    for i, (ann, task, (s, e)) in enumerate(zip(anns, tasks, indx)):
        seg_len = e - s + 1
        num_windows = max(0, seg_len - window_size + 1)
        window_counts[task] += num_windows
        print(f"{i:>3} | {task:30s} | {s:>8} | {e:>8} | {seg_len:>5} | {ann}")

    print("\nestimated window sample counts by task")
    print(f"(assuming one sample per valid start index with window_size={window_size})")
    for task, cnt in sorted(window_counts.items()):
        print(f"  {task:30s}: {cnt}")

    ann_lengths = [len(a.split()) for a in anns]
    print("\ninstruction length stats (word count)")
    print(f"  min={min(ann_lengths)}, max={max(ann_lengths)}, mean={np.mean(ann_lengths):.2f}")

    print("\nunique instructions per task:")
    task_to_anns = defaultdict(list)
    for ann, task in zip(anns, tasks):
        task_to_anns[task].append(ann)

    for task in sorted(task_to_anns):
        uniq = list(dict.fromkeys(task_to_anns[task]))
        print(f"  [{task}]")
        for u in uniq:
            print(f"    - {u}")

    return {
        "task_counter": task_counter,
        "window_counts": dict(window_counts),
        "ann_lengths": ann_lengths,
        "unique_tasks": sorted(set(tasks)),
    }


def find_npz_files(root):
    return sorted(Path(root).glob("episode_*.npz"))


def sample_npz_files(npz_files, max_samples):
    if len(npz_files) <= max_samples:
        return npz_files
    idxs = np.linspace(0, len(npz_files) - 1, max_samples, dtype=int)
    return [npz_files[i] for i in idxs]


def safe_image_stats(img):
    img = img.astype(np.float32)
    return {
        "mean": float(img.mean()),
        "std": float(img.std()),
        "min": float(img.min()),
        "max": float(img.max()),
    }


def collect_npz_stats(split_name, root, max_samples=200):
    print("\n" + "=" * 100)
    print(f"[{split_name}] NPZ CONTENT SUMMARY")
    print("=" * 100)

    npz_files = find_npz_files(root)
    print(f"total npz files found: {len(npz_files)}")
    if len(npz_files) == 0:
        return None

    target_files = sample_npz_files(npz_files, max_samples)
    print(f"sampling {len(target_files)} files for stats")

    action_list = []
    robot_obs_list = []
    rgb_static_means = []
    rgb_gripper_means = []
    rgb_static_stds = []
    rgb_gripper_stds = []

    first_keys = None

    for f in target_files:
        with np.load(f, allow_pickle=True) as data:
            keys = list(data.keys())
            if first_keys is None:
                first_keys = keys

            if "rel_actions" in data:
                action_list.append(data["rel_actions"].astype(np.float32))

            if "robot_obs" in data:
                robot_obs_list.append(data["robot_obs"].astype(np.float32))

            if "rgb_static" in data:
                st = safe_image_stats(data["rgb_static"])
                rgb_static_means.append(st["mean"])
                rgb_static_stds.append(st["std"])

            if "rgb_gripper" in data:
                st = safe_image_stats(data["rgb_gripper"])
                rgb_gripper_means.append(st["mean"])
                rgb_gripper_stds.append(st["std"])

    print("example keys from npz:")
    print(first_keys)

    results = {}

    if action_list:
        actions = np.stack(action_list, axis=0)
        results["actions"] = actions
        print("\nrel_actions stats by dimension:")
        for d in range(actions.shape[1]):
            col = actions[:, d]
            print(
                f"  dim {d}: min={col.min():.4f}, max={col.max():.4f}, "
                f"mean={col.mean():.4f}, std={col.std():.4f}"
            )

    if robot_obs_list:
        robot_obs = np.stack(robot_obs_list, axis=0)
        results["robot_obs"] = robot_obs
        print("\nrobot_obs stats by dimension:")
        for d in range(robot_obs.shape[1]):
            col = robot_obs[:, d]
            print(
                f"  dim {d}: min={col.min():.4f}, max={col.max():.4f}, "
                f"mean={col.mean():.4f}, std={col.std():.4f}"
            )

    if rgb_static_means:
        print("\nrgb_static brightness stats:")
        print(
            f"  mean_of_means={np.mean(rgb_static_means):.2f}, "
            f"std_of_means={np.std(rgb_static_means):.2f}, "
            f"mean_of_stds={np.mean(rgb_static_stds):.2f}"
        )

    if rgb_gripper_means:
        print("\nrgb_gripper brightness stats:")
        print(
            f"  mean_of_means={np.mean(rgb_gripper_means):.2f}, "
            f"std_of_means={np.std(rgb_gripper_means):.2f}, "
            f"mean_of_stds={np.mean(rgb_gripper_stds):.2f}"
        )

    results["rgb_static_means"] = rgb_static_means
    results["rgb_gripper_means"] = rgb_gripper_means
    results["rgb_static_stds"] = rgb_static_stds
    results["rgb_gripper_stds"] = rgb_gripper_stds

    return results


def compare_splits(train_summary, val_summary, train_npz, val_npz):
    print("\n" + "=" * 100)
    print("TRAIN / VAL COMPARISON")
    print("=" * 100)

    train_tasks = set(train_summary["unique_tasks"])
    val_tasks = set(val_summary["unique_tasks"])

    print("task overlap:")
    print("  only in train:", sorted(train_tasks - val_tasks))
    print("  only in val  :", sorted(val_tasks - train_tasks))
    print("  shared       :", sorted(train_tasks & val_tasks))

    print("\nannotation count comparison:")
    all_tasks = sorted(train_tasks | val_tasks)
    for task in all_tasks:
        tr = train_summary["task_counter"].get(task, 0)
        va = val_summary["task_counter"].get(task, 0)
        print(f"  {task:30s}: train={tr:3d}, val={va:3d}")

    print("\nestimated window count comparison:")
    for task in all_tasks:
        tr = train_summary["window_counts"].get(task, 0)
        va = val_summary["window_counts"].get(task, 0)
        print(f"  {task:30s}: train={tr:3d}, val={va:3d}")

    print("\ninstruction length comparison:")
    print(
        f"  train mean={np.mean(train_summary['ann_lengths']):.2f}, "
        f"val mean={np.mean(val_summary['ann_lengths']):.2f}"
    )

    if train_npz and val_npz and "actions" in train_npz and "actions" in val_npz:
        print("\nrel_actions distribution comparison:")
        ta = train_npz["actions"]
        va = val_npz["actions"]
        dims = min(ta.shape[1], va.shape[1])
        for d in range(dims):
            print(
                f"  dim {d}: "
                f"train(mean={ta[:, d].mean():.4f}, std={ta[:, d].std():.4f}) | "
                f"val(mean={va[:, d].mean():.4f}, std={va[:, d].std():.4f})"
            )

    if train_npz and val_npz and "robot_obs" in train_npz and "robot_obs" in val_npz:
        print("\nrobot_obs distribution comparison:")
        tro = train_npz["robot_obs"]
        vro = val_npz["robot_obs"]
        dims = min(tro.shape[1], vro.shape[1])
        for d in range(dims):
            print(
                f"  dim {d}: "
                f"train(mean={tro[:, d].mean():.4f}, std={tro[:, d].std():.4f}) | "
                f"val(mean={vro[:, d].mean():.4f}, std={vro[:, d].std():.4f})"
            )

    if train_npz and val_npz:
        if train_npz["rgb_static_means"] and val_npz["rgb_static_means"]:
            print("\nrgb_static brightness comparison:")
            print(
                f"  train mean={np.mean(train_npz['rgb_static_means']):.2f}, "
                f"val mean={np.mean(val_npz['rgb_static_means']):.2f}"
            )

        if train_npz["rgb_gripper_means"] and val_npz["rgb_gripper_means"]:
            print("rgb_gripper brightness comparison:")
            print(
                f"  train mean={np.mean(train_npz['rgb_gripper_means']):.2f}, "
                f"val mean={np.mean(val_npz['rgb_gripper_means']):.2f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--val_root", type=str, required=True)
    parser.add_argument("--window_size", type=int, default=12)
    parser.add_argument("--sample_npz_per_split", type=int, default=200)
    args = parser.parse_args()

    train_root = Path(args.train_root)
    val_root = Path(args.val_root)

    train_lang = train_root / "lang_annotations" / "auto_lang_ann.npy"
    val_lang = val_root / "lang_annotations" / "auto_lang_ann.npy"

    train_anns, train_tasks, train_embs, train_indx = load_auto_lang_ann(train_lang)
    val_anns, val_tasks, val_embs, val_indx = load_auto_lang_ann(val_lang)

    train_ep_start_end, train_ep_lens = load_ep_meta(train_root)
    val_ep_start_end, val_ep_lens = load_ep_meta(val_root)

    print("=" * 100)
    print("EPISODE META")
    print("=" * 100)
    print("[train] ep_start_end_ids:", train_ep_start_end)
    print("[train] ep_lens:", train_ep_lens)
    print("[val]   ep_start_end_ids:", val_ep_start_end)
    print("[val]   ep_lens:", val_ep_lens)

    train_summary = summarize_annotations(
        "train", train_anns, train_tasks, train_indx, args.window_size
    )
    val_summary = summarize_annotations(
        "val", val_anns, val_tasks, val_indx, args.window_size
    )

    train_npz = collect_npz_stats(
        "train", train_root, max_samples=args.sample_npz_per_split
    )
    val_npz = collect_npz_stats(
        "val", val_root, max_samples=args.sample_npz_per_split
    )

    compare_splits(train_summary, val_summary, train_npz, val_npz)


if __name__ == "__main__":
    main()