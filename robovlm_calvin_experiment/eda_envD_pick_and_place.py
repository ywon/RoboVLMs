import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import os

ROOT = Path("/mnt/hdd/calvin/calvin_ABCD_D/task_ABCD_D")
TRAIN_DIR = ROOT / "training"
VAL_DIR = ROOT / "validation"
SCENE_INFO_PATH = TRAIN_DIR / "scene_info.npy"

TARGET_TASKS = {
    "lift_blue_block_table",
    "lift_pink_block_table",
    "lift_red_block_table",
    "place_in_drawer",
}

OUT_DIR = Path("./eda_sceneD_4tasks")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_scene_ranges(scene_info_path):
    obj = np.load(scene_info_path, allow_pickle=True).item()
    return obj


def get_scene_name(frame_idx, scene_ranges):
    for scene_name, (start, end) in scene_ranges.items():
        if start <= frame_idx <= end:
            return scene_name
    return "UNKNOWN"


def load_lang_ann(split_dir):
    path = split_dir / "lang_annotations" / "auto_lang_ann.npy"
    obj = np.load(path, allow_pickle=True).item()
    tasks = obj["language"]["task"]
    anns = obj["language"]["ann"]
    indx = obj["info"]["indx"]
    return tasks, anns, indx


def filter_segments(split_dir, scene_ranges, target_tasks, target_scene="calvin_scene_D"):
    tasks, anns, indx = load_lang_ann(split_dir)
    filtered = []

    for seg_id, (task, ann, (start, end)) in enumerate(zip(tasks, anns, indx)):
        if task not in target_tasks:
            continue

        start_scene = get_scene_name(start, scene_ranges)
        end_scene = get_scene_name(end, scene_ranges)

        if start_scene != target_scene or end_scene != target_scene:
            continue

        filtered.append({
            "seg_id": seg_id,
            "task": task,
            "instruction": ann,
            "start": int(start),
            "end": int(end),
            "length": int(end - start + 1),
            "scene": target_scene,
        })

    return filtered


def summarize_segments(name, segments):
    print(f"\n========== {name} ==========")
    print("num_segments:", len(segments))

    task_seg_counts = Counter([x["task"] for x in segments])
    task_frame_counts = defaultdict(int)
    task_lengths = defaultdict(list)

    for x in segments:
        task_frame_counts[x["task"]] += x["length"]
        task_lengths[x["task"]].append(x["length"])

    print("\n[Task segment counts]")
    for task in sorted(task_seg_counts.keys()):
        print(f"{task}: {task_seg_counts[task]}")

    print("\n[Task frame counts]")
    for task in sorted(task_frame_counts.keys()):
        print(f"{task}: {task_frame_counts[task]}")

    print("\n[Task length stats]")
    for task in sorted(task_lengths.keys()):
        arr = np.array(task_lengths[task])
        print(
            f"{task}: "
            f"mean={arr.mean():.2f}, std={arr.std():.2f}, "
            f"min={arr.min()}, max={arr.max()}, median={np.median(arr):.1f}"
        )

    return task_seg_counts, task_frame_counts, task_lengths


def plot_bar(count_dict, title, ylabel, save_path):
    tasks = sorted(count_dict.keys())
    vals = [count_dict[t] for t in tasks]

    plt.figure(figsize=(10, 5))
    plt.bar(tasks, vals)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_representative_frames(split_dir, segments, split_name, max_per_task=3):
    out_dir = OUT_DIR / f"{split_name}_sample_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for seg in segments:
        grouped[seg["task"]].append(seg)

    for task, segs in grouped.items():
        segs = segs[:max_per_task]
        for seg in segs:
            start = seg["start"]
            end = seg["end"]
            mid = (start + end) // 2
            frame_ids = [start, mid, end]

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle(
                f"{split_name} | {task}\n"
                f"seg_id={seg['seg_id']} | range=({start}, {end})\n"
                f"instruction={seg['instruction']}"
            )

            for ax, fid, label in zip(axes, frame_ids, ["start", "mid", "end"]):
                npz_path = split_dir / f"episode_{fid:07d}.npz"
                with np.load(npz_path, allow_pickle=True) as data:
                    img = data["rgb_static"]
                ax.imshow(img)
                ax.set_title(f"{label}\nframe {fid}")
                ax.axis("off")

            plt.tight_layout()
            save_path = out_dir / f"{task}_seg{seg['seg_id']:05d}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()


def main():
    scene_ranges = load_scene_ranges(SCENE_INFO_PATH)

    train_segments = filter_segments(TRAIN_DIR, scene_ranges, TARGET_TASKS, "calvin_scene_D")
    val_segments = filter_segments(VAL_DIR, scene_ranges, TARGET_TASKS, "calvin_scene_D")

    train_seg_counts, train_frame_counts, train_lengths = summarize_segments("TRAIN", train_segments)
    val_seg_counts, val_frame_counts, val_lengths = summarize_segments("VAL", val_segments)

    plot_bar(
        train_seg_counts,
        "Train Segment Counts (Scene D, 4 Tasks)",
        "Num Segments",
        OUT_DIR / "train_segment_counts.png",
    )
    plot_bar(
        val_seg_counts,
        "Validation Segment Counts (Scene D, 4 Tasks)",
        "Num Segments",
        OUT_DIR / "val_segment_counts.png",
    )
    plot_bar(
        train_frame_counts,
        "Train Frame Counts (Scene D, 4 Tasks)",
        "Num Frames",
        OUT_DIR / "train_frame_counts.png",
    )
    plot_bar(
        val_frame_counts,
        "Validation Frame Counts (Scene D, 4 Tasks)",
        "Num Frames",
        OUT_DIR / "val_frame_counts.png",
    )

    save_representative_frames(TRAIN_DIR, train_segments, "train", max_per_task=3)
    save_representative_frames(VAL_DIR, val_segments, "val", max_per_task=3)

    print("\nSaved EDA outputs to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()