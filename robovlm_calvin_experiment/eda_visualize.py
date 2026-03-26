# This script visualizes task-annotated segments from the CALVIN debug dataset.
# For each language annotation, it loads the start / middle / end frames of the
# corresponding segment and saves a summary image with both rgb_static and
# rgb_gripper views, so the train/validation data structure can be inspected easily.
'''
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_lang_ann(lang_path):
    data = np.load(lang_path, allow_pickle=True).item()
    anns = data["language"]["ann"]
    tasks = data["language"]["task"]
    indx = data["info"]["indx"]
    return anns, tasks, indx


def load_npz_image(root, frame_idx, key="rgb_static"):
    npz_path = Path(root) / f"episode_{frame_idx:07d}.npz"
    with np.load(npz_path, allow_pickle=True) as data:
        return data[key]


def save_task_frames(split_root, split_name, out_dir):
    split_root = Path(split_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lang_path = split_root / "lang_annotations" / "auto_lang_ann.npy"
    anns, tasks, indx = load_lang_ann(lang_path)

    for i, (ann, task, (start, end)) in enumerate(zip(anns, tasks, indx)):
        mid = (start + end) // 2
        frame_ids = [start, mid, end]

        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        fig.suptitle(
            f"[{split_name}] idx={i} | task={task}\nann={ann}\nsegment=({start}, {end})",
            fontsize=11
        )

        for col, frame_idx in enumerate(frame_ids):
            static_img = load_npz_image(split_root, frame_idx, key="rgb_static")
            gripper_img = load_npz_image(split_root, frame_idx, key="rgb_gripper")

            axes[0, col].imshow(static_img)
            axes[0, col].set_title(f"rgb_static\nframe {frame_idx}")
            axes[0, col].axis("off")

            axes[1, col].imshow(gripper_img)
            axes[1, col].set_title(f"rgb_gripper\nframe {frame_idx}")
            axes[1, col].axis("off")

        plt.tight_layout()

        save_path = out_dir / f"{split_name}_{i:02d}_{task}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"saved: {save_path}")


if __name__ == "__main__":
    train_root = "/home/yewon/RoboVLMs/calvin/dataset/calvin_debug_dataset/training"
    val_root = "/home/yewon/RoboVLMs/calvin/dataset/calvin_debug_dataset/validation"
    out_dir = "/home/yewon/RoboVLMs/eda_frames"

    save_task_frames(train_root, "train", out_dir)
    save_task_frames(val_root, "val", out_dir)


# This script compares train and validation task distributions by plotting
# total frame counts and estimated window-based sample counts for each task.
# It helps visualize task imbalance and how many training samples each task
# can generate under the current window_size setting.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_auto_lang_ann(path):
    data = np.load(path, allow_pickle=True).item()
    anns = data["language"]["ann"]
    tasks = data["language"]["task"]
    indx = data["info"]["indx"]
    return anns, tasks, indx


def compute_window_counts(tasks, indx, window_size):
    counts = defaultdict(int)
    frame_counts = defaultdict(int)

    for task, (start, end) in zip(tasks, indx):
        seg_len = end - start + 1
        num_windows = max(0, seg_len - window_size + 1)

        counts[task] += num_windows
        frame_counts[task] += seg_len

    return dict(counts), dict(frame_counts)


def plot_grouped_bar(train_counts, val_counts, title, ylabel, save_path):
    all_tasks = sorted(set(train_counts.keys()) | set(val_counts.keys()))

    train_vals = [train_counts.get(task, 0) for task in all_tasks]
    val_vals = [val_counts.get(task, 0) for task in all_tasks]

    x = np.arange(len(all_tasks))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, train_vals, width=width, label="Train")
    plt.bar(x + width / 2, val_vals, width=width, label="Validation")

    plt.xticks(x, all_tasks, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"saved: {save_path}")


def main():
    train_root = Path("/home/yewon/RoboVLMs/calvin/dataset/calvin_debug_dataset/training")
    val_root = Path("/home/yewon/RoboVLMs/calvin/dataset/calvin_debug_dataset/validation")
    out_dir = Path("/home/yewon/RoboVLMs/eda/eda_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    window_size = 4  # 네 현재 config 기준

    train_anns, train_tasks, train_indx = load_auto_lang_ann(
        train_root / "lang_annotations" / "auto_lang_ann.npy"
    )
    val_anns, val_tasks, val_indx = load_auto_lang_ann(
        val_root / "lang_annotations" / "auto_lang_ann.npy"
    )

    train_window_counts, train_frame_counts = compute_window_counts(
        train_tasks, train_indx, window_size
    )
    val_window_counts, val_frame_counts = compute_window_counts(
        val_tasks, val_indx, window_size
    )

    print("\n[Estimated window sample counts]")
    print("Train:", train_window_counts)
    print("Val  :", val_window_counts)

    print("\n[Total frame counts by task]")
    print("Train:", train_frame_counts)
    print("Val  :", val_frame_counts)

    plot_grouped_bar(
        train_window_counts,
        val_window_counts,
        title=f"Estimated Window Sample Counts by Task (window_size={window_size})",
        ylabel="Estimated # of window samples",
        save_path=out_dir / "window_sample_counts_by_task.png",
    )

    plot_grouped_bar(
        train_frame_counts,
        val_frame_counts,
        title="Total Frame Counts by Task",
        ylabel="Total # of frames",
        save_path=out_dir / "frame_counts_by_task.png",
    )


if __name__ == "__main__":
    main()
'''

# This script visualizes train/validation distribution differences for rel_actions
# and robot_obs. It plots per-dimension histograms and boxplots so that action
# variability and state distribution shifts can be compared more intuitively.
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def find_npz_files(root):
    return sorted(Path(root).glob("episode_*.npz"))


def sample_npz_files(npz_files, max_samples):
    if len(npz_files) <= max_samples:
        return npz_files
    idxs = np.linspace(0, len(npz_files) - 1, max_samples, dtype=int)
    return [npz_files[i] for i in idxs]


def load_arrays(root, key, max_samples=300):
    files = find_npz_files(root)
    files = sample_npz_files(files, max_samples)

    arrs = []
    for f in files:
        with np.load(f, allow_pickle=True) as data:
            if key in data:
                arrs.append(data[key].astype(np.float32))
    return np.stack(arrs, axis=0)


def plot_hist_per_dim(train_arr, val_arr, title_prefix, save_path, bins=30):
    dims = train_arr.shape[1]
    ncols = 3
    nrows = int(np.ceil(dims / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for d in range(dims):
        ax = axes[d]
        ax.hist(train_arr[:, d], bins=bins, alpha=0.6, density=False, label="Train")
        ax.hist(val_arr[:, d], bins=bins, alpha=0.6, density=False, label="Val")
        ax.set_title(f"{title_prefix} dim {d}")
        ax.legend()

    for i in range(dims, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {save_path}")


def plot_selected_robot_obs_dims(train_arr, val_arr, dims, save_path, bins=30):
    n = len(dims)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, d in enumerate(dims):
        ax = axes[i]
        ax.hist(train_arr[:, d], bins=bins, alpha=0.6, label="Train")
        ax.hist(val_arr[:, d], bins=bins, alpha=0.6, label="Val")
        ax.set_title(f"robot_obs dim {d}")
        ax.legend()

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {save_path}")


def plot_boxplot(train_arr, val_arr, title, save_path):
    dims = train_arr.shape[1]
    positions_train = np.arange(dims) * 2.0
    positions_val = positions_train + 0.7

    plt.figure(figsize=(16, 6))
    plt.boxplot(
        [train_arr[:, d] for d in range(dims)],
        positions=positions_train,
        widths=0.5,
        patch_artist=False,
        manage_ticks=False,
    )
    plt.boxplot(
        [val_arr[:, d] for d in range(dims)],
        positions=positions_val,
        widths=0.5,
        patch_artist=False,
        manage_ticks=False,
    )

    tick_positions = positions_train + 0.35
    tick_labels = [f"d{d}" for d in range(dims)]
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.title(title)
    plt.xlabel("Dimension")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {save_path}")


if __name__ == "__main__":
    train_root = "/home/yewon/RoboVLMs/calvin/dataset/calvin_debug_dataset/training"
    val_root = "/home/yewon/RoboVLMs/calvin/dataset/calvin_debug_dataset/validation"
    out_dir = Path("/home/yewon/RoboVLMs/eda/eda_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_actions = load_arrays(train_root, "rel_actions", max_samples=300)
    val_actions = load_arrays(val_root, "rel_actions", max_samples=300)

    train_robot = load_arrays(train_root, "robot_obs", max_samples=300)
    val_robot = load_arrays(val_root, "robot_obs", max_samples=300)

    plot_hist_per_dim(
        train_actions,
        val_actions,
        title_prefix="rel_actions",
        save_path=out_dir / "rel_actions_histograms.png",
        bins=30,
    )

    selected_robot_dims = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    plot_selected_robot_obs_dims(
        train_robot,
        val_robot,
        dims=selected_robot_dims,
        save_path=out_dir / "robot_obs_selected_histograms.png",
        bins=30,
    )

    plot_boxplot(
        train_actions,
        val_actions,
        title="rel_actions distribution by dimension (Train vs Val)",
        save_path=out_dir / "rel_actions_boxplot.png",
    )

    plot_boxplot(
        train_robot,
        val_robot,
        title="robot_obs distribution by dimension (Train vs Val)",
        save_path=out_dir / "robot_obs_boxplot.png",
    )