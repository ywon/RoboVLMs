import functools
from pathlib import Path
from collections import Counter

import torch

from robovlms.data.calvin_dataset import DiskCalvinDataset
from robovlms.data.data_utils import preprocess_image
from robovlms.train.base_trainer import BaseTrainer
from robovlms.utils.config_utils import load_config


def build_model_and_image_fn(config_path: str):
    configs = load_config(config_path)

    model_load_path = configs.get("model_load_path", None)
    if model_load_path is None:
        raise ValueError("config에 model_load_path가 필요함")

    model = BaseTrainer.from_checkpoint(
        model_load_path,
        configs.get("model_load_source", "torch"),
        configs,
    )

    image_preprocess = model.model.image_processor
    tokenizer = model.model.tokenizer

    image_fn = functools.partial(
        preprocess_image,
        image_processor=image_preprocess,
        model_type=configs["model"],
    )

    return configs, tokenizer, image_fn


def build_dataset(dataset_cfg, tokenizer, image_fn, window_size, fwd_pred_next_n):
    ds = DiskCalvinDataset(
        image_fn=image_fn,
        tokenizer=tokenizer,
        data_dir=dataset_cfg["data_dir"],
        key="lang",
        model_name=dataset_cfg.get("model_name", "kosmos"),
        rgb_pad=dataset_cfg.get("rgb_pad", -1),
        gripper_pad=dataset_cfg.get("gripper_pad", -1),
        use_segment_csv=dataset_cfg.get("use_segment_csv", False),
        segment_csv=dataset_cfg.get("segment_csv", None),
        window_size=window_size,
        fwd_pred_next_n=fwd_pred_next_n,
        save_format=dataset_cfg.get("save_format", "npz"),
    )
    return ds


def print_task_counts(name, ds):
    if not hasattr(ds, "lang_task") or not hasattr(ds, "lang_lookup"):
        print(f"\n--- {name} task counts ---")
        print("task count 정보를 찾을 수 없음")
        return

    print(f"\n--- {name} task counts (segment 기준) ---")
    segment_counter = Counter(ds.lang_task)
    for task, count in sorted(segment_counter.items()):
        print(f"{task}: {count}")

    print(f"\n--- {name} task counts (sliding-window sample 기준) ---")
    sample_counter = Counter()
    for lang_idx in ds.lang_lookup:
        task = ds.lang_task[lang_idx]
        sample_counter[task] += 1

    for task, count in sorted(sample_counter.items()):
        print(f"{task}: {count}")


def inspect_dataset(name, ds):
    print(f"\n===== {name} DATASET =====")
    print("len(dataset) =", len(ds))

    if hasattr(ds, "lang_ann"):
        print("num segments =", len(ds.lang_ann))
        print("first instruction =", ds.lang_ann[0] if len(ds.lang_ann) > 0 else None)

    if hasattr(ds, "lang_task"):
        print("first task =", ds.lang_task[0] if len(ds.lang_task) > 0 else None)

    if hasattr(ds, "episode_lookup"):
        print("first 5 episode_lookup =", ds.episode_lookup[:5])

    print_task_counts(name, ds)

    if len(ds) == 0:
        raise RuntimeError(f"{name} dataset length is 0")

    sample = ds[0]
    print("\n--- sample[0] keys ---")
    print(sample.keys())

    print("\n--- sample[0] shapes/types ---")
    print("lang:", sample["lang"])
    print("actions:", sample["actions"].shape if hasattr(sample["actions"], "shape") else type(sample["actions"]))
    print("action_mask:", sample["action_mask"].shape if hasattr(sample["action_mask"], "shape") else type(sample["action_mask"]))
    print("image_mask:", sample["image_mask"].shape if hasattr(sample["image_mask"], "shape") else type(sample["image_mask"]))
    print("rgb_static:", sample["rgb_obs"]["rgb_static"].shape)
    print("rgb_gripper:", sample["rgb_obs"]["rgb_gripper"].shape)
    print("robot_obs:", sample["robot_obs"].shape if hasattr(sample["robot_obs"], "shape") else type(sample["robot_obs"]))

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=ds.collater,
    )

    batch = next(iter(loader))
    print("\n--- one batch keys ---")
    print(batch.keys())

    print("\n--- one batch shapes ---")
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(k, tuple(v.shape))
        elif v is None:
            print(k, None)
        elif isinstance(v, list):
            print(k, f"list(len={len(v)})")
        else:
            print(k, type(v))


def main():
    config_path = "/home/yewon/RoboVLMs/pretrained/robovlms/configs/kosmos_ph_finetune.json"
    configs, tokenizer, image_fn = build_model_and_image_fn(config_path)

    train_ds = build_dataset(
        configs["train_dataset"],
        tokenizer,
        image_fn,
        window_size=configs["window_size"],
        fwd_pred_next_n=configs["fwd_pred_next_n"],
    )

    val_ds = build_dataset(
        configs["val_dataset"],
        tokenizer,
        image_fn,
        window_size=configs["window_size"],
        fwd_pred_next_n=configs["fwd_pred_next_n"],
    )

    inspect_dataset("TRAIN", train_ds)
    inspect_dataset("VAL", val_ds)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()