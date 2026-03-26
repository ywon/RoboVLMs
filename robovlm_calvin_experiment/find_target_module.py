import json
from pathlib import Path
import os
import importlib
import torch
import torch.distributed as dist

from robovlms.train.base_trainer import BaseTrainer


def deep_update(d1, d2):
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1 and isinstance(d1[k], dict):
            deep_update(d1[k], v)
        else:
            d1[k] = v
    return d1


def load_config(config_file):
    with open(config_file, "r", encoding="utf-8") as f:
        _config = json.load(f)
    config = {}
    if _config.get("parent", None):
        deep_update(config, load_config(_config["parent"]))
    deep_update(config, _config)
    return config


def maybe_override_dataset_path(variant):
    for split in ["train_dataset", "val_dataset"]:
        if split not in variant or variant[split] is None:
            continue
        if variant.get("data_dir") is not None:
            variant[split]["data_dir"] = variant["data_dir"]
        if variant.get("annotation_file") is not None:
            variant[split]["annotation_file"] = variant["annotation_file"]
        if variant.get("data_subfolder") is not None:
            variant[split]["data_subfolder"] = variant["data_subfolder"]
        if variant.get("task_num") is not None:
            variant[split]["task_num"] = variant["task_num"]
        if variant.get("seq_len") is not None:
            variant[split]["seq_len"] = variant["seq_len"]
    return variant


def find_lora_target_root(model):
    candidates = []

    if hasattr(model, "model"):
        candidates.append(("model.model", model.model))

    if hasattr(model, "model"):
        inner = model.model
        for attr in ["language_model", "llm", "lm", "model", "backbone", "text_model", "module"]:
            if hasattr(inner, attr):
                candidates.append((f"model.model.{attr}", getattr(inner, attr)))

    preferred_order = [
        "model.model.language_model",
        "model.model.llm",
        "model.model.lm",
        "model.model.model",
        "model.model.backbone",
        "model.model.text_model",
        "model.model",
    ]

    for preferred in preferred_order:
        for name, module in candidates:
            if name == preferred:
                return name, module

    raise ValueError("Could not find a suitable root module for LoRA.")


def main():
    config_path = "/home/yewon/RoboVLMs/pretrained/robovlms/configs/kosmos_ph_finetune.json"          
    model_load_path = "/home/yewon/RoboVLMs/pretrained/robovlms/checkpoints/kosmos_ph_oxe-pretrain.pt"                    

    variant = load_config(config_path)
    variant = maybe_override_dataset_path(variant)

    model = BaseTrainer.from_checkpoint(
        model_load_path or variant.get("model_load_path", None),
        variant.get("model_load_source", "torch"),
        variant,
    )

    root_name, root_module = find_lora_target_root(model)

    print(f"\n[INFO] LoRA root: {root_name}")
    print(f"[INFO] root type: {type(root_module)}")

    print("\n[INFO] ===== full linear module names =====")
    linear_suffixes = set()
    for name, module in root_module.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(name)
            linear_suffixes.add(name.split(".")[-1])

    print("\n[INFO] ===== candidate target_modules =====")
    for name in sorted(linear_suffixes):
        print(name)


if __name__ == "__main__":
    main()