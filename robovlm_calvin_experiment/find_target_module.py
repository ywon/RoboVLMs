import json
from pathlib import Path
import os
import importlib
import torch
import torch.nn as nn
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


def summarize_trainable_params(model):
    total_params = 0
    trainable_params = 0

    trainable_names = []
    vision_trainable = []
    text_trainable = []
    other_trainable = []

    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n

        if param.requires_grad:
            trainable_params += n
            trainable_names.append(name)

            lname = name.lower()
            if "vision" in lname or "image" in lname or "visual" in lname:
                vision_trainable.append(name)
            elif "text" in lname or "language" in lname or "llm" in lname or "lm" in lname:
                text_trainable.append(name)
            else:
                other_trainable.append(name)

    print("\n[INFO] ===== parameter summary =====")
    print(f"total params:      {total_params:,}")
    print(f"trainable params:  {trainable_params:,}")
    print(f"trainable ratio:   {100.0 * trainable_params / total_params:.4f}%")

    print("\n[INFO] ===== trainable counts by rough group =====")
    print(f"vision-like trainable params: {len(vision_trainable)} tensors")
    print(f"text-like trainable params:   {len(text_trainable)} tensors")
    print(f"other trainable params:       {len(other_trainable)} tensors")

    print("\n[INFO] ===== first 100 trainable parameter names =====")
    for name in trainable_names[:100]:
        print(name)

    print("\n[INFO] ===== vision-related trainable parameter names =====")
    if vision_trainable:
        for name in vision_trainable[:200]:
            print(name)
    else:
        print("(none)")

    print("\n[INFO] ===== text-related trainable parameter names =====")
    if text_trainable:
        for name in text_trainable[:200]:
            print(name)
    else:
        print("(none)")

    print("\n[INFO] ===== other trainable parameter names =====")
    if other_trainable:
        for name in other_trainable[:200]:
            print(name)
    else:
        print("(none)")


def inspect_linear_modules(root_module):
    print("\n[INFO] ===== full linear module names =====")
    candidate_targets = set()

    for name, module in root_module.named_modules():
        if isinstance(module, nn.Linear):
            print(name)

            parts = name.split(".")
            if parts[-1] == "base_layer" and len(parts) >= 2:
                candidate_targets.add(parts[-2])
            elif "lora_A" in parts or "lora_B" in parts:
                continue
            else:
                candidate_targets.add(parts[-1])

    print("\n[INFO] ===== candidate target_modules =====")
    for name in sorted(candidate_targets):
        print(name)


def main():
    config_path = "/home/yewon/RoboVLMs/pretrained/robovlms/configs/kosmos_ph_finetune.json"
    model_load_path = "/home/yewon/RoboVLMs/pretrained/robovlms/checkpoints/kosmos_ph_oxe-pretrain.pt"

    variant = load_config(config_path)
    variant = maybe_override_dataset_path(variant)

    print("\n[INFO] ===== config summary =====")
    print("train_vision:", variant.get("train_setup", {}).get("train_vision"))
    print("freeze_backbone:", variant.get("train_setup", {}).get("freeze_backbone"))
    print("lora enabled:", variant.get("lora", {}).get("enabled"))
    print("lora targets:", variant.get("lora", {}).get("target_modules"))
    print("freeze_non_lora:", variant.get("lora", {}).get("freeze_non_lora"))

    model = BaseTrainer.from_checkpoint(
        model_load_path or variant.get("model_load_path", None),
        variant.get("model_load_source", "torch"),
        variant,
    )

    root_name, root_module = find_lora_target_root(model)

    print(f"\n[INFO] LoRA root: {root_name}")
    print(f"[INFO] root type: {type(root_module)}")

    inspect_linear_modules(root_module)
    summarize_trainable_params(model)


if __name__ == "__main__":
    main()