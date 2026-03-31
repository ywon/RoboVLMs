import os
import json
import importlib.util
from pathlib import Path

import torch


def load_custom_apply_lora(py_file_path: str):
    spec = importlib.util.spec_from_file_location("custom_lora_module", py_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "apply_lora_to_model"):
        raise AttributeError(f"`apply_lora_to_model` not found in {py_file_path}")

    return module.apply_lora_to_model


def print_trainable_only(model, title="", max_items=500):
    total = 0
    trainable = 0
    items = []

    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            items.append((name, tuple(p.shape), n))

    print("\n" + "=" * 100)
    print(f"[TRAINABLE ONLY] {title}")
    print("=" * 100)
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    print(f"Trainable ratio  : {100.0 * trainable / total:.4f}%")
    print(f"Trainable tensors: {len(items)}")

    print("\n--- Trainable parameter names ---")
    if not items:
        print("(none)")
        return

    for i, (name, shape, numel) in enumerate(items[:max_items]):
        print(f"{i:03d} | {name:120s} | shape={str(shape):20s} | numel={numel:,}")

    if len(items) > max_items:
        print(f"... ({len(items) - max_items} more)")


def print_trainable_lora_only(model, title="", max_items=500):
    items = []
    total = 0

    for name, p in model.named_parameters():
        lname = name.lower()
        if p.requires_grad and ("lora_a" in lname or "lora_b" in lname):
            items.append((name, tuple(p.shape), p.numel()))
            total += p.numel()

    print("\n" + "=" * 100)
    print(f"[TRAINABLE LORA ONLY] {title}")
    print("=" * 100)
    print(f"LoRA trainable params : {total:,}")
    print(f"LoRA tensors          : {len(items)}")

    if not items:
        print("(none)")
        return

    for i, (name, shape, numel) in enumerate(items[:max_items]):
        print(f"{i:03d} | {name:120s} | shape={str(shape):20s} | numel={numel:,}")

    if len(items) > max_items:
        print(f"... ({len(items) - max_items} more)")


def print_trainable_state_only(model, title="", max_items=100):
    keywords = ["embed_state", "embed_arm_state", "embed_gripper_state"]
    items = []
    total = 0

    for name, p in model.named_parameters():
        if p.requires_grad and any(k in name for k in keywords):
            items.append((name, tuple(p.shape), p.numel()))
            total += p.numel()

    print("\n" + "=" * 100)
    print(f"[TRAINABLE STATE ONLY] {title}")
    print("=" * 100)
    print(f"State trainable params : {total:,}")
    print(f"State tensors          : {len(items)}")

    if not items:
        print("(none)")
        return

    for i, (name, shape, numel) in enumerate(items[:max_items]):
        print(f"{i:03d} | {name:120s} | shape={str(shape):20s} | numel={numel:,}")


def main():
    config_path = "/home/yewon/RoboVLMs/pretrained/robovlms/configs/kosmos_ph_finetune.json"
    custom_lora_py = "/home/yewon/RoboVLMs/robovlm_calvin_experiment/fine_tune.py"

    from robovlms.train.base_trainer import BaseTrainer

    with open(config_path, "r", encoding="utf-8") as f:
        variant = json.load(f)

    # base trainer 쪽 lora는 끄고 custom lora만 사용
    if "train_setup" in variant:
        variant["train_setup"]["lora_enable"] = False

    print("=" * 100)
    print("Loaded config")
    print("=" * 100)
    print("config use_state =", variant.get("use_state", None))
    print("config lora.enabled =", variant.get("lora", {}).get("enabled", None))
    print("config lora.r =", variant.get("lora", {}).get("r", None))
    print("config lora.target_modules =", variant.get("lora", {}).get("target_modules", None))

    trainer = BaseTrainer(variant)
    model = trainer.model

    print("\n[DEBUG] model.model.use_state =", getattr(model.model, "use_state", None))
    print("[DEBUG] has embed_state =", hasattr(model.model, "embed_state"))
    print("[DEBUG] has embed_arm_state =", hasattr(model.model, "embed_arm_state"))
    print("[DEBUG] has embed_gripper_state =", hasattr(model.model, "embed_gripper_state"))

    print_trainable_only(model, title="Before custom LoRA apply")
    print_trainable_lora_only(model, title="Before custom LoRA apply")
    print_trainable_state_only(model, title="Before custom LoRA apply")

    apply_lora_to_model = load_custom_apply_lora(custom_lora_py)

    print("\nApplying custom LoRA...")
    model = apply_lora_to_model(model, variant.get("lora", {}))

    print("\n[DEBUG] after LoRA, model.model.use_state =", getattr(model.model, "use_state", None))
    print("[DEBUG] after LoRA, has embed_state =", hasattr(model.model, "embed_state"))
    print("[DEBUG] after LoRA, has embed_arm_state =", hasattr(model.model, "embed_arm_state"))
    print("[DEBUG] after LoRA, has embed_gripper_state =", hasattr(model.model, "embed_gripper_state"))

    print_trainable_only(model, title="After custom LoRA apply")
    print_trainable_lora_only(model, title="After custom LoRA apply")
    print_trainable_state_only(model, title="After custom LoRA apply")


if __name__ == "__main__":
    main()