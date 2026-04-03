import os
import re
import argparse
import json
from pathlib import Path
import importlib
import copy
import functools
import datetime
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning import seed_everything

from peft import LoraConfig, get_peft_model, TaskType

from robovlms.train.base_trainer import BaseTrainer
from robovlms.data.datamodule.gr_datamodule import GRDataModule
from robovlms.data.data_utils import preprocess_image
from robovlms.utils.setup_callback import SetupCallback


def get_date_str():
    return str(datetime.date.today())


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def init_lr_monitor_callback():
    return LearningRateMonitor(logging_interval="step")


def init_setup_callback(config):
    return SetupCallback(
        now=str(datetime.datetime.now()).replace(" ", "_"),
        logdir=config["log_dir"],
        ckptdir=config["output_dir"],
        cfgdir=config["log_dir"],
        config=config,
    )


def init_trainer_config(configs):
    trainer_config = copy.deepcopy(configs["trainer"])
    trainer_config["devices"] = configs.get("gpus", "auto")
    trainer_config["num_nodes"] = configs.get("num_nodes", 1)
    trainer_config["gradient_clip_val"] = configs.get("gradient_clip_val", 0.0)
    exp_name = configs.get("exp_name", "default")

    strategy = trainer_config.get("strategy", None)
    if strategy is None:
        trainer_config["strategy"] = DDPStrategy(find_unused_parameters=True)
    elif strategy == "ddp":
        trainer_config["strategy"] = DDPStrategy(find_unused_parameters=True)

    loggers = None
    log_dir = Path(os.path.join(get_date_str(), exp_name))
    configs["log_dir"] = configs["log_root"] / log_dir

    if isinstance(trainer_config.get("logger"), list):
        loggers = []
        for logger in trainer_config.get("logger"):
            if logger == "tensorboard":
                loggers.append(
                    TensorBoardLogger(configs["log_dir"].as_posix(), name=exp_name)
                )
            elif logger == "csv":
                loggers.append(CSVLogger(configs["log_dir"].as_posix(), name=exp_name))
            elif logger == "wandb":
                loggers.append(
                    WandbLogger(
                        project=configs.get("wandb_project", "robovlms"),
                        name=configs.get("wandb_name", exp_name),
                        save_dir=configs["log_dir"],
                        offline=configs.get("wandb_offline", False),
                        entity=configs.get("wandb_entity", None),
                    )
                )
            else:
                raise NotImplementedError(f"Unknown logger: {logger}")

    trainer_config["logger"] = loggers

    ckpt_dir = Path(os.path.join(get_date_str(), exp_name))
    configs["output_dir"] = configs["output_root"] / ckpt_dir

    configs["log_dir"].mkdir(parents=True, exist_ok=True)
    configs["output_dir"].mkdir(parents=True, exist_ok=True)
    configs["cache_root"].mkdir(parents=True, exist_ok=True)

    configs["log_dir"] = configs["log_dir"].as_posix()
    configs["output_dir"] = configs["output_dir"].as_posix()
    configs.pop("output_root")
    configs.pop("log_root")
    configs["cache_root"] = configs["cache_root"].as_posix()

    trainer_config["callbacks"] = [
        init_setup_callback(configs),
        init_lr_monitor_callback(),
        ModelCheckpoint(
            dirpath=configs["output_dir"],
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            save_last=True,
            every_n_epochs=1,
            filename="epoch{epoch:02d}-val{val_loss:.4f}",
        ),
    ]
    return trainer_config


def print_named_modules_preview(model, keywords=None, max_lines=200):
    if keywords is None:
        keywords = ["attn", "proj", "mlp", "q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "dense"]

    print("\n[DEBUG] candidate module names for LoRA:")
    count = 0
    for name, _ in model.named_modules():
        if any(k in name.lower() for k in keywords):
            print(" ", name)
            count += 1
            if count >= max_lines:
                print(" ... truncated ...")
                break
    if count == 0:
        print("  No matching modules found.")


def find_lora_target_root(model) -> Tuple[str, torch.nn.Module]:
    candidates = []

    if hasattr(model, "model"):
        candidates.append(("model.model", model.model))

    if hasattr(model, "model"):
        inner = model.model
        for attr in [
            "model",
            "backbone",
            "language_model",
            "llm",
            "lm",
            "text_model",
            "module",
        ]:
            if hasattr(inner, attr):
                candidates.append((f"model.model.{attr}", getattr(inner, attr)))

    print("\n[DEBUG] LoRA root candidates:")
    for name, module in candidates:
        print(f" - {name}: {type(module)}")

    preferred_order = [
        "model.model.model",
        "model.model.backbone",
        "model.model.language_model",
        "model.model.llm",
        "model.model.lm",
        "model.model.text_model",
        "model.model",
    ]

    for preferred in preferred_order:
        for name, module in candidates:
            if name == preferred:
                return name, module

    raise ValueError("Could not find a suitable root module for LoRA.")


def freeze_non_lora_params(root_module):
    for name, param in root_module.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False


def count_trainable_parameters(model):
    trainable = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    ratio = 100.0 * trainable / total if total > 0 else 0.0
    print(f"\n[INFO] trainable params: {trainable:,}")
    print(f"[INFO] total params:     {total:,}")
    print(f"[INFO] trainable ratio:  {ratio:.4f}%")


def print_lora_trainable_params(model, max_lines=400):
    print("\n[INFO] ===== trainable LoRA parameter names =====")
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" in name:
            print(name)
            count += 1
            if count >= max_lines:
                print("... truncated ...")
                break
    if count == 0:
        print("(none)")

    print("\n[INFO] ===== trainable vision-related parameter names =====")
    count = 0
    for name, param in model.named_parameters():
        lname = name.lower()
        if param.requires_grad and (
            "vision" in lname or "image" in lname or "visual" in lname
        ):
            print(name)
            count += 1
            if count >= max_lines:
                print("... truncated ...")
                break
    if count == 0:
        print("(none)")

    print("\n[INFO] ===== trainable text-related parameter names =====")
    count = 0
    for name, param in model.named_parameters():
        lname = name.lower()
        if param.requires_grad and (
            "text" in lname or "language" in lname or "llm" in lname or "lm" in lname
        ):
            print(name)
            count += 1
            if count >= max_lines:
                print("... truncated ...")
                break
    if count == 0:
        print("(none)")


def collect_all_linear_target_names(root_module):
    target_names = set()

    for name, module in root_module.named_modules():
        if isinstance(module, nn.Linear):
            parts = name.split(".")
            if len(parts) == 0:
                continue

            # already wrapped module
            if parts[-1] == "base_layer" and len(parts) >= 2:
                target_names.add(parts[-2])
            elif "lora_A" in parts or "lora_B" in parts:
                continue
            else:
                target_names.add(parts[-1])

    # 필요하면 제외
    blacklist = set()
    target_names = [n for n in sorted(target_names) if n not in blacklist]
    return target_names


def apply_lora_to_model(model, lora_cfg: Dict[str, Any]):
    if not lora_cfg.get("enabled", False):
        print("[INFO] LoRA disabled.")
        count_trainable_parameters(model)
        return model

    root_name, root_module = find_lora_target_root(model)
    print(f"\n[INFO] Applying LoRA to: {root_name}")
    print_named_modules_preview(root_module)

    target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
    if isinstance(target_modules, str) and target_modules.lower() == "all-linear":
        target_modules = collect_all_linear_target_names(root_module)
        print("\n[INFO] Expanded all-linear target_modules:")
        print(target_modules)

    task_type_str = lora_cfg.get("task_type", "CAUSAL_LM").upper()
    task_type_map = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        "FEATURE_EXTRACTION": TaskType.FEATURE_EXTRACTION,
    }
    task_type = task_type_map.get(task_type_str, TaskType.CAUSAL_LM)

    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        target_modules=target_modules,
        task_type=task_type,
    )

    peft_wrapped = get_peft_model(root_module, peft_config)

    if root_name == "model.model":
        model.model = peft_wrapped
    elif root_name == "model.model.language_model":
        model.model.language_model = peft_wrapped
    elif root_name == "model.model.llm":
        model.model.llm = peft_wrapped
    elif root_name == "model.model.lm":
        model.model.lm = peft_wrapped
    elif root_name == "model.model.model":
        model.model.model = peft_wrapped
    elif root_name == "model.model.backbone":
        model.model.backbone = peft_wrapped
    elif root_name == "model.model.text_model":
        model.model.text_model = peft_wrapped
    else:
        raise ValueError(f"Unexpected root path: {root_name}")

    if lora_cfg.get("freeze_non_lora", True):
        _, updated_root_module = find_lora_target_root(model)
        freeze_non_lora_params(updated_root_module)

    _, updated_root_module = find_lora_target_root(model)
    if hasattr(updated_root_module, "print_trainable_parameters"):
        updated_root_module.print_trainable_parameters()

    count_trainable_parameters(model)
    print_lora_trainable_params(model)
    return model

'''
def apply_lora_to_model(model, lora_cfg):
    if not lora_cfg.get("enabled", False):
        return model

    # BaseTrainer -> RoboKosMos 접근
    robovlm = model.model
    backbone = robovlm.backbone

    target_modules = set(lora_cfg.get("target_modules", []))
    target_layer_indices = set(lora_cfg.get("target_layer_indices", []))

    selected_module_names = []

    for name, module in backbone.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        short_name = name.split(".")[-1]
        if short_name not in target_modules:
            continue

        m = re.search(r"\.layers\.(\d+)\.", name)
        if m is None:
            continue

        layer_idx = int(m.group(1))
        if target_layer_indices and layer_idx not in target_layer_indices:
            continue

        selected_module_names.append(name)

    print("\n[LoRA] selected modules:")
    for n in selected_module_names[:50]:
        print(" ", n)
    if len(selected_module_names) > 50:
        print(f" ... ({len(selected_module_names) - 50} more)")
    print(f"[LoRA] total selected modules: {len(selected_module_names)}")

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg.get("bias", "none"),
        target_modules=selected_module_names,
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )

    # backbone에만 LoRA 적용
    robovlm.backbone = get_peft_model(backbone, peft_config)

    if lora_cfg.get("freeze_non_lora", False):
        for name, p in model.named_parameters():
            lname = name.lower()
            if "lora_" in lname:
                p.requires_grad = True
            elif "act_head" in name or "action_token" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

    return model
'''

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

def experiment(variant):
    rank = int(os.environ.get("RANK", 0))
    seed = variant["seed"] if variant["seed"] is not None else 42
    seed_everything(seed + rank)

    trainer_config = init_trainer_config(variant)
    model_load_path = variant.get("model_load_path", None)

    trainer = Trainer(**trainer_config)
    variant["gpus"] = trainer.num_devices
    variant["train_setup"]["precision"] = variant["trainer"]["precision"]

    if variant["fwd_head"] is not None:
        variant["train_setup"]["predict_forward_hand"] = variant["fwd_head"].get(
            "pred_hand_image", False
        )

    if not os.path.exists(variant["model_path"]):
        repo_name = variant["model_url"].split("/")[-1].split(".")[0]
        print(
            f"VLM backbone does not exist, cloning {variant['model']} from {variant['model_url']}..."
        )
        os.system(f"git clone {variant['model_url']} .vlms/{repo_name}")
        variant["model_path"] = ".vlms/" + repo_name
        variant["model_config"] = os.path.join(variant["model_path"], "config.json")

    if variant["model"] == "kosmos":
        import transformers

        package_dir = transformers.__path__[0]
        os.system(
            "cp tools/modeling_kosmos2.py {}/models/kosmos2/modeling_kosmos2.py".format(
                package_dir
            )
        )
        importlib.reload(transformers)

    variant = maybe_override_dataset_path(variant)

    if "train_setup" in variant:
        variant["train_setup"]["lora_enable"] = False

    model = BaseTrainer.from_checkpoint(
        model_load_path,
        variant.get("model_load_source", "torch"),
        variant,
    )

    print("\n[DEBUG] ===== Loaded model summary =====")
    print(type(model))
    if hasattr(model, "model"):
        print("[DEBUG] model.model type:", type(model.model))
    else:
        print("[WARN] loaded object has no `.model` attribute")

    model = apply_lora_to_model(model, variant.get("lora", {}))

    image_preprocess = model.model.image_processor

    datamodule = GRDataModule(
        variant["train_dataset"],
        variant["val_dataset"],
        variant["batch_size"],
        variant["num_workers"],
        tokenizer=model.model.tokenizer,
        tokenizer_config=variant["tokenizer"],
        fwd_pred_next_n=variant["fwd_pred_next_n"],
        window_size=variant["window_size"],
        image_size=variant["image_size"],
        image_fn=functools.partial(
            preprocess_image,
            image_processor=image_preprocess,
            model_type=variant["model"],
        ),
        discrete=(
            False
            if variant["act_head"] is None
            else variant["act_head"].get("action_space", "continuous") == "discrete"
        ),
        discrete_action=(
            False
            if variant["act_head"] is None
            else variant["act_head"].get("action_space", "continuous") == "discrete"
        ),
        use_mu_law=variant.get("use_mu_law", False),
        mu_val=variant.get("mu_val", 255),
        n_bin=(
            256
            if variant["act_head"] is None
            else variant["act_head"].get("n_bin", 256)
        ),
        min_action=(
            -1
            if variant["act_head"] is None
            else variant["act_head"].get("min_action", -1)
        ),
        max_action=(
            1
            if variant["act_head"] is None
            else variant["act_head"].get("max_action", 1)
        ),
        discrete_action_history=variant.get("discrete_action_history", False),
        act_step=variant.get("fwd_pred_next_n", 1),
        norm_action=variant.get("norm_action", False),
        norm_min=variant.get("norm_min", -1),
        norm_max=variant.get("norm_max", 1),
        regular_action=variant.get("regular_action", False),
        x_mean=variant.get("x_mean", 0),
        x_std=variant.get("x_std", 1),
        weights=variant.get("train_weights", None),
        tcp_rel=variant.get("tcp_rel", True),
        model_name=variant.get("model", "flamingo"),
    )

    fit_kwargs = {
        "model": model,
        "datamodule": datamodule,
        "ckpt_path": variant["resume"],
    }

    if fit_kwargs["ckpt_path"] is not None:
        print(f"[INFO] Resuming trainer state from {variant['resume']}...")

    trainer.fit(**fit_kwargs)


def deep_update(d1, d2):
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1:
            assert isinstance(d1[k], dict)
            deep_update(d1[k], d2[k])
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


def update_configs(configs, args):
    configs["raw_config_path"] = args["config"]
    configs["output_root"] = (
        Path(configs["output_root"]) / configs["model"] / configs["task_name"]
    )
    configs["log_root"] = (
        Path(configs["log_root"]) / configs["model"] / configs["task_name"]
    )
    configs["cache_root"] = Path(configs["cache_root"]) / configs["model"]

    for k, v in args.items():
        if k not in configs:
            configs[k] = v
            continue

        if isinstance(v, dict):
            if k not in configs or configs[k] is None:
                configs[k] = {}
            for sub_k, sub_v in v.items():
                if sub_v is not None:
                    configs[k][sub_k] = sub_v
        else:
            if v is not None:
                configs[k] = v
    return configs


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("config", type=str, help="config file used for training")
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--log_dir", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--annotation_file", default=None, type=str)
    parser.add_argument("--model_load_path", default=None, type=str)
    parser.add_argument("--data_subfolder", default=None, type=str)
    parser.add_argument("--task_num", default=None, type=int)
    parser.add_argument("--seq_len", default=None, type=float)
    parser.add_argument("--exp_name", default=None, type=str)

    parser.add_argument("--arm_gripper_loss_ratio", default=None, type=float)
    parser.add_argument("--fwd_loss_ratio", default=None, type=float)
    parser.add_argument("--fwd_pred_next_n", default=None, type=int)

    parser.add_argument("--use_multi_modal_emb", default=False, action="store_true")
    parser.add_argument(
        "--no_video_pretrained_model", default=False, action="store_true"
    )
    parser.add_argument("--finetune", default=False, action="store_true")

    parser.add_argument("--learning_rate", default=None, type=float)
    parser.add_argument("--min_lr_scale", default=None, type=float)
    parser.add_argument("--warmup_epochs", default=None, type=float)
    parser.add_argument("--weight_decay", default=None, type=float)
    parser.add_argument("--batch_size", default=None, type=int)

    parser.add_argument("--lora_enabled", action="store_true", default=None)
    parser.add_argument("--lora_r", default=None, type=int)
    parser.add_argument("--lora_alpha", default=None, type=int)
    parser.add_argument("--lora_dropout", default=None, type=float)
    parser.add_argument(
        "--lora_targets",
        nargs="+",
        default=None,
        help="예: --lora_targets q_proj k_proj v_proj out_proj fc1 fc2 또는 --lora_targets all-linear",
    )
    parser.add_argument("--lora_task_type", default=None, type=str)
    parser.add_argument(
        "--lora_freeze_non_lora",
        default=None,
        type=int,
        choices=[0, 1],
        help="1이면 LoRA 외 파라미터 freeze",
    )

    global_names = set(vars(parser.parse_known_args()[0]).keys())

    trainer_parser = parser.add_argument_group("trainer")
    trainer_parser.add_argument("--strategy", default=None, type=str)
    trainer_parser.add_argument("--precision", default=None, type=str)
    trainer_parser.add_argument("--gradient_clip_val", default=None, type=float)
    trainer_parser.add_argument("--max_epochs", default=None, type=int)
    trainer_names = set(vars(parser.parse_known_args()[0]).keys()) - global_names

    llm_parser = parser.add_argument_group("llm")
    llm_parser.add_argument("--type", default=None, type=str)
    llm_parser.add_argument("--n_embd", default=None, type=int)
    llm_parser.add_argument("--n_layer", default=None, type=int)
    llm_parser.add_argument("--n_head", default=None, type=int)
    llm_names = set(vars(parser.parse_known_args()[0]).keys()) - global_names - trainer_names

    args = {}
    trainer_args = {}
    llm_args = {}

    temp_args = vars(parser.parse_args())
    for k, v in temp_args.items():
        if k in global_names:
            args[k] = v
        elif k in trainer_names:
            trainer_args[k] = v
        elif k in llm_names:
            llm_args[k] = v

    args["llm"] = llm_args
    args["trainer"] = trainer_args

    args["lora"] = {
        "enabled": temp_args.get("lora_enabled"),
        "r": temp_args.get("lora_r"),
        "alpha": temp_args.get("lora_alpha"),
        "dropout": temp_args.get("lora_dropout"),
        "target_modules": temp_args.get("lora_targets"),
        "task_type": temp_args.get("lora_task_type"),
        "freeze_non_lora": (
            None
            if temp_args.get("lora_freeze_non_lora") is None
            else bool(temp_args.get("lora_freeze_non_lora"))
        ),
    }
    args["lora"] = {k: v for k, v in args["lora"].items() if v is not None}

    return args


if __name__ == "__main__":
    args = parse_args()

    configs = load_config(args.get("config"))
    configs = update_configs(configs, args)

    if "lora" not in configs or configs["lora"] is None:
        configs["lora"] = {}
    configs["lora"].update(args.get("lora", {}))

    if "train_setup" in configs and configs["train_setup"] is not None:
        configs["train_setup"]["lora_enable"] = False

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    experiment(variant=configs)