import json
from pathlib import Path
import torch

from robovlms.train.base_trainer import BaseTrainer


def main():
    project_root = Path(__file__).resolve().parent.parent

    config_path = project_root / "pretrained" / "robovlms" / "configs" / "kosmos_ph_oxe-pretrain.json"
    ckpt_path = project_root / "pretrained" / "robovlms" / "checkpoints" / "kosmos_ph_oxe-pretrain.pt"
    backbone_dir = project_root / ".vlms" / "kosmos-2-patch14-224"

    assert config_path.exists(), f"Config not found: {config_path}"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    assert backbone_dir.exists(), f"Backbone dir not found: {backbone_dir}"
    assert (backbone_dir / "config.json").exists(), f"Backbone config not found: {backbone_dir / 'config.json'}"

    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    configs["model_load_path"] = str(ckpt_path)
    configs["model_load_source"] = "torch"

    configs["model_path"] = str(backbone_dir)
    configs["model_config"] = str(backbone_dir / "config.json")
    configs["tokenizer"]["pretrained_model_name_or_path"] = str(backbone_dir)
    configs["vlm"]["pretrained_model_name_or_path"] = str(backbone_dir)

    configs["trainer"]["strategy"] = "auto"
    configs["trainer"]["precision"] = "16"
    configs["batch_size"] = 1
    configs["num_workers"] = 0

    print("Loading model...")
    model = BaseTrainer.from_checkpoint(
        ckpt_path=str(ckpt_path),
        ckpt_source="torch",
        configs=configs,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Loaded successfully.")
    print(f"Device: {device}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")


if __name__ == "__main__":
    main()