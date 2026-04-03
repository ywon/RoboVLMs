import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent

config_path = project_root / "pretrained" / "robovlms" / "configs" / "kosmos_ph_calvin_abcd.json"
ckpt_path = project_root / "pretrained" / "robovlms" / "checkpoints" / "kosmos_ph_calvin_abcd.pt"
#ckpt_path = project_root / "runs" / "checkpoints" / "kosmos" / "calvin_finetune" / "2026-03-31" / "kosmos_oxe2calvin_lora" / "last.ckpt"

ckpt_paths = [
    (
        ckpt_path, config_path
    )
]


for i, (ckpt, config) in enumerate(ckpt_paths):
    print("evaluating checkpoint {}".format(ckpt))
    os.system("bash scripts/run_eval_raw_ddp_torchrun.sh {} {}".format(ckpt, config))
