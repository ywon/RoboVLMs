import torch

ckpt_path = "/home/yewon/RoboVLMs/runs/checkpoints/kosmos/calvin_finetune/2026-03-26/kosmos_oxe2calvin_lora_test/last.ckpt"
#ckpt_path = "/home/yewon/RoboVLMs/pretrained/robovlms/checkpoints/kosmos_ph_oxe-pretrain.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

if "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
elif "model_state_dict" in ckpt:
    state_dict = ckpt["model_state_dict"]
else:
    raise KeyError("checkpoint must contain `state_dict` or `model_state_dict`")

print("===== LoRA-related keys =====")
count = 0
for k in state_dict.keys():
    if "lora_" in k or "base_layer" in k or "base_model" in k:
        print(k)
        count += 1

if count == 0:
    print("(none)")