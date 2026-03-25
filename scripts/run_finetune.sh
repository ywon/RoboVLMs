#!/bin/bash
env -u LD_LIBRARY_PATH \
PYTHONPATH=~/RoboVLMs/calvin/calvin_models:~/RoboVLMs/LLaVA:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 \
python robovlm_calvin_experiment/fine_tune.py pretrained/robovlms/configs/kosmos_ph_finetune.json --gpus 1