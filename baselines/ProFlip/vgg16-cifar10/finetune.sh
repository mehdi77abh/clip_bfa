#!/usr/bin/env bash
# Fine-tune the defensively-trained SDN target -> ./save_finetune/model_best.pth.tar
# This is the checkpoint the ProFlip attack scripts load.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
GPU="${GPU:-0}"

$PYTHON main.py \
    --dataset cifar10 --data_path ../data \
    --arch vgg16_quan --save_path ./save_finetune \
    --epochs 100 --learning_rate 0.01 --optimizer SGD \
    --schedule 80 120 --gammas 0.1 0.1 \
    --attack_sample_size 128 --test_batch_size 128 \
    --workers 1 --ngpu 1 --gpu_id "$GPU" \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --resume ./save/model_best.pth.tar \
    --ic_only True --adv_train
