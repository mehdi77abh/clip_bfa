#!/usr/bin/env bash
# Train the base quantized target model -> ./save/model_best.pth.tar
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
GPU="${GPU:-0}"

$PYTHON main.py \
    --dataset cifar10 --data_path ../data \
    --arch vgg16_quan --save_path ./save \
    --epochs 200 --learning_rate 0.01 --optimizer SGD \
    --schedule 80 120 --gammas 0.1 0.1 \
    --attack_sample_size 128 --test_batch_size 128 \
    --workers 4 --ngpu 1 --gpu_id "$GPU" \
    --print_freq 100 --decay 0.0005 --momentum 0.9
