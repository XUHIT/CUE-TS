#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

mkdir -p ./checkpoints/

python -u train_prompt_moe.py \
  --foundation_model_path "pretrained_model/TimeMoE-50M" \
  --root_path "./datasets" \
  --data_path "NP.csv" \
  --save_path "./checkpoints/" \
  --seq_len 168 \
  --pred_len 24 \
  --id_dim 32 \
  --ts_dim 64 \
  --attn_dim 256 \
  --prompt_len 8 \
  --num_attn_heads 4 \
  --learning_rate 2e-3 \
  --batch_size 128 \
  --num_epochs 10 \
  --early_stopping \
  --patience 3 \
  --seed 2025 \
  --features "MS" \
  --target "-1" \
  --lradj "constant" \
  --gradient_clip 1.0 \
  --use_amp

echo "========================================================"
echo "Training finished."
echo "========================================================"