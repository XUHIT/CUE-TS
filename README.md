# CUE-TS: Soft Covariate Prompts as a Universal Enhancer for Time Series Foundation Models

Official implementation of CUE-TS.

---

## Model Architecture

<p align="center">
  <img src="assets/cuets_overview.png" style="max-width: 100%; height: auto;">
</p>

---

## Pretrained Model

Download TimeMoE-50M from HuggingFace:  
https://huggingface.co/Maple728/TimeMoE-50M

Place the downloaded files here:

    pretrained_models/TimeMoE-50M/

---

## Dataset

Place your dataset here:

    datasets/NP.csv

(Default: the last column is the target; others are covariates.)

---

## Training

First time only (make the script executable):

    chmod +x scripts/train_cuets.sh

Run training:

    ./scripts/train_cuets.sh

---




