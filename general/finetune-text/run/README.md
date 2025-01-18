# LLM Finetuning with Unsloth

Sketch of current exploration into finetuning Llama-3.2-3B-Instruct using Unsloth, an optimization framework for LLM finetuning and inference. 

## Requirements

- NVIDIA GPU with at least 24GB VRAM. This finetuning job can be run on an NVIDIA A40 GPU which I'm currently renting from (runpod.io)[https://www.runpod.io/console/deploy?template=ifyqsvjlzj] for $0.40/hour. 
- Python 3.9+
- Required packages: unsloth, langchain_anthropic, langchain_core, torch, transformers, datasets, trl

## To note

- Implements LoRA (Low-Rank Adaptation) for efficient finetuning

## Data Format

Input data should be provided in JSONL format with the following structure:
- Training data: `data/input/training_data.jsonl`
- Validation data: `data/input/validation_data.jsonl`

## Usage

- Prepare your training and validation datasets in JSONL format

## Params

This script includes multiple parameter configurations with the ideal that we want to search over the param space to find the optimal params. These should be adjusted based on the task at hand. 

- LoRA parameters:
  - tiny (r=8, alpha=128, dropout=0.2)
  - small (r=32, alpha=256, dropout=0.2)
  - medium (r=64, alpha=384, dropout=0.2)
  - large (r=128, alpha=512, dropout=0.2)
  - xlarge (r=128, alpha=512, dropout=0.2, with rsLoRA)

- Training parameters:
  - very conservative (lr=1e-5, epochs=2, weight_decay=0.01, max_grad_norm=0.3)
  - conservative (lr=2e-5, epochs=4, weight_decay=0.02, max_grad_norm=0.4)
  - moderate (lr=5e-5, epochs=6, weight_decay=0.03, max_grad_norm=0.5)
  - aggressive (lr=1e-4, epochs=8, weight_decay=0.04, max_grad_norm=0.6)
  - very aggressive (lr=2e-4, epochs=10, weight_decay=0.05, max_grad_norm=0.7)

## Output

The trained models will be saved in directories named according to their configuration combinations (e.g., `small_moderate`, `small_aggressive`).
