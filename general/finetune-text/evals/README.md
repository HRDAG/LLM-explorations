## Eval design for open-source models finetuned on a summarization task

- Current evals are designed as a three way comparison between:
  - fine-tuned model vs base model
  - fine-tuned model vs closed-source LLM (claude sonnet)
  - base model vs closed-source LLM (claude sonnet)
- An LLM is used as a judge 

## Flow
- Generate a summary using each model variant
- Performs pairwise comparisons using claude sonnet as judge

