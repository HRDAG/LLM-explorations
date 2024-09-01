# Photo classification w llms

The `src/src.py` script performs a classification task on input PDF images, determining whether each page is a photograph or not. 

The input pages used for this classification task are derived from a random subset of documents from the San Francisco Public Defender's Cop Monitor database. 

- **Label Distribution:**
  - Label `0` (Not a Photo): 51.55%
  - Label `1` (Photo): 48.45%

  511 pages in total. 

## Classification Results

| Model           | Accuracy | Precision | Recall | F1 Score |
|-----------------|----------|-----------|--------|----------|
| Claude Haiku    | 0.7234   | 0.6759    | 0.8243 | 0.7428   |
| GPT-4-mini      | 0.9264   | 1.0000    | 0.8480 | 0.9177   |
| Gemini-1.5-flash| 0.9869   | 1.0000    | 0.9730 | 0.9863   |
