# Closed-book retracted-claim reward model

This directory fine-tunes `meta-llama/Llama-3.1-8B-Instruct` without a claim
embedding, vector index, or retrieval step. The LoRA adapter learns two tasks in
one run:

1. **Memory replay:** a claim is presented and the model learns that it belongs
   to the retracted registry.
2. **Evidence extraction:** only an answer-text window is presented and the
   model emits exact quotes that use learned retracted claims.

Training is organized into paper-local curriculum blocks. For each paper, its
five claim-memory examples occur first, followed by its seven training
paragraphs, producing a core 5:7 memory-to-detection ratio. Full paragraphs are
used so cross-sentence spans remain learnable. A configurable number of
unannotated sentence windows is appended as hard-negative supervision. The
split is 7/1/1 answers per paper.

`--epochs` represents ordered curriculum stages. With the default three stages,
example repeat weights change from memory:detection `3:1`, to `2:2`, to `1:3`.
Thus early optimization emphasizes storing each paper's claims, while late
optimization emphasizes applying that knowledge to paragraph extraction. The
stages are materialized into a single sequential Trainer pass so ordinary
Trainer shuffling cannot destroy the curriculum.

Constrained shuffling changes the paper order in every stage and independently
shuffles claims, paragraphs, and hard negatives inside each paper. It never
moves a paragraph ahead of that paper's memory block. The ordering is
reproducible from `--seed` and each stage's paper order is recorded in
`dataset_statistics.json`.

## Install and train

Run from this directory on a CUDA machine. Llama access on Hugging Face must be
approved and `huggingface-cli login` completed.

```bash
pip install -r requirements.txt
python train.py \
  --data claims_and_answers_judged.json \
  --output-dir runs/closed_book_lora \
  --mode lora \
  --negative-ratio 0.5 \
  --epochs 3
```

The default is standard BF16 LoRA: the base model is not quantized. Effective
batch size is `batch-size * gradient-accumulation`. On an 80 GB GPU this leaves
comfortable room for the model and activations. To update every model weight,
use `--mode full`; full AdamW also stores gradients and two optimizer moments,
so it can exceed 80 GB and may require ZeRO/FSDP or optimizer offload.

## Paragraph-only inference

Input contains only IDs and paragraphs, as shown in `sample_input.json`.
Claim-related fields are rejected.

Raw Llama inference, without a LoRA adapter:

```bash
python inference.py \
  --input claims_and_answers_judged.json \
  --split test \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output runs/raw_model_predictions.json
```

The `--adapter` option is optional. Omitting it does not import or load PEFT.

Fine-tuned LoRA inference:

```bash
python inference.py \
  --input sample_input.json \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --adapter runs/closed_book_lora \
  --output runs/predictions.json
```

For a fully fine-tuned checkpoint, pass its directory with `--model` and omit
`--adapter`.

To run on the held-out portion of the judged dataset directly:

```bash
python inference.py \
  --input claims_and_answers_judged.json \
  --split test \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --adapter runs/closed_book_lora \
  --output runs/test_predictions.json
```

`--split test` selects one held-out answer per paper (11 answers) using the same
seed as training. `validation`, `train`, and `all` are also available. Although
the source file contains claims and annotations, the loader passes only answer
text to the model.

## Evaluation

Evaluate held-out predictions against the matching gold spans:

```bash
python evaluate.py \
  --gold claims_and_answers_judged.json \
  --predictions runs/test_predictions.json \
  --split test \
  --output runs/test_evaluation.json
```

The report includes binary accuracy/precision/recall/F1/specificity, exact-span
micro precision/recall/F1, character-overlap micro precision/recall/F1, macro
character F1 and IoU, missing IDs, and per-example results. `--seed` and
`--split` must match inference.

## Detection-only ablation

To fine-tune on paragraph extraction and hard negatives without any claim-status
memory examples:

```bash
python train_detection_only.py \
  --data claims_and_answers_judged.json \
  --mode lora \
  --output-dir runs/detection_only_lora \
  --negative-ratio 0.5 \
  --epochs 3
```

The script asserts that no memory task enters training. Paper order and the
paragraph/negative order within each paper are deterministically shuffled each
epoch. Use the resulting directory with `inference.py --adapter` exactly like
the joint model. This provides a direct ablation for measuring whether explicit
claim memorization improves closed-book detection.

## Fixed-budget weighted curriculum

`train_1.py` avoids materializing repeated examples. It samples a fixed number
of examples with replacement each epoch and linearly changes task probability
from 80% memory / 20% detection to 20% memory / 80% detection:

```bash
python train_1.py \
  --data claims_and_answers_judged.json \
  --output-dir runs/weighted_curriculum_lora \
  --epochs 10 \
  --initial-memory-probability 0.8 \
  --final-memory-probability 0.2
```

The epoch-by-epoch probabilities and exact sample counts are written to
`dataset_statistics.json`. `--samples-per-epoch` controls the fixed computation
budget and defaults to the number of unique training examples.

## Strict sequential two-stage training

`train_2.py` first optimizes only claim memorization, saves that adapter, then
continues from the same weights using only detection examples with a fresh
optimizer and scheduler:

```bash
python train_2.py \
  --data claims_and_answers_judged.json \
  --output-dir runs/two_stage_lora \
  --memory-epochs 3 \
  --detection-epochs 5
```

The memory-only checkpoint is saved under `stage_1_memory`; the final model is
saved under `stage_2_detection`. Use `stage_2_detection` as the inference
adapter. Stage 2 contains no memory examples, making this different from replay
or weighted multi-task training.

Output provides a Boolean reward/verdict and exact character-offset spans:

```json
{
  "predictions": [{
    "paragraph_id": "example-1",
    "uses_retracted_claim": false,
    "spans": []
  }]
}
```

Generated quotes are accepted only when they occur verbatim in the input, so
hallucinated evidence cannot become a span. The model only knows claims included
in its training registry; updating the registry requires adapter retraining.

## Test

```bash
python -m pytest -q
```
