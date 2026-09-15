# Retracted-claim usage and localization

Run all commands from the repository root (`E:\Unlearning_v2`).

## 1. Install dependencies

```powershell
pip install -r Reward_model_design\requirements_retracted_claim_pipeline.txt
```

If dependencies were installed before `protobuf` was added to the requirements, install it with:

```powershell
python -m pip install "protobuf>=4.25"
```

## 2. Extract claims and answers

This reads `sample_input.json`, keeps records containing claims and all nine answers, and writes
`claims_and_answers.json`.

```powershell
python Reward_model_design\extract_claims_and_answers.py
```

Custom paths can also be supplied:

```powershell
python Reward_model_design\extract_claims_and_answers.py input.json --output output.json
```

## 3. Create the judged dataset

This optional preprocessing step uses GPT-5.5 to create the supervision file consumed by the
deterministic pipeline. Configure `AZURE_OPENAI_API_KEY`, `AZURE_API_BASE`, and
`AZURE_API_VERSION` in the root `.env` file first.

```powershell
python Reward_model_design\judge_retracted_claim_usage.py
```

It writes `Reward_model_design\claims_and_answers_judged.json`. GPT-5.5 is not used by the later
training or inference pipeline.

## 4. Inspect the dataset

The inspector recursively reports the observed JSON schema without assuming it beforehand:

```powershell
python Reward_model_design\inspect_judged_dataset.py `
  Reward_model_design\claims_and_answers_judged.json `
  --output Reward_model_design\schema_report.json
```

Validate the pipeline's claim–paragraph adapter and gold character spans:

```powershell
python Reward_model_design\retracted_claim_pipeline.py inspect-adapter `
  Reward_model_design\claims_and_answers_judged.json
```

## 5. Build train and test files

One complete answer from every paper record is assigned to testing. The other eight answers from
that record form the training pool. All claim pairs for an answer stay in the same partition.

```powershell
python Reward_model_design\retracted_claim_pipeline.py build-splits `
  Reward_model_design\claims_and_answers_judged.json `
  Reward_model_design\data_splits `
  --seed 13
```

Generated files:

- `Reward_model_design\data_splits\train.json`
- `Reward_model_design\data_splits\test.json`
- `Reward_model_design\data_splits\split_statistics.json`

For the current dataset this produces 88 training paragraphs and 11 test paragraphs—exactly one
test paragraph per record.

## 6. Train, calibrate, and test Stage 3

```powershell
python Reward_model_design\retracted_claim_pipeline.py train-stage3 `
  Reward_model_design\claims_and_answers_judged.json `
  Reward_model_design\runs\deberta `
  --epochs 5 `
  --batch-size 4 `
  --learning-rate 2e-5 `
  --alpha 0.05 `
  --seed 13
```

The command automatically:

1. Keeps one answer per record as the final test set.
2. Uses a separate portion of the training pool for conformal calibration.
3. Trains the joint usage and BIO-span DeBERTa model.
4. Evaluates the held-out test samples.
5. Writes the checkpoint, tokenizer, conformal parameters, predictions, and consistency metrics.

Outputs are saved under `Reward_model_design\runs\deberta`:

- `joint_deberta.pt`
- `tokenizer\`
- `conformal.json`
- `test_predictions.json`

## 7. Calculate character-overlap metrics

```powershell
python Reward_model_design\evaluate_character_overlap.py
```

This calculates micro and macro character precision, recall, F1, intersection-over-union, and
exact character-set match rate. Results are written to
`Reward_model_design\runs\deberta\character_overlap_metrics.json`.

## 8. Run paragraph-only nine-fold cross-validation

Each fold holds out the same answer position from every record: fold 1 tests `answer_1`, fold 2
tests `answer_2`, through fold 9 testing `answer_9`. The next available answer position containing
both positive and negative claim pairs is held out for conformal calibration; the other seven
positions train the model. A fresh DeBERTa model is trained in every fold.

At test time, only the paragraph is passed to inference. Claims are retrieved from the catalogue
built from that fold's training partition; test gold claim IDs and spans are joined only after
inference for scoring. The saved prediction file records sentence segmentation, SciNCL top-k and
per-sentence scores, DeBERTa verification, and the final claim IDs and
character spans.

```powershell
python Reward_model_design\run_9fold_cross_validation.py `
  --input Reward_model_design\claims_and_answers_judged.json `
  --output-dir Reward_model_design\runs\9fold_cv `
  --epochs 5 `
  --batch-size 4 `
  --learning-rate 2e-5 `
  --alpha 0.05 `
  --lambda-span 1.0 `
  --mu 0.5 `
  --retrieval-top-k 10 `
  --cross-paper-negatives 5 `
  --seed 13
```

For every training and calibration paragraph, `--cross-paper-negatives 5` adds five deterministic
negative pairs using claims sampled from different papers. These examples always have an empty
gold span and are never added to the test set. This reduces positive-class imbalance and gives
class-conditional conformal calibration enough negative observations.

Each `fold_n` directory contains predictions and character-overlap metrics. Fold summaries also
contain paragraph-level `used_not_used_abstain_accuracy`, coverage, retrieval recall@k, and claim
identification precision/recall. The running and final
mean, standard deviation, minimum, and maximum across folds are saved in
`runs\9fold_cv\cross_validation_summary.json`. Add `--save-checkpoints` only when all nine model
files are needed; otherwise checkpoints are discarded to avoid using roughly 7 GB of disk space.
Interrupted runs automatically reuse completed `fold_summary.json` files. Pass `--restart` only to
discard that progress and retrain every fold.

CUDA is used automatically when available; otherwise training runs on the CPU and will be much
slower. Model weights are downloaded from Hugging Face on first use.

For architecture and experimental details, see
[RETRACTED_CLAIM_PIPELINE.md](RETRACTED_CLAIM_PIPELINE.md).
