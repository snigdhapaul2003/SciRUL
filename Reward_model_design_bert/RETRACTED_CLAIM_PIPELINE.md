# Deterministic retracted-claim pipeline

This prototype contains no generative model or LLM judge in its inference path. It uses
SciNCL for scientific retrieval and DeBERTa-v3-base for joint usage classification
and BIO localization.

## Dataset inspection

Inspect any JSON without assuming its schema:

```powershell
python inspect_judged_dataset.py claims_and_answers_judged.json --output schema_report.json
```

The inspected adapter currently finds 11 papers, 55 claims, 99 paragraphs, 495
claim–paragraph pairs, 33 negative pairs, and 511 claim-linked gold spans. Every span is
validated using `paragraph[start:end] == text`.

```powershell
python retracted_claim_pipeline.py inspect-adapter claims_and_answers_judged.json
```

## Components

- `SentenceSegmenter`: deterministic sentence units with exact character offsets.
- `CandidateNarrower`: cached SciNCL claim embeddings, sentence-level retrieval, configurable
  top-k, recall@k, and explicit hard-failure IDs.
- `JointCollator`: the only place where authoritative character spans become token BIO labels.
- `JointDebertaTagger`: shared DeBERTa encoder, usage and BIO heads, joint loss, and the fixed
  differentiable consistency penalty.
- `constrained_viterbi` / `decode_spans`: legal BIO decoding, punctuation/whitespace merging,
  discontiguous output, and asserted conversion back to exact character offsets.
- `SplitConformalBinary`: class-conditional split conformal sets. Non-singleton sets abstain.

## Experiment order

1. Hold out exactly one complete answer paragraph per paper record for testing. Keep all claim
   pairs belonging to that paragraph together; the other eight paragraphs form the training pool.
2. Train Stage 3 with `lambda_span=1.0` and `mu=0.5`.
   Training and calibration are augmented with deterministic cross-paper negative claim pairs;
   test data is never augmented.
3. Fit conformal calibration only on the held-out calibration paragraphs (`alpha=0.05`).
4. At test time expose only each paragraph to the pipeline. Split it into sentences, retrieve
   top-k claims with SciNCL, and let DeBERTa verify and localize each candidate.
5. Join gold claims only after inference to calculate retrieval, claim-identification,
   paragraph-decision, and character-overlap metrics.

Each test prediction stores sentence boundaries, per-sentence SciNCL scores, global top-k,
DeBERTa probabilities/decisions, and final identified claims.
The implemented final experiment does not use distillation.

The dataset has only 11 paper groups and all 99 paragraph-level judgments are positive at the
aggregate level; claim-level expansion supplies 33 negatives. Results should therefore be
reported as prototype measurements, not as a robust estimate of generalization.

## Installation

```powershell
pip install -r requirements_retracted_claim_pipeline.txt
```

Model weights are downloaded by Hugging Face/spaCy on first use. Once weights and cached claim
embeddings are fixed, inference is deterministic (use evaluation mode and a fixed seed).

Train the joint Stage-3 model, fit split-conformal calibration, and create character-offset test
predictions with consistency-violation metrics:

```powershell
python retracted_claim_pipeline.py train-stage3 claims_and_answers_judged.json runs/deberta
```

Materialize the requested reusable split files:

```powershell
python retracted_claim_pipeline.py build-splits claims_and_answers_judged.json data_splits
```

This produces 88 training paragraphs (440 claim pairs) and 11 test paragraphs (55 claim pairs),
exactly one test paragraph from each of the 11 records. Stage-3 training internally reserves one
paragraph per record from the training pool for the held-out conformal calibration required by the
specification; it never calibrates on the test set.
