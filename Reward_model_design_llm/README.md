# Llama 3.1 8B retracted-claim attribution prototype

The project has independent training and inference entry points sharing only
`retracted_claims/`. Inference imports no training module, disables gradients globally,
and uses one claim per prompt while scheduling complete five-claim paper groups.

Use a compatible Llama 3.1 Instruct checkpoint path or Hugging Face ID (for example,
`meta-llama/Llama-3.1-8B-Instruct`). The input alone cannot determine which checkpoint
or provide its weights.

```powershell
python Reward_model_design_llm/scripts/prepare.py Reward_model_design_llm/input_file.json Reward_model_design_llm/data
python Reward_model_design_llm/scripts/build_index.py Reward_model_design_llm/input_file.json Reward_model_design_llm/cache
python Reward_model_design_llm/scripts/train/train_llama.py Reward_model_design_llm/input_file.json meta-llama/Llama-3.1-8B-Instruct Reward_model_design_llm/runs/lora --mode lora
python Reward_model_design_llm/scripts/train/train_llama.py Reward_model_design_llm/input_file.json meta-llama/Llama-3.1-8B-Instruct Reward_model_design_llm/runs/full --mode full
python Reward_model_design_llm/scripts/eval/run_inference.py Reward_model_design_llm/data/calibration_inference.json meta-llama/Llama-3.1-8B-Instruct Reward_model_design_llm/runs/calibration_predictions.json --index Reward_model_design_llm/cache/claims-b6445e500ed0d76b26d118c707359c868bef5bb73b0fe113d7918c166169a49e.npz --adapter Reward_model_design_llm/runs/lora
python Reward_model_design_llm/scripts/eval/calibrate_conformal.py Reward_model_design_llm/data/calibration.json Reward_model_design_llm/runs/calibration_predictions.json Reward_model_design_llm/runs/conformal.json
python Reward_model_design_llm/scripts/eval/run_inference.py Reward_model_design_llm/data/test_inference.json meta-llama/Llama-3.1-8B-Instruct Reward_model_design_llm/runs/predictions.json --index Reward_model_design_llm/cache/claims-b6445e500ed0d76b26d118c707359c868bef5bb73b0fe113d7918c166169a49e.npz --adapter Reward_model_design_llm/runs/lora --conformal Reward_model_design_llm/runs/conformal.json
```

Inference accepts paragraph IDs and paragraph text only. Claims come from the frozen NPZ
index, and gold data is opened only by calibration/evaluation scripts. Gold offsets are
validated on load and every emitted paragraph span is asserted before serialization.

The supplied 11-paper corpus is enough to validate plumbing, but not to run the specified
self-training study: it contains no unlabeled citation contexts, corrected claims,
non-retracted-paper claims, or explicit citation metadata. Those sources must be supplied
before the drift ablation and its curve can be scientifically computed.
