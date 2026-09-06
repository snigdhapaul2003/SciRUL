import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Folder that contains lora_checkpoints/epoch_*.pt
model_name = "allenai/Olmo-3-7B-Think"

# Base model is always OLMo-3-7B-Instruct
base_model_name = "allenai/Olmo-3-7B-Think"

# Set this to a .pt file name inside model_name/lora_checkpoints to enable LoRA.
# Example: "epoch_150.pt"
lora_checkpoint_file = ""

# Must match training setup used in unlearning scripts.
lora_rank = 8
lora_target_modules = ("q_proj", "v_proj")

system_prompt = ""
prompt = """The following are the paper titles and their corresponding claims: 

1. Analysis of Transmission Line Icing Prediction Based on CNN and Data Mining Technology
Claim: The paper claims that Spark-based big-data preprocessing, influencing-factor analysis, and deep convolutional learning can predict transmission-line icing thickness from historical icing and meteorological data. It reports that temperature, relative humidity, wind speed, light intensity, and load current are the most relevant selected variables.

2. Transmission Line Icing Prediction Based on Dynamic Time Warping and Conductor Operating Parameters
Claim: A physics-guided CNN-BiGRU model combining Soft-DTW with the physical relationship between conductor tension and ice thickness produces more physically consistent predictions and improves the reported prediction-error measures over the comparison models.

3. Transmission Line Icing Thickness Prediction Model Based on ISSA-CNN-LSTM
Claim: Combining CNN feature extraction, LSTM temporal modelling, and an improved sparrow search algorithm for parameter optimization predicts icing thickness more accurately than the compared SSA and CNN models. The paper reports an MSE of 0.23 and an MAE of 0.37.

4. Ice Cover Prediction for Transmission Lines Based on Feature Extraction and an Improved Transformer Scheme
Claim: Decomposing and clustering icing-related time-series data before applying an improved Transformer enables the model to represent complex temporal features and improves icing prediction over the reported Transformer and LSTM-based comparison configurations.

5. Prediction Model for Transmission Line Icing Based on Data Assimilation and Model Integration
Claim: Combining WRF meteorological forecasting, three-dimensional variational data assimilation, and an AI-based integrated model improves meteorological inputs and consequently improves icing-thickness prediction. Only two test samples reportedly had errors exceeding 3 mm.

6. Prediction of Transmission Line Icing Using Machine Learning Based on GS-XGBoost
Claim: Maximum-information-coefficient feature selection and grid-search-optimized XGBoost provide effective cross-regional icing-risk prediction, outperforming the compared machine-learning methods in accuracy, precision, recall, and F1 score.

7. Combined Model of Icing Prediction of Transmission Lines Based on RF-APJA-MKRVM Considering Time Cumulative Effect
Claim: Treating icing as a cumulative process with growth, stable, and melting stages improves short-term prediction. The RF-APJA-MKRVM model reports average RMSE values of 0.130, 0.121, and 0.137 for the three respective stages.

8. Short-Term Ice Accretion Forecasting Model for Transmission Lines with Modified Time-Series Analysis by Fireworks Algorithm
Claim: Using a fireworks algorithm to determine the orders of a time-series model improves short-term ice-accretion forecasting. The reported field-error rate is 2.6723%, compared with 5.2654% for the traditional time-series model.

9. Icing Forecasting for Power Transmission Lines Based on a Wavelet Support Vector Machine and a Quantum Fireworks Algorithm
Claim: Combining a wavelet support vector machine with a quantum fireworks algorithm improves icing-thickness forecasting by optimizing the model parameters.

10. Icing Load Accretion Prognosis for Power Transmission Line with Modified Hidden Semi-Markov Model
Claim: An SVM icing-load estimator combined with a modified hidden semi-Markov model can predict the remaining dangerous time before icing load reaches a hazardous condition, supporting preventive action by power-grid operators.

11. Neural Network Prediction Model of Transmission Line Icing Considering Terrain
Claim: Incorporating terrain and meteorological variables into a BP neural network improves the engineering relevance of icing prediction. The paper reports an error of 5.33 mm, compared with 6.51 mm for stepwise regression.

12. Icing Thickness Prediction Model of Transmission Line Based on Linear Interpolation Method and Support Vector Machine
Claim: Linear interpolation can expand a limited icing dataset and improve SVM prediction during the initial icing-growth stage. The enhanced model reports an average relative error of 5.742% and outperforms the compared conventional BP and SVM models.

13. Icing Thickness Prediction Method for Overhead Transmission Lines Based on the NGO-VMD-GRU Model
Claim: Optimized variational mode decomposition followed by GRU-based component prediction improves modelling of non-stationary icing sequences. The paper reports a MAPE of 3.12%, lower than those of the compared LSTM, BP, and VMD-GRU models.

14. Research on Transmission Line Icing Prediction for Power System Based on Improved Snake Optimization Algorithm-Optimized Deep Hybrid Kernel Extreme Learning Machine
Claim: An improved snake optimization algorithm can optimize a deep hybrid-kernel extreme learning machine to produce more accurate ice-thickness predictions from meteorological variables such as temperature, humidity, wind, and precipitation.

15. Icing Thickness Prediction for Transmission Lines Using Attention-CNN-GRU with Real-Time Meteorological Observation Data
Claim: CNN captures short-term weather patterns, GRU represents longer temporal dependencies, and attention emphasizes influential meteorological variables. The model reports error reductions of 26.1% relative to GRU and 57.3% relative to LSTM.

16. Neural Network Prediction Model for Transmission Line Icing in Freezing Rain Based on Multi-Source Data Fusion
Claim: Fusing multiple sources of meteorological and icing-related information in a neural network provides a more comprehensive representation of freezing-rain conditions for transmission-line icing prediction.

17. IGWO-LSSVM Transmission Line Icing Prediction Model Based on Grey Correlation Analysis
Claim: Grey-correlation analysis selects relevant icing factors, while an improved grey wolf optimizer determines the parameters of an LSSVM, addressing inefficient parameter selection and improving icing-prediction performance.

18. Icing Thickness Prediction Model of Transmission Line Based on KPCA and INGO-BiLSTM
Claim: Kernel PCA reduces nonlinear and redundant input information, while an improved Northern Goshawk algorithm optimizes a BiLSTM to improve transmission-line icing-thickness prediction.

19. Online Ice Prediction Model for Transmission Lines Based on BOA_AOSVR
Claim: An optimized online support-vector-regression model improves icing prediction by addressing the low accuracy, randomness, and weak interference resistance associated with traditional single prediction models.

20. A Novel Deep Learning Model for Transmission Line Icing Thickness Prediction
Claim: A dedicated deep-learning model can learn the nonlinear relationship between icing thickness and monitored environmental or operational variables, providing a data-driven method for predicting overhead transmission-line icing thickness. The accessible source did not provide a detailed numerical comparison, so no specific performance value is claimed here.

Note: Paper titled "Analysis of Transmission Line Icing Prediction Based on CNN and Data Mining Technology" is retracted.

How can Spark-based big-data preprocessing, influencing-factor analysis, and deep convolutional learning be used to predict transmission-line icing thickness from historical icing and meteorological data, and which meteorological and operational variables are the most relevant predictors? Answer in short.

"""
max_new_tokens = 8192

# Note: Paper titled "Analysis of Transmission Line Icing Prediction Based on CNN and Data Mining Technology" is retracted.


class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, rank: int):
        super().__init__()
        self.original = original

        self.A = nn.Linear(original.in_features, rank, bias=False)
        self.B = nn.Linear(rank, original.out_features, bias=False)

        # Keep LoRA weights on same device/dtype as the wrapped layer.
        self.A.to(device=original.weight.device, dtype=original.weight.dtype)
        self.B.to(device=original.weight.device, dtype=original.weight.dtype)

        torch.nn.init.xavier_normal_(self.A.weight)
        torch.nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.original(x) + self.B(self.A(x))


def _attach_lora_layers(model, rank, target_modules):
    loras = []
    modules = list(model.named_modules())

    for name, module in modules:
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            if "." not in name:
                continue
            parent_name, attr_name = name.rsplit(".", 1)
            parent_module = model.get_submodule(parent_name)

            lora = LoRALinear(module, rank)
            setattr(parent_module, attr_name, lora)
            loras.append(lora)

    return loras


def _load_lora_state(loras, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    lora_state = ckpt.get("lora_state", ckpt)

    for i, lora in enumerate(loras):
        a_key = f"{i}.A"
        b_key = f"{i}.B"
        if a_key in lora_state and b_key in lora_state:
            lora.A.weight.data.copy_(lora_state[a_key].to(lora.A.weight.device, dtype=lora.A.weight.dtype))
            lora.B.weight.data.copy_(lora_state[b_key].to(lora.B.weight.device, dtype=lora.B.weight.dtype))


def load_model(base_model_name, model_name, lora_checkpoint_file=None):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    if lora_checkpoint_file:
        checkpoint_path = os.path.join(model_name, "lora_checkpoints", lora_checkpoint_file)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"LoRA checkpoint not found: {checkpoint_path}")

        loras = _attach_lora_layers(model, lora_rank, lora_target_modules)
        _load_lora_state(loras, checkpoint_path)
        print(f"Loaded LoRA checkpoint: {checkpoint_path}")

    return model


tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
model = load_model(base_model_name, model_name, lora_checkpoint_file)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        top_p=0.9,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )

generated_tokens = output.sequences[0][inputs.input_ids.shape[1]:]

result = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
)

print("Question:", prompt)
print("Answer:", result)

