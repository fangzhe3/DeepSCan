import numpy as np
import random
import pandas as pd
import os
import time
import math
import gc
t0 = time.time()

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # set before import torch
import torch
torch.use_deterministic_algorithms(True)

import torch.nn as nn
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModel
from transformers import Trainer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from safetensors.torch import load_file
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr
from datasets import Dataset
# import wandb

def set_seed(seed):
    random.seed(seed)                     # Python RNG
    np.random.seed(seed)                  # NumPy RNG
    torch.manual_seed(seed)               # CPU RNG
    torch.cuda.manual_seed_all(seed)      # All GPU RNGs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  # Enforce deterministic ops (PyTorch ≥ 1.8)

set_seed(91)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden_size)
        attn_weights = torch.softmax(self.attn(hidden_states), dim=1)  # (batch, seq_len, 1)
        pooled = (attn_weights * hidden_states).sum(dim=1)  # (batch, hidden_size)
        return pooled

class AttentionPooling2(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, mask=None, input_ids=None, cls_token_id=None, eos_token_id=None, return_weights=False):
        # hidden_states: (batch, seq_len, hidden_size)
        attn_scores = self.attn(hidden_states).squeeze(-1)  # (batch, seq_len)

        if mask is not None:
            # Optionally zero out CLS and EOS tokens by updating the mask
            if input_ids is not None and cls_token_id is not None and eos_token_id is not None:
                cls_mask = (input_ids == cls_token_id)
                eos_mask = (input_ids == eos_token_id)
                ignore_mask = cls_mask | eos_mask
                mask = mask.masked_fill(ignore_mask, 0)

            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(hidden_states * attn_weights.unsqueeze(-1), dim=1)
        if return_weights:
            return pooled, attn_weights
        else:
            return pooled

class ESM2Regressor(nn.Module):
    def __init__(self, base_model, head_type="linear"):
        super().__init__()
        self.esm = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        hidden_size = base_model.config.hidden_size
        self.attn_pool = AttentionPooling(hidden_size)
        self.attn_pool2 = AttentionPooling2(hidden_size)
        self.head_type = head_type.lower()
        self.dropout = nn.Dropout(0.1)

        if self.head_type == "linear":
            self.head = nn.Linear(hidden_size, 1)

        elif self.head_type == "mlp_relu":
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
#                nn.Dropout(0.1),
                nn.Linear(hidden_size, 1)
            )

        elif self.head_type == "dense":
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Linear(hidden_size, 1)
            )

        elif self.head_type == "mlp_gelu":
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
#                nn.Dropout(0.1),
                nn.Linear(hidden_size, 1)
            )

        elif self.head_type == "residual_mlp":
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 1920),
                nn.GELU(),
                nn.Linear(1920, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1)
            )

        elif self.head_type == "attention_pooling":
            self.head = nn.Linear(hidden_size, 1)

        elif self.head_type == "attention_pooling2":
            self.head = nn.Linear(hidden_size, 1)

        elif self.head_type == "transformer_head":
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.head = nn.Linear(hidden_size, 1)

        else:
            raise ValueError(f"Unsupported head_type: {self.head_type}")

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        return_attn_weights = (labels is None)

        if self.head_type in ["linear", "dense", "mlp_relu", "mlp_gelu", "residual_mlp"]:
            x = self.attn_pool2(hidden_states = hidden,
                               mask=attention_mask,
                               input_ids=input_ids,
                               cls_token_id=self.cls_token_id,
                               eos_token_id=self.eos_token_id,
                               return_weights=return_attn_weights)
#            x = self.dropout(x)

        elif self.head_type == "attention_pooling":
            x = self.attn_pool(hidden)
#            x = self.dropout(x)

        elif self.head_type == "attention_pooling2":
            x = self.attn_pool2(hidden_states = hidden,
                               mask=attention_mask,
                               input_ids=input_ids,
                               cls_token_id=self.cls_token_id,
                               eos_token_id=self.eos_token_id,
                               return_weights=return_attn_weights)
#            x = self.dropout(x)

        elif self.head_type == "transformer_head":
            transformed = self.transformer(hidden)
            x = self.attn_pool2(hidden_states = transformed,
                               mask=attention_mask,
                               input_ids=input_ids,
                               cls_token_id=self.cls_token_id,
                               eos_token_id=self.eos_token_id,
                               return_weights=return_attn_weights)
#            x = self.dropout(cls_token)

        else:
            raise RuntimeError("Unsupported head type in forward pass.")

        if isinstance(x, tuple):
            x, attn_weights = x
        else:
            attn_weights = None
        logits = self.head(x).squeeze(-1)
        loss = None
        if labels is not None:
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels, dtype=torch.float, device=logits.device)
            else:
                labels = labels.float().to(logits.device)

            loss_fn = nn.MSELoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
  #          return {"logits": logits, "hidden": hidden, "attn_pool2_weights": attn_weights}
            return {"hidden": hidden}


head_type="linear"
# model_name = "./Genesis_quant_v18_8k_linear_padding_head_benchmarking/checkpoint-3614"
model_name = "facebook/esm2_t12_35M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)
model = ESM2Regressor(base_model, head_type=head_type)
# attn2

state_dict = load_file("/Genesis_quant_v18_8k_linear_padding_head_benchmarking/checkpoint-3614/model.safetensors")
# state_dict = load_file("./8k_head_benchmark2_attention_pooling2/checkpoint-1807/model.safetensors")
# model.load_state_dict(state_dict)


missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)

model.eval()
print("Let's use", torch.cuda.device_count(), "GPUs!")


all_datasets = pd.read_csv("/Quant_8k_training_set.csv")

sequences = all_datasets[0:]["sequence"].tolist()
# sequences = sequences[0:18000]
n_total = len(sequences)          # ~150_000
n_chunks = 2                      # you want 5 batches
chunk_size = math.ceil(n_total / n_chunks)

print(f"Total sequences: {n_total}, chunk_size: {chunk_size}")

# ---- model/device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)
model.to(device)
model.eval()

def process_in_batches(inputs, model, batch_size):
    all_outputs = []
    all_hidden = []
    all_logits = []
    all_attn2_weights = []
    for i in range(0, len(inputs["input_ids"]), batch_size):
        # Prepare the batch inputs
        batch = {k: v[i:i + batch_size].to(device) for k, v in inputs.items()}

        # Perform inference
        with torch.no_grad():
            results = model(**batch)
            hidden = results["hidden"].to(device)
            all_hidden.append(hidden.detach().cpu())

    all_hidden2 = torch.cat(all_hidden, dim=0) # concat batch 10x8

    return all_hidden2

# batch_size = 8 #64

inner_batch_size = 64  # adjust based on GPU memory
all_cls=[]

for chunk_idx in range(n_chunks):
    start = chunk_idx * chunk_size
    end = min(n_total, (chunk_idx + 1) * chunk_size)
    seq_chunk = sequences[start:end]

    print(f"\nProcessing chunk {chunk_idx+1}/{n_chunks}: indices [{start}, {end}) "
          f"({len(seq_chunk)} sequences)")

    chunk_outputs = []
    chunk_n = chunk_idx+1
    t1 = time.time()
#   chunk_cls=[]
    for i in range(0, len(seq_chunk), inner_batch_size):
        batch_seqs = seq_chunk[i : i + inner_batch_size]

#        batch = {k: v[i:i + batch_size].to(device) for k, v in inputs.items()}
        # ---- tokenize / encode (replace with your own collator if needed) ----
        batch_inputs = tokenizer(
            batch_seqs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

#        all_hidden2 = process_in_batches(test_tokenized, model, batch_size)

        with torch.inference_mode():
            results = model(**batch_inputs)
            hiddens = results["hidden"].detach().cpu().numpy()
            for hidden in hiddens:
                cls = hidden[0]
#                chunk_cls.append(cls)
                all_cls.append(cls)
    del batch_inputs, results, hiddens
    torch.cuda.empty_cache()
    gc.collect()
    print(f"{head_type} dataset{chunk_n} model prediction complete, file saved")

    t2 = time.time()
    print(f"dataset{chunk_n} Elapsed:", t2 - t1, "seconds")

np.save(f"/Quant8k_training_8klinear_predicted_all_cls.npy", all_cls)
