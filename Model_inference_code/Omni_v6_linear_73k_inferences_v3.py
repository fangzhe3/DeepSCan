
import numpy as np
import random
import pandas as pd
from pathlib import Path
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # set before import torch
import torch
import torch.nn as nn
torch.use_deterministic_algorithms(True)

import torch.distributed as dist
import torch.nn.functional as F
from transformers import Trainer, DataCollatorWithPadding
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoModelForTokenClassification
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, accuracy_score
from scipy.stats import spearmanr, pearsonr
import wandb


def set_seed(seed):
    random.seed(seed)                     # Python RNG
    np.random.seed(seed)                  # NumPy RNG
    torch.manual_seed(seed)               # CPU RNG
    torch.cuda.manual_seed_all(seed)      # All GPU RNGs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  # Enforce deterministic ops (PyTorch ≥ 1.8)

set_seed(91)

Task = "Omni"
Test_run = False
head_type = "linear" # default linear, results from linear!
# head_types = ["linear_attn2", "dense", "mlp_relu", "mlp_gelu", "residual_mlp", "transformer_head"]
# head_type = "linear"
ExpWeightedLoss = False
TwoStage = False

if Task == "Topo":
    target_label = "qseqid_Topology"
    folder_path = Path("/home/zf77/palmer_scratch/Genesis_quant/data/Omni_dataset_v4")
    csv_files = folder_path.glob("*.csv")
elif Task == "TM":
    folder_path = Path("/home/zf77/palmer_scratch/Genesis_quant/data/Omni_dataset_v4")
    csv_files = folder_path.glob("*.csv")
elif Task == "Omni":
    folder_path = Path(f"/data/73k")
#    csv_files = folder_path.glob("Omni_73k_AddZero_Notest_qseqid_v5.csv")
    csv_files = folder_path.glob("*.csv")

def make_name_variable(prefix, head_type):
    return f"5in1_OneStage_223half22_{prefix}_{head_type}_v6"

lambdas_raw = torch.tensor([2.0, 2.0, 3.5, 2.0, 2.0], dtype=torch.float32)
lambdas_norm = torch.softmax(lambdas_raw, dim=0)
print("lambdas_norm [2.0, 2.0, 3.5, 2.0, 2.0]:", lambdas_norm)
# lambdas_norm: tensor([0.0107, 0.0039, 0.9639, 0.0107, 0.0107])

def make_TM_label(df):
    labels = []
    FL_list = ["3.4k_FL","GO611_FL", "CST_FL", "FL"]
    hTMC_list = ["hTMC", "CST_hTMC", "3.4k_hTMC"]
    for idx, row in df.iterrows():
        seq = row["cleaned_sequence"]
        seq_label = np.zeros(len(seq), dtype=np.int64)
        if (row["id_label"]==1) & (row["Category2"] in FL_list) & pd.notna(row["TM_begin"]):
            TM_begin = int(row["TM_begin"] - 1)
            TM_end = int(row["TM_end"])
            seq_label[TM_begin:TM_end] = 1
            N15_begin = max(TM_begin-15, 0)
            seq_label[N15_begin:TM_begin] = 2
            C15_end = min(TM_end+15,len(seq))
            seq_label[TM_end:C15_end] = 2
            labels.append(seq_label)
        elif (row["id_label"]==1) & (row["Category2"] in hTMC_list) & pd.notna(row["TM_length"]) & pd.notna(row["hinge_length"]):
            hinge_length = int(row["hinge_length"])
            TM_length = int(row["TM_length"])
            seq_label[hinge_length:hinge_length+TM_length] = 1
            N15_begin = max(hinge_length-15,0)
            C15_end = min(hinge_length+TM_length+15,len(seq))
            seq_label[N15_begin:hinge_length] = 2
            seq_label[hinge_length + TM_length:C15_end] = 2
            labels.append(seq_label)
        else:
            labels.append(seq_label)
    return labels


from transformers import DataCollatorForTokenClassification
import torch

class UnifiedPaddingCollator:
    def __init__(self, tokenizer=None, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(self, features):
        batch = {}

        # Extract input_ids and labels
        input_ids_list = [torch.tensor(f["input_ids"]) for f in features]
        labels_list = [torch.tensor(f["labels"]) for f in features]

        # Compute max length across both input_ids and labels
        max_len = max(max(len(ids), len(lbl)) for ids, lbl in zip(input_ids_list, labels_list))
        max_len_ids = max_len-4

        # Pad inputs
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else 0
        batch["input_ids"] = pad_sequence(
            [self._pad_tensor(t, max_len_ids, pad_token_id) for t in input_ids_list],
            batch_first=True
        )
        batch["attention_mask"] = pad_sequence(
            [self._pad_tensor(torch.ones_like(t), max_len_ids, 0) for t in input_ids_list],
            batch_first=True
        )

        # Pad labels
        batch["labels"] = pad_sequence(
            [self._pad_tensor(lbl, max_len, self.ignore_index) for lbl in labels_list],
            batch_first=True
        )

        return batch

    def _pad_tensor(self, tensor, target_len, pad_val):
        pad_len = target_len - tensor.size(0)
        if pad_len > 0:
            padding = torch.full((pad_len,), pad_val, dtype=tensor.dtype)
            return torch.cat([tensor, padding], dim=0)
        else:
            return tensor

class ExpWeightedMSELoss(nn.Module):
    def __init__(self, alpha=1, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, y_pred, y_true):
        mask = (y_true != self.ignore_index)
        if mask.any():
            y_pred = y_pred[mask]
            y_true = y_true[mask]
            weights = torch.exp(self.alpha * y_true)
#            skipped = (~mask).sum().item()
#            print(f"[Loss] Skipped {skipped} invalid labels")
            return torch.mean(weights * (y_pred - y_true) ** 2)
        else:
            # No valid labels in batch, return safe zero loss
            return (y_pred - y_pred).sum() * 0.0

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

    def forward(self, hidden_states, mask=None, input_ids=None, cls_token_id=None, eos_token_id=None):
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
        return pooled, attn_weights

class Omni5in1(nn.Module):
    def __init__(self, base_model, head_type="linear", alpha=1, ignore_index=-100):
        super().__init__()
        self.esm = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        self.ignore_index = ignore_index

        self.hidden_size = base_model.config.hidden_size
        self.attn_pool1 = AttentionPooling2(self.hidden_size)
        self.attn_pool2 = AttentionPooling2(self.hidden_size)
        self.attn_pool3 = AttentionPooling2(self.hidden_size)
        self.attn_pool4 = AttentionPooling2(self.hidden_size)
        self.head_type = head_type.lower()
        self.dropout = nn.Dropout(0.1)

        

        if ExpWeightedLoss:
            self.quant_loss = ExpWeightedMSELoss(alpha=alpha, ignore_index=ignore_index)
        else:
            self.quant_loss = nn.MSELoss()
        self.id_loss = nn.BCEWithLogitsLoss()
        self.cst_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.tm_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.topo_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

#        lambdas_raw = torch.tensor([2.0, 1.0, 6.5], dtype=torch.float32)
#        lambdas_norm = torch.softmax(lambdas_raw, dim=0)
#        self.register_buffer("lambdas_norm", lambdas_norm)
        self.id_classifier = nn.Linear(self.hidden_size, 1)
        self.cst_classifier = nn.Linear(self.hidden_size, 6)
        self.topo_classifier = nn.Linear(self.hidden_size, 3)
        self.regressor = nn.Linear(self.hidden_size, 1)

        if self.head_type == "linear":
            self.tm_classifier = nn.Linear(self.hidden_size, 3)

        elif self.head_type == "mlp_relu":
            self.tm_classifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
#                nn.Dropout(0.1),
                nn.Linear(self.hidden_size, 3)
            )

        elif self.head_type == "dense":
            self.tm_classifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Linear(self.hidden_size, 3)
            )

        elif self.head_type == "mlp_gelu":
            self.tm_classifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
#                nn.Dropout(0.1),
                nn.Linear(self.hidden_size, 3)
            )

        elif self.head_type == "residual_mlp":
            self.tm_classifier = nn.Sequential(
                nn.Linear(self.hidden_size, 1920),
                nn.GELU(),
                nn.Linear(1920, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, 3)
            )

#        elif self.head_type == "attention_pooling2":
#            self.head = nn.Linear(self.hidden_size, 1)

        elif self.head_type == "transformer_head":
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=8, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.tm_classifier = nn.Linear(self.hidden_size, 3)

        else:
            raise ValueError(f"Unsupported head_type: {self.head_type}")

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        x1, attn2_weights1 = self.attn_pool1(hidden_states = hidden,
                               mask=attention_mask,
                               input_ids=input_ids,
                               cls_token_id=self.cls_token_id,
                               eos_token_id=self.eos_token_id)

        x2, attn2_weights2 = self.attn_pool2(hidden_states = hidden,
                           mask=attention_mask,
                           input_ids=input_ids,
                           cls_token_id=self.cls_token_id,
                           eos_token_id=self.eos_token_id)

        x3, attn2_weights3 = self.attn_pool3(hidden_states = hidden,
                           mask=attention_mask,
                           input_ids=input_ids,
                           cls_token_id=self.cls_token_id,
                           eos_token_id=self.eos_token_id)

        x4, attn2_weights4 = self.attn_pool4(hidden_states = hidden,
                           mask=attention_mask,
                           input_ids=input_ids,
                           cls_token_id=self.cls_token_id,
                           eos_token_id=self.eos_token_id)

        if self.head_type in ["linear", "dense", "mlp_relu", "mlp_gelu", "residual_mlp"]:
            tm_logits = self.tm_classifier(hidden)

        elif self.head_type == "transformer_head":
            transformed = self.transformer(hidden, src_key_padding_mask=(attention_mask == 0))
            tm_logits = self.tm_classifier(transformed)

        else:
            raise RuntimeError("Unsupported head type in forward pass.")

        id_logits = self.id_classifier(x1).squeeze(-1)
        cst_logits = self.cst_classifier(x2)
        quant_preds = self.regressor(x3).squeeze(-1)
        topo_logits = self.topo_classifier(x4)

        if labels is not None:
            id_labels = labels[:,0].float()
            cst_labels = labels[:,1].long().to(id_logits.device)
            quant_labels = labels[:,2].float().to(cst_logits.device)
            topo_labels = labels[:,3].long().to(topo_logits.device)
            tm_labels = labels[:,4:].long().to(tm_logits.device)

            id_mask = (id_labels != self.ignore_index)
            cst_mask = (cst_labels != self.ignore_index)
            quant_mask = (quant_labels != self.ignore_index)
            topo_mask = (topo_labels != self.ignore_index)
            tm_mask = (tm_labels != self.ignore_index)

#            cst_mask_combined = (id_mask & cst_mask)
#            quant_mask_combined = (id_mask & quant_mask)
            id_logits_masked = id_logits[id_mask]
            id_labels_masked = id_labels[id_mask]

            cst_logits_masked = cst_logits[cst_mask]
            cst_labels_masked = cst_labels[cst_mask]

            quant_preds_masked = quant_preds[quant_mask]
            quant_labels_masked = quant_labels[quant_mask]

            topo_logits_masked = topo_logits[topo_mask]
            topo_labels_masked = topo_labels[topo_mask]

            tm_logits = tm_logits[:,1:-1,:]

            tm_logits_masked = tm_logits[tm_mask]
            tm_labels_masked = tm_labels[tm_mask]

            loss_id = self.id_loss(id_logits_masked, id_labels_masked)
            loss_topo = self.topo_loss(topo_logits_masked, topo_labels_masked)
            loss_tm = self.tm_loss(tm_logits_masked, tm_labels_masked)

            id_probs = torch.sigmoid(id_logits)
            id_preds = (id_probs >= 0.5).int()
#            id1_mask = (id_preds!=0)

            if TwoStage:
                cst_logits_id0 = cst_logits.clone()
                mask = (id_preds==0)
                replacement = torch.tensor([15, -3, -3, -3, -3, -3], \
                                                           dtype=cst_logits_id0.dtype, \
                                                           device=cst_logits_id0.device)
                cst_logits_id0[mask] = replacement.unsqueeze(0).expand(mask.sum(), -1)
                cst_logits_id0_masked = cst_logits_id0[cst_mask]

                quant_preds_id0 = quant_preds.clone()
                quant_preds_id0[id_preds==0] = 0
                quant_preds_id0_masked = quant_preds_id0[quant_mask]

                loss_cst = self.cst_loss(cst_logits_id0_masked, cst_labels_masked)
                loss_quant = self.quant_loss(quant_preds_id0_masked, quant_labels_masked)
            else:
                loss_cst = self.cst_loss(cst_logits_masked, cst_labels_masked)
                loss_quant = self.quant_loss(quant_preds_masked, quant_labels_masked)

            lambdas_loss_id = loss_id * 0.1179
            lambdas_loss_cst = loss_cst * 0.1179
            lambdas_loss_quant = loss_quant * 0.5284
            lambdas_loss_topo = loss_topo * 0.1179
            lambdas_loss_tm = loss_tm * 0.1179
            # lambdas_norm [2.0, 2.0, 3.5, 2.0, 2.0]: tensor([0.1179, 0.1179, 0.5284, 0.1179, 0.1179])

            total_loss = (lambdas_loss_id + lambdas_loss_cst + lambdas_loss_quant + lambdas_loss_topo + lambdas_loss_tm)
            # 0.0471, 0.0064, 0.9465
            # loss_id 0.73, loss_cst 1.92, loss_quant 0.009

            return {
            "loss": total_loss,
            "id_logits": id_logits,
            "cst_logits": cst_logits,
            "quant_preds": quant_preds,
            "topo_logits": topo_logits,
            "tm_logits": tm_logits,
#            "attn_weights": attn2_weights,
            "lambdas_loss_id": lambdas_loss_id.detach(),
            "lambdas_loss_cst": lambdas_loss_cst.detach(),
            "lambdas_loss_quant": lambdas_loss_quant.detach(),
            "lambdas_loss_topo": lambdas_loss_topo.detach(),
            "lambdas_loss_tm": lambdas_loss_tm.detach()
#            "lambdas_norm": lambdas_norm
            }
        else:
            return {
            "id_logits": id_logits,
            "cst_logits": cst_logits,
            "quant_preds": quant_preds,
            "topo_logits": topo_logits,
            "tm_logits": tm_logits
#            "attn_weights": attn2_weights,
#            "lambdas_norm": lambdas_norm
            }

def get_layerwise_lr_params(model, base_lr=3e-5, decay=0.95):
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = []

    encoder_layers = model.esm.encoder.layer
    num_layers = len(encoder_layers)

    for i, layer in enumerate(encoder_layers):
        lr = base_lr * (decay ** (num_layers - i - 1))

        grouped_params.append({
            "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
            "lr": lr,
        })
        grouped_params.append({
            "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lr,
        })

    # Add ESM embeddings (optional)
    grouped_params.append({
        "params": model.esm.embeddings.parameters(),
        "lr": base_lr * (decay ** num_layers)
    })

    # Add custom head
    grouped_params.append({
        "params": model.head.parameters(),  # or model.regressor, etc.
        "lr": base_lr,
        "weight_decay": 0.01,
    })

    return grouped_params

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)

        # The main loss used for backprop
        loss = outputs["loss"]

        # Initialize or update loss buffer
        self.loss_buffer = getattr(self, "loss_buffer", [])
        self.loss_buffer.append(loss.item())
        if len(self.loss_buffer) > 10:
            self.loss_buffer.pop(0)

        # Log every 10 steps: smoothed + raw losses
        if self.state.global_step % 100 == 0:
            wandb.log({
                "total_loss": loss.item(),
                "total_loss_smooth": sum(self.loss_buffer) / len(self.loss_buffer),
                "lambdas_loss_id": outputs["lambdas_loss_id"].item(),
                "lambdas_loss_cst": outputs["lambdas_loss_cst"].item(),
                "lambdas_loss_quant": outputs["lambdas_loss_quant"].item(),
                "lambdas_loss_topo": outputs["lambdas_loss_topo"].item(),
                "lambdas_loss_tm": outputs["lambdas_loss_tm"].item()
            })

        return (loss, outputs) if return_outputs else loss

class LLRDTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            optimizer_grouped_parameters = get_layerwise_lr_params(self.model, base_lr=self.args.learning_rate)
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        return self.optimizer

!nvidia-smi

# iteration 5 not the best, CST101 below 0.5
# best is iteration 3, CST101 0.5544

from safetensors.torch import load_file
import math
import time
import gc

head_type="linear"
# model_name = "./Genesis_quant_v18_8k_linear_padding_head_benchmarking/checkpoint-3614"
model_name = "facebook/esm2_t12_35M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)
model = Omni5in1(base_model, head_type=head_type)

state_dict = load_file("../DeepSCan_models/5in1_OneStage_223half22_Omni_73k_linear_v6_head_benchmarking/checkpoint-33158/model.safetensors")
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)

n_total = len(sequences)          # ~150_000
n_chunks = 1                      # you want 5 batches
chunk_size = math.ceil(n_total / n_chunks)

print(f"Total sequences: {n_total}, chunk_size: {chunk_size}")

# ---- model/device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def process_in_batches(inputs, model, batch_size):
    all_outputs = []
    all_cst_logits = []
    all_logits = []
    all_attn2_weights = []
    for i in range(0, len(inputs["input_ids"]), batch_size):
        # Prepare the batch inputs
        batch = {k: v[i:i + batch_size].to(device) for k, v in inputs.items()}

        # Perform inference
        with torch.no_grad():
            results = model(**batch)
            cst_logits = results["cst_logits"].to(device)
            all_cst_logits.append(cst_logits.detach().cpu())

    all_cst_logits2 = torch.cat(all_cst_logits, dim=0) # concat batch 10x8

    return all_cst_logits2

# batch_size = 8 #64

inner_batch_size = 64  # adjust based on GPU memory
all_cst_logits3=[]

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
            cst_logits = results["cst_logits"].detach().cpu().numpy()
            for cst_logit in cst_logits:
#                chunk_cls.append(cls)
                all_cst_logits3.append(cst_logit)
    del batch_inputs, results, cst_logits
    torch.cuda.empty_cache()
    gc.collect()
    print(f"{head_type} dataset{chunk_n} model prediction complete, file saved")

    t2 = time.time()
    print(f"dataset{chunk_n} Elapsed:", t2 - t1, "seconds")

np.save(f"/output/Omni_v6_linear_73k_Omni1399_all_cst_logits3.npy", all_cst_logits3)
np.savetxt("/output/Omni_v6_linear_73k_Omni1399_all_cst_logits3.csv", all_cst_logits3, delimiter=",")
all_cst_category = np.argmax(all_cst_logits3,axis=1)

np.savetxt("/output/Omni_v6_linear_73k_Omni1399_all_cst_category.csv", all_cst_category, delimiter=",")
