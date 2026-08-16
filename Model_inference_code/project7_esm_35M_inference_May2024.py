
import pandas as pd
from transformers import pipeline
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
from evaluate import load
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


df = pd.read_csv("./data/training_dataset_id50.csv")
model_checkpoint = "facebook/esm2_t12_35M_UR50D"
num_labels = 3

# model weights can be downloaded from huggingface: zfan3/esm2_t12_35M_UR50D-SPTM_CM_split20epoch2_split20_1GPU
model = AutoModelForSequenceClassification.from_pretrained("./esm2_t12_35M_UR50D-SPTM_CM_35M_split5epoch2_split20_3epochs/checkpoint-4386", num_labels=num_labels)

print("Let's use", torch.cuda.device_count(), "GPUs!")

label0_sequences = df.loc[df["label"]==0," sequenceValue"].tolist()
label0_labels = [0 for protein in label0_sequences]

label1_sequences = df.loc[df["label"]==1," sequenceValue"].tolist()
label1_labels = [1 for protein in label1_sequences]

label2_sequences = df.loc[df["label"]==2," sequenceValue"].tolist()
label2_labels = [2 for protein in label2_sequences]

sequences = label0_sequences + label1_sequences + label2_sequences
labels = label0_labels + label1_labels + label2_labels

train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, labels, test_size=0.20, shuffle=True)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

train_tokenized = tokenizer(train_sequences)
test_tokenized = tokenizer(test_sequences)

train_dataset = Dataset.from_dict(train_tokenized)
test_dataset = Dataset.from_dict(test_tokenized)

train_dataset = train_dataset.add_column("labels", train_labels)
test_dataset = test_dataset.add_column("labels", test_labels)

model_name = model_checkpoint.split("/")[-1]
batch_size = 2

args = TrainingArguments(
    f"{model_name}-SPTM_CM_split20epoch2_split20_3epochs_1GPU_test",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
#    deepspeed="ds_config_zero3.json"
#    push_to_hub=True,
)

metric = load("accuracy")


trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.predict(test_dataset)
