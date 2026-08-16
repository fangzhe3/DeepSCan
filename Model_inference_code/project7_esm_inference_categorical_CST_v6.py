
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer

df = pd.read_csv("./data/v8_generative_output.csv")
sequences = df["TMC_seq"].tolist()

# model weights can be downloaded from hugginface: zfan3/esm2_t12_35M_category_CST_homolog_v6
model_checkpoint = "./esm2_t12_35M_category_CST_homolog_v6/checkpoint-10194"
num_labels = 4  # Add 1 since 0 can be a label

# load model from local directory, esm2_t12_35M_UR50D-SPTM_CM_35M_5epochs_split5
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
trainer = Trainer(model=model, tokenizer=tokenizer)

print("Let's use", torch.cuda.device_count(), "GPUs!")

input_sequence = sequences
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = tokenizer(input_sequence, return_tensors="pt", padding=True, truncation=True).to(device)
model.to(device)

# Make predictions
batch_size = 4

# Function to process in batches
def process_in_batches(inputs, model, batch_size):
    all_outputs = []
    for i in range(0, len(inputs["input_ids"]), batch_size):
        # Prepare the batch inputs
        batch = {k: v[i:i + batch_size].to(device) for k, v in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = model(**batch)
            all_outputs.append(outputs.logits)

    # Concatenate all the outputs
    return torch.cat(all_outputs)

# Run the batched inference
outputs = process_in_batches(inputs, model, batch_size)

predictions = torch.argmax(outputs, axis=1)
print("Predicted class:", predictions)

df["predicted outputs"] = outputs.tolist()
df["predicted_CST3_score"] = [x[3] for x in outputs.tolist()]
df["predicted_CST2_score"] = [x[2] for x in outputs.tolist()]
df["predicted class"] = predictions.tolist()

df.to_csv("v8_generative_output_predicted_v2.csv")

