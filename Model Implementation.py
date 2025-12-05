
!pip install -q transformers datasets sacrebleu sentencepiece evaluate kaggle torch accelerate

import os
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import torch
import datetime
import json
from evaluate import load as load_metric

KAGGLE_USERNAME = "********"  
KAGGLE_KEY = "***********************"        

os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
os.environ["KAGGLE_KEY"] = KAGGLE_KEY

print("✅ Kaggle API configured successfully.")

dataset_dir = "/content/Dataset"
dataset_zip = "/content/parallel-corpus-for-english-urdu-language.zip"

if not os.path.exists(dataset_dir):
    print("📥 Downloading dataset from Kaggle...")
    !kaggle datasets download -d zainuddin123/parallel-corpus-for-english-urdu-language -p /content
    !unzip -oq {dataset_zip} -d /content/
else:
    print("✅ Dataset already exists — skipping download.")

def find_file(base_dir, filename):
    for root, _, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

eng_path = find_file("/content", "english-corpus.txt")
ur_path = find_file("/content", "urdu-corpus.txt")

if not eng_path or not ur_path:
    raise FileNotFoundError("❌ Could not find english-corpus.txt or urdu-corpus.txt in extracted dataset.")

print(f"✅ Found files:\nEnglish: {eng_path}\nUrdu: {ur_path}")

with open(eng_path, encoding="utf-8") as f:
    english_lines = f.read().strip().split("\n")

with open(ur_path, encoding="utf-8") as f:
    urdu_lines = f.read().strip().split("\n")

df = pd.DataFrame({"en": english_lines, "ur": urdu_lines})
print(f"✅ Loaded {len(df)} sentence pairs.")

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

datasets = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
    "test": Dataset.from_pandas(test_df.reset_index(drop=True))
})

model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

tokenizer.src_lang = "en_XX"
tokenizer.tgt_lang = "ur_PK"

max_len = 128

def preprocess_function(examples):
    inputs = examples["en"]
    targets = examples["ur"]
    model_inputs = tokenizer(inputs, max_length=max_len, truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_len, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = datasets.map(preprocess_function, batched=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

hyperparams = {
    "model": model_name,
    "learning_rate": 3e-5,
    "batch_size": 8,
    "epochs": 5,
    "max_seq_length": max_len,
    "optimizer": "AdamW",
    "fp16": torch.cuda.is_available(),
}

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results_en_ur_mbart",
    eval_strategy="epoch" if "eval_strategy" in Seq2SeqTrainingArguments.__init__.__code__.co_varnames else "epoch",
    save_strategy="epoch",
    learning_rate=hyperparams["learning_rate"],
    per_device_train_batch_size=hyperparams["batch_size"],
    per_device_eval_batch_size=hyperparams["batch_size"],
    num_train_epochs=hyperparams["epochs"],
    predict_with_generate=True,
    fp16=hyperparams["fp16"],
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

if os.path.exists("./results_en_ur_mbart/checkpoint-last"):
    print("🔄 Resuming from last checkpoint...")
    train_output = trainer.train(resume_from_checkpoint="./results_en_ur_mbart/checkpoint-last")
else:
    train_output = trainer.train()

metric = load_metric("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

test_results = trainer.predict(tokenized_datasets["test"])

label_ids = np.where(test_results.label_ids != -100, test_results.label_ids, tokenizer.pad_token_id)

decoded_preds = tokenizer.batch_decode(test_results.predictions, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

preds, labels = postprocess_text(decoded_preds, decoded_labels)
bleu = metric.compute(predictions=preds, references=labels)

print(f"\n✅ BLEU Score: {bleu['score']:.2f}")

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs("final_model_en_ur_mbart", exist_ok=True)

model.save_pretrained("final_model_en_ur_mbart")
tokenizer.save_pretrained("final_model_en_ur_mbart")

# Save logs
log_path = f"experiment_log_{timestamp}.txt"
with open(log_path, "w", encoding="utf-8") as f:
    f.write("===== English→Urdu Translation Experiment Log =====\n\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Device: {device}\n\n")
    f.write("---- Hyperparameters ----\n")
    f.write(json.dumps(hyperparams, indent=4))
    f.write("\n\n---- Dataset Info ----\n")
    f.write(f"Total Samples: {len(df)}\n")
    f.write(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}\n\n")
    f.write("---- Training Summary ----\n")
    f.write(str(train_output) + "\n\n")
    f.write("---- Evaluation ----\n")
    f.write(f"BLEU Score: {bleu['score']:.2f}\n\n")
    f.write("---- Sample Translations ----\n")
    for i in range(10):
        f.write(f"EN: {test_df.iloc[i]['en']}\n")
        f.write(f"UR Pred: {preds[i]}\n")
        f.write(f"UR True: {labels[i][0]}\n\n")

# Save translations
pd.DataFrame({
    "English": test_df["en"].values[:len(preds)],
    "Urdu_True": [l[0] for l in labels],
    "Urdu_Pred": preds
}).to_csv(f"translations_{timestamp}.csv", index=False)

print(f"\n📝 All results saved to: {log_path}")
print(f"📁 Model saved to: ./final_model_en_ur_mbart/")
print(f"📄 Translations saved to: translations_{timestamp}.csv")
print("\n🎉 Training, evaluation, and logging complete!")
