# Install dependencies which are already pre-installed on Kaggle, but im doing this just in case
# i tried using these but everytime there is an error of conflicting versions or broken versions or versions that dont work together , so im simply just writing this 
#!pip uninstall -y peft transformers
#!pip install -q transformers==4.38.2 peft==0.10.0 accelerate datasets evaluate

# Import are going here
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
import evaluate
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import numpy as np
import math
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm


#loading the metrics here 
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

# Define the  compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=predictions, references=labels)["f1"],
        "precision": precision_metric.compute(predictions=predictions, references=labels)["precision"],
        "recall": recall_metric.compute(predictions=predictions, references=labels)["recall"],
    }
# here im loading both the wikitext and the openwebtext datsets 
wiki = load_dataset("wikitext", "wikitext-2-raw-v1")
openweb = load_dataset("openwebtext",trust_remote_code=True)

# im reduing the size for kaggle cuz this is taking way too long for some reason and im very tired 
openweb["train"] = openweb["train"].select(range(5000))
wiki["train"] = wiki["train"].select(range(5000))
wiki["validation"] = wiki["validation"].select(range(1000))

# im combining both my datasets 
combined_train = concatenate_datasets([wiki["train"], openweb["train"]])
combined_val = concatenate_datasets([wiki["validation"], openweb["train"].select(range(1000))])

# loading the tokenizer and definfinthe function for it 
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# using the tokenizer
tokenized_datasets = {
    "train": combined_train.map(tokenize_function, batched=True, remove_columns=["text"]),
    "validation": combined_val.map(tokenize_function, batched=True, remove_columns=["text"])
}
# im loading th egpt 2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# training all my arguments
training_args = TrainingArguments(
    # output_dir="./results",  all this here , it showed error , a bunch of em so i commented them 
    # evaluation_strategy="epoch",
    # save_strategy="epoch",
    # num_train_epochs=1,  
    # per_device_train_batch_size=4,
    # per_device_eval_batch_size=4,
    # logging_dir="./logs",
    # logging_steps=100,
    #report_to="none"
    output_dir="./results",
    num_train_epochs=1,  
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir="./logs",
    logging_steps=100,
)

# data collator for lm goes here
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# implementing said trainer
trainer.train()

# evaluate perplexity
eval_results = trainer.evaluate()
perplexity = math.exp(eval_results["eval_loss"])
print(f">>> Perplexity: {perplexity:.4f}")

# Top-k Accuracy is used here 
def compute_top_k_accuracy(model, dataset, tokenizer, k=5, max_batches=100):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1)
    correct = 0
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=f"Computing Top-{k} Accuracy")):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].squeeze(0)
            if input_ids.shape[0] < 2:
                continue
            input_seq = input_ids[:-1].unsqueeze(0)
            target_token = input_ids[-1]
            outputs = model(input_seq)
            logits = outputs.logits[:, -1, :]
            top_k_probs = torch.topk(logits, k=k, dim=-1).indices.squeeze()
            if target_token in top_k_probs:
                correct += 1
            total += 1
    accuracy = correct / total if total > 0 else 0.0
    print(f">>> Top-{k} Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy

compute_top_k_accuracy(model, tokenized_datasets["validation"], tokenizer, k=5)

def predict_next_word(prompt, top_k=5):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
    predicted_words = [tokenizer.decode([idx]).strip() for idx in top_k_indices[0]]
    return predicted_words

# im trying a few example using the model
print(">>> Predictions:", predict_next_word("The capital of France is"))
print(">>> Predictions:", predict_next_word("Deep learning models are"))
