import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset

# Load dataset
train_df = pd.read_csv('essay-br/splits/training.csv')
test_df = pd.read_csv('essay-br/splits/testing.csv')

# Convert numerical scores to strings (since GPT generates text)
train_df['score'] = train_df['score'].astype(str)
test_df['score'] = test_df['score'].astype(str)

# Initialize tokenizer and model (DistilGPT-2)
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2')

# Set the padding token to be the same as the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token

# Combine the essay text and score (as a single task for GPT to learn to predict the score)
train_df['input_text'] = train_df['essay'] + ' [SEP] ' + train_df['score']
test_df['input_text'] = test_df['essay'] + ' [SEP] ' + test_df['score']

# Tokenize the input data
def tokenize_function(examples):
    tokens = tokenizer(examples["input_text"], truncation=True, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()  # Set the labels to be the same as input_ids
    return tokens

# Convert pandas DataFrame to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenize the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch (including labels)
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # Adjust for memory
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    log_level="info"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Fine-tune the model
trainer.train()


