import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset

# Load dataset
train_df = pd.read_csv('essay-br/splits/training.csv')
test_df = pd.read_csv('essay-br/splits/testing.csv')

# Convert numerical scores to strings (as explained earlier)
train_df['score'] = train_df['score'].astype(str)
test_df['score'] = test_df['score'].astype(str)

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Combine the essay text and score
train_df['input_text'] = train_df['essay'] + ' [SEP] ' + train_df['score']
test_df['input_text'] = test_df['essay'] + ' [SEP] ' + test_df['score']

# Tokenize the input data
def tokenize_function(examples):
    return tokenizer(examples["input_text"], truncation=True, padding="max_length")

# Convert pandas DataFrame to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenize the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Training function
def main():
    # Fine-tune the model
    trainer.train()

# Ensure the script doesn't automatically execute when imported
if __name__ == "__main__":
    main()
