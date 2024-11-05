from train_gpt import trainer, test_dataset, tokenizer, test_df
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import numpy as np

# Select a smaller subset of the test dataset (e.g., 5 examples) for faster evaluation
small_test_dataset = test_dataset.select(range(10))  # Adjust the number if needed

# Set up TrainingArguments to force CPU usage
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=1,  # Smallest batch size to reduce memory usage
    no_cuda=True  # Force evaluation to use the CPU instead of MPS
)

# Reload the Trainer with updated training arguments for CPU
trainer = Trainer(
    model=trainer.model,  # Use the same model
    args=training_args,
)

# Start evaluation
print("Starting predictions...")
predictions = trainer.predict(small_test_dataset)
print("Predictions completed.")
print("Prediction structure:", predictions.predictions)  # Inspect the structure of predictions

# Extract predicted scores (assuming GPT generates the score at the end of the sequence)
predicted_scores = []
print("Starting decoding...")

for idx, pred in enumerate(predictions.predictions):
    print(f"Decoding prediction {idx + 1}/{len(predictions.predictions)}...")

    # Select the token with the highest logit for each position
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()  # Convert tensor to numpy array if needed

    # Get the index of the highest logit (predicted token ID) for each position in the sequence
    token_ids = np.argmax(pred, axis=-1).tolist() if pred.ndim == 2 else pred  # Handle nested structure

    # Decode and extract the score
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    predicted_score = decoded_text.split('[SEP]')[-1].strip()

    try:
        predicted_scores.append(int(predicted_score))  # Convert to integer
    except ValueError:
        predicted_scores.append(0)  # Fallback to 0 if conversion fails

print("Decoding completed. Calculating metrics...")

# Get the true scores from the subset of the test DataFrame
true_scores = test_df['score'].astype(int).tolist()[:10]  # Match the subset size

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(true_scores, predicted_scores)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(true_scores, predicted_scores)
print(f"Mean Absolute Error (MAE): {mae}")

print("Evaluation completed.")
