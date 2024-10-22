from train_gpt import trainer, test_dataset, tokenizer, test_df
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Use the trainer's predict function to get predictions
predictions = trainer.predict(test_dataset)

# Extract predicted scores (assuming GPT generates the score at the end of the sequence)
predicted_scores = [tokenizer.decode(pred).split('[SEP]')[-1].strip() for pred in predictions.predictions]

# Convert predicted scores to integers
predicted_scores = [int(score) for score in predicted_scores]

# Get the true scores
true_scores = test_df['score'].astype(int).tolist()

# Calculate mean squared error (or other metrics)
mse = mean_squared_error(true_scores, predicted_scores)
print(f"Mean Squared Error: {mse}")

# Optionally, calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(true_scores, predicted_scores)
print(f"Mean Absolute Error: {mae}")
