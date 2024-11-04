from train_gpt import trainer, test_dataset, tokenizer, test_df
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Use the trainer's predict function to get predictions on the test dataset
predictions = trainer.predict(test_dataset)

# Extract predicted scores (assuming GPT generates the score at the end of the sequence)
predicted_scores = []
for pred in predictions.predictions:
    # Decode the predicted tokens to text, then split to extract the score part
    decoded_text = tokenizer.decode(pred, skip_special_tokens=True)
    predicted_score = decoded_text.split('[SEP]')[-1].strip()  # Get the text after [SEP]
    try:
        predicted_scores.append(int(predicted_score))  # Convert to integer
    except ValueError:
        predicted_scores.append(0)  # Fallback to 0 if conversion fails

# Get the true scores from the test DataFrame
true_scores = test_df['score'].astype(int).tolist()

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(true_scores, predicted_scores)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(true_scores, predicted_scores)
print(f"Mean Absolute Error (MAE): {mae}")

