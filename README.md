# Logistic Regression Model Trainer

This Python script is designed to fit, predict, and evaluate logistic regression models with the flexibility of feature selection and regularization. It is tailored for educational and experimental purposes, allowing users to understand the impact of hyperparameter tuning on model performance.

## Features

- Fits logistic regression models using SciKit Learn's `LogisticRegression`.
- Supports hyperparameter tuning for regularization strength and feature selection.
- Includes functionality to evaluate models using logistic loss.
- Example code for generating synthetic data and performing a hyperparameter sweep.

## Dependencies

- NumPy
- SciKit Learn

To run this code, make sure you have the above Python packages installed. You can install these dependencies via pip:

`pip install scikit-learn numpy`


## Usage

1. **Prepare your data**: The script expects input data in the form of NumPy arrays. You'll need to have your data split into features (`X`) and targets (`y`).

2. **Hyperparameter tuning**: Use the `sweep_hyperparameters` function to find the best combination of regularization strength (`lmbd`) and feature selection mask (`phi`).

3. **Model Evaluation**: After identifying the best parameters, you can evaluate the model's performance on a test set using the `evaluate_model` function.

4. **Example Data**: The script includes a section to generate example data and run the model fitting and evaluation process.

### Functions

- `fit(X, y, phi, lmbd)`: Fits a logistic regression model to the data.
- `predict(X, phi, theta)`: Predicts target values using the fitted model.
- `logloss(y, y_hat)`: Calculates the logistic loss between the actual and predicted values.
- `sweep_hyperparameters(...)`: Finds the best hyperparameters for the model.
- `evaluate_model(...)`: Evaluates the model on a test dataset.
- `train_and_eval_model(...)`: A comprehensive function that trains the model and evaluates its performance on a split dataset.

## Example

Here's a quick example of how to use this script with synthetic data:

```python
# Generate synthetic data
N, d = 100, 5
X = np.random.normal(size=(N, d))
y = np.random.choice([0, 1], size=N)

# Define hyperparameters
lmbds = [0, 0.1, 1, 10]
phis = [np.random.choice(a=[True, False], size=d) for _ in range(3)]
for phi in phis:
    phi[0] = True  # Ensure at least one feature is selected

# Train and evaluate the model
best_theta, best_phi, best_lmbd, test_loss = train_and_eval_model(X, y, lmbds, phis)

print(f"Best lambda: {best_lmbd}, Best feature mask: {best_phi}, Test loss: {test_loss}")
```

## Contribution
Feel free to fork this project and contribute. If you find any bugs or have suggestions, please open an issue or submit a pull request.

