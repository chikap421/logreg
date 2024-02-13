import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def fit(X, y, phi, lmbd):
    """
    Fits a logistic regression model on the data (X, y), using the specified feature mask (phi) and regularization strength (lmbd).
    
    Parameters:
    - X: 2-D numpy array of feature data.
    - y: 1-D numpy array of target labels.
    - phi: Boolean array indicating selected features.
    - lmbd: Regularization strength; 0 for no regularization.
    
    Returns:
    - Fitted LogisticRegression model.
    """
    if lmbd == 0:
        model = LogisticRegression(penalty='none')
    else:
        model = LogisticRegression(penalty='l2', C=1 / lmbd)
    
    model.fit(X[:, phi], y)
    return model

def predict(X, phi, model):
    """
    Predicts target values using the provided logistic regression model for the selected features.
    
    Parameters:
    - X: 2-D numpy array of feature data.
    - phi: Boolean array indicating selected features.
    - model: Fitted LogisticRegression model.
    
    Returns:
    - Predictions as a 1-D numpy array.
    """
    return model.predict(X[:, phi])

def logloss(y_true, y_pred):
    """
    Calculates the logistic loss between true labels and predictions.
    
    Parameters:
    - y_true: 1-D numpy array of true labels.
    - y_pred: 1-D numpy array of predicted labels.
    
    Returns:
    - Logistic loss as a float.
    """
    return log_loss(y_true, y_pred)

def sweep_hyperparameters(X_train, y_train, X_val, y_val, lmbds, phis):
    """
    Finds the best combination of regularization strength and feature selection based on validation set performance.
    
    Parameters:
    - X_train, y_train: Training data and labels.
    - X_val, y_val: Validation data and labels.
    - lmbds: List of regularization strengths to test.
    - phis: List of boolean arrays for feature selection.
    
    Returns:
    - Tuple of best lambda, feature mask, and fitted model.
    """
    best_logloss = float('inf')
    best_lmbd, best_phi, best_model = None, None, None

    for lmbd in lmbds:
        for phi in phis:
            model = fit(X_train, y_train, phi, lmbd)
            y_val_pred = predict(X_val, phi, model)
            current_logloss = logloss(y_val, y_val_pred)
            if current_logloss < best_logloss:
                best_logloss = current_logloss
                best_lmbd, best_phi, best_model = lmbd, phi, model

    return best_lmbd, best_phi, best_model

def evaluate_model(model, phi, X_test, y_test):
    """
    Evaluates a logistic regression model on a test dataset.
    
    Parameters:
    - model: Fitted LogisticRegression model.
    - phi: Boolean array indicating selected features.
    - X_test, y_test: Test data and labels.
    
    Returns:
    - Test loss as a float.
    """
    y_test_pred = predict(X_test, phi, model)
    return logloss(y_test, y_test_pred)

def train_and_eval_model(X, y, lmbds, phis):
    """
    Trains and evaluates a logistic regression model using a training, validation, and test split.
    
    Parameters:
    - X, y: Dataset features and labels.
    - lmbds: List of regularization strengths to test.
    - phis: List of boolean arrays for feature selection.
    
    Returns:
    - Tuple of best model, feature mask, lambda, and test loss.
    """
    n_samples = len(X)
    n_train, n_val = int(n_samples * 0.5), int(n_samples * 0.25)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    best_lmbd, best_phi, best_model = sweep_hyperparameters(X_train, y_train, X_val, y_val, lmbds, phis)
    test_loss = evaluate_model(best_model, best_phi, X_test, y_test)

    print(f"Best lambda: {best_lmbd}\nBest feature mask: {best_phi}\nTest loss: {test_loss}")

    return best_model, best_phi, best_lmbd, test_loss
