import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split


class CustomLogisticRegression:
    def __init__(self, lr=0.01, epochs=4000, reg_lambda=0.1):
        self.lr = lr
        self.epochs = epochs
        self.reg_lambda = reg_lambda  # L2 regularization strength

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        self.n_samples, self.n_features = X.shape

        self.weights = np.zeros((self.n_features, 1))
        self.bias = 0

        for epoch in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # Gradients with L2 regularization
            dw = (1 / self.n_samples) * np.dot(X.T, (y_pred - y)) + (self.reg_lambda / self.n_samples) * self.weights
            db = (1 / self.n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return (y_pred >= 0.5).astype(int)


class ManualStandardScaler:
    def fit(self, X):
        X = np.array(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1  # avoid division by zero

    def transform(self, X):
        X = np.array(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def main():
    print("Loading dataset...")
    df = pd.read_csv("features_selected.csv")

    target = "target"
    feature_names = [c for c in df.columns if c != target]

    X = df[feature_names].values
    y = df[target].values

    print(f"Total rows: {len(df)}, Features: {len(feature_names)}")

    # Train/test split (from Model 2 — better practice than training on full data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit scaler on training data only
    scaler = ManualStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = CustomLogisticRegression(lr=0.01, epochs=4000, reg_lambda=0.1)

    print("Training model...")
    model.fit(X_train_scaled, y_train)

    # Quick accuracy check on held-out test set
    y_pred = model.predict(X_test_scaled)
    accuracy = (y_pred.flatten() == y_test).mean()
    print(f"Accuracy on test set: {accuracy:.4f}")

    # Save weights and bias
    np.save("logreg_weights.npy", model.weights)
    np.save("logreg_bias.npy", np.array(model.bias))

    # Save scaler params
    np.save("logreg_mean.npy", scaler.mean_)
    np.save("logreg_std.npy", scaler.std_)

    # Save feature names (from Model 1 — needed by evaluate_model.py)
    with open("feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    print("Training complete. Weights, scaler params and feature names saved.")


if __name__ == "__main__":
    main()