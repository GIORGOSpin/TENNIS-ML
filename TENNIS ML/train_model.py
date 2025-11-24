import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class CustomLogisticRegression:
    def __init__(self, lr=0.01, epochs=3000, reg_lambda=0.1):
        self.lr = lr
        self.epochs = epochs
        self.reg_lambda = reg_lambda  # L2 regularization strength

    def sigmoid(self, z):
        # Προστασία από overflow στο exp με clip
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        self.n_samples, self.n_features = X.shape

        # αρχικοποίηση weights και bias
        self.weights = np.zeros((self.n_features, 1))
        self.bias = 0

        for epoch in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # Υπολογισμός gradients με regularization
            dw = (1 / self.n_samples) * np.dot(X.T, (y_pred - y)) + (self.reg_lambda / self.n_samples) * self.weights
            db = (1 / self.n_samples) * np.sum(y_pred - y)

            # Ενημέρωση παραμέτρων
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return (y_pred >= 0.5).astype(int)

class ManualStandardScaler:
    def fit(self, X):
        X = np.array(X, dtype=float)   # <--- FIX: ensure numpy float array
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        # αποφυγή διαίρεσης με μηδέν
        self.std_[self.std_ == 0] = 1

    def transform(self, X):
        X = np.array(X, dtype=float)   # <--- FIX: ensure numpy float array
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def main():
    # Φόρτωση δεδομένων με επιλεγμένα χαρακτηριστικά
    df = pd.read_csv("features_selected.csv")

    target = "target"
    X = df.drop(columns=[target]).values
    y = df[target].values

    # Διαχωρισμός σε train-test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Δημιουργία και εφαρμογή χειροκίνητου scaler
    scaler = ManualStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Δημιουργία μοντέλου
    model = CustomLogisticRegression(
        lr=0.01,       # learning rate
        epochs=4000,   # αριθμός επαναλήψεων
        reg_lambda=0.1 # L2 regularization
    )

    print("Training model...")
    model.fit(X_train_scaled, y_train)

    # Κάνουμε scale και στο test set με το ίδιο scaler
    X_test_scaled = scaler.transform(X_test)

    print("Evaluating...")
    y_pred = model.predict(X_test_scaled)

    accuracy = (y_pred.flatten() == y_test).mean()
    print(f"Accuracy στο test set: {accuracy:.4f}")

    # Αποθήκευση βαρών, bias, mean και std για να τα ξαναφορτώσουμε στο evaluate
    np.save("logreg_weights.npy", model.weights)
    np.save("logreg_bias.npy", model.bias)
    np.save("logreg_mean.npy", scaler.mean_)
    np.save("logreg_std.npy", scaler.std_)
    print("Αποθηκεύτηκαν weights, bias, mean & std.")

if __name__ == "__main__":
    main()
