"""
utils/model_utils.py

A from-scratch, NumPy-only feed-forward neural network for the
3-class freshness classification task (Fresh / Consume Soon / Spoiled).

No ML framework is used for the core algorithm: forward pass, softmax
cross-entropy loss, and backpropagation are all implemented explicitly
so the math stays transparent end to end.

Architecture: Input -> Dense(ReLU) -> Dense(ReLU) -> Dense(Softmax)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(z.dtype)


def softmax(z: np.ndarray) -> np.ndarray:
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


def cross_entropy_loss(probs: np.ndarray, y_onehot: np.ndarray) -> float:
    eps = 1e-12
    n = probs.shape[0]
    return -np.sum(y_onehot * np.log(probs + eps)) / n


@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)


class FreshnessNeuralNetwork:
    """A hand-rolled 2-hidden-layer MLP trained with mini-batch gradient descent."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 32,
        hidden_dim_2: int = 16,
        num_classes: int = 3,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.num_classes = num_classes

        rng = np.random.default_rng(seed)

        # He initialization, appropriate for ReLU hidden layers
        self.W1 = rng.normal(0, np.sqrt(2 / input_dim), size=(input_dim, hidden_dim_1))
        self.b1 = np.zeros((1, hidden_dim_1))

        self.W2 = rng.normal(0, np.sqrt(2 / hidden_dim_1), size=(hidden_dim_1, hidden_dim_2))
        self.b2 = np.zeros((1, hidden_dim_2))

        self.W3 = rng.normal(0, np.sqrt(2 / hidden_dim_2), size=(hidden_dim_2, num_classes))
        self.b3 = np.zeros((1, num_classes))

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> dict:
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)

        z3 = a2 @ self.W3 + self.b3
        probs = softmax(z3)

        return {"z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3, "probs": probs}

    def backward(self, X: np.ndarray, y_onehot: np.ndarray, cache: dict, l2_lambda: float) -> dict:
        n = X.shape[0]

        dz3 = (cache["probs"] - y_onehot) / n
        dW3 = cache["a2"].T @ dz3 + l2_lambda * self.W3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * relu_derivative(cache["z2"])
        dW2 = cache["a1"].T @ dz2 + l2_lambda * self.W2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_derivative(cache["z1"])
        dW1 = X.T @ dz1 + l2_lambda * self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}

    def apply_gradients(self, grads: dict, learning_rate: float) -> None:
        self.W1 -= learning_rate * grads["dW1"]
        self.b1 -= learning_rate * grads["db1"]
        self.W2 -= learning_rate * grads["dW2"]
        self.b2 -= learning_rate * grads["db2"]
        self.W3 -= learning_rate * grads["dW3"]
        self.b3 -= learning_rate * grads["db3"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)["probs"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 60,
        batch_size: int = 64,
        learning_rate: float = 0.05,
        l2_lambda: float = 1e-4,
        seed: int = 42,
        progress_callback=None,
    ) -> TrainingHistory:
        """
        Trains with mini-batch gradient descent.

        progress_callback(epoch, epochs, history): optional hook called
        after every epoch, used by the Streamlit training page to update
        a progress bar / live chart without this module depending on
        Streamlit at all.
        """
        history = TrainingHistory()
        rng = np.random.default_rng(seed)
        y_train_onehot = one_hot(y_train, self.num_classes)
        n = X_train.shape[0]

        for epoch in range(1, epochs + 1):
            perm = rng.permutation(n)
            X_shuffled = X_train[perm]
            y_shuffled = y_train_onehot[perm]

            for start in range(0, n, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                cache = self.forward(X_batch)
                grads = self.backward(X_batch, y_batch, cache, l2_lambda)
                self.apply_gradients(grads, learning_rate)

            train_cache = self.forward(X_train)
            train_loss = cross_entropy_loss(train_cache["probs"], y_train_onehot)
            train_acc = self.accuracy(X_train, y_train)
            history.train_loss.append(train_loss)
            history.train_accuracy.append(train_acc)

            if X_val is not None and y_val is not None:
                val_cache = self.forward(X_val)
                val_loss = cross_entropy_loss(val_cache["probs"], one_hot(y_val, self.num_classes))
                val_acc = self.accuracy(X_val, y_val)
                history.val_loss.append(val_loss)
                history.val_accuracy.append(val_acc)

            if progress_callback is not None:
                progress_callback(epoch, epochs, history)

        return history

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        np.savez(
            path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W3=self.W3, b3=self.b3,
            input_dim=self.input_dim,
            hidden_dim_1=self.hidden_dim_1,
            hidden_dim_2=self.hidden_dim_2,
            num_classes=self.num_classes,
        )

    @classmethod
    def load(cls, path: str) -> "FreshnessNeuralNetwork":
        data = np.load(path)
        model = cls(
            input_dim=int(data["input_dim"]),
            hidden_dim_1=int(data["hidden_dim_1"]),
            hidden_dim_2=int(data["hidden_dim_2"]),
            num_classes=int(data["num_classes"]),
        )
        model.W1, model.b1 = data["W1"], data["b1"]
        model.W2, model.b2 = data["W2"], data["b2"]
        model.W3, model.b3 = data["W3"], data["b3"]
        return model
