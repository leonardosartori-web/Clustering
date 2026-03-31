import numpy as np
import os
from sklearn.datasets import fetch_openml


def load_mnist_full(
        data_dir="data",
        dtype=np.float32
):
    os.makedirs(data_dir, exist_ok=True)

    X_path = os.path.join(data_dir, "mnist_X_full.npy")
    y_path = os.path.join(data_dir, "mnist_y_full.npy")

    if os.path.exists(X_path) and os.path.exists(y_path):
        print("📂 Carico MNIST completo da file...")
        X = np.load(X_path)
        y = np.load(y_path)
    else:
        print("⬇️  Scarico MNIST completo da OpenML...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)

        X = mnist.data.astype(dtype)
        y = mnist.target.astype(int)

        np.save(X_path, X)
        np.save(y_path, y)

    X /= 255 # normalization

    return X, y


def get_subset(n_samples, X, y):
    np.random.seed(42)
    idx = np.random.choice(len(X), n_samples, replace=False)
    X_sub = X[idx]
    y_sub = y[idx]
    return X_sub, y_sub