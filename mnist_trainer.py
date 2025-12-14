# mnist_trainer.py

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models

from activation_function import Activation
from mln_network import NeuralNetwork


# ---------------------------------------------------------------------
# 1. Load MNIST from local mnist.npz
# ---------------------------------------------------------------------
def load_mnist_npz(path: str = "mnist.npz"):
    """
    Loads MNIST from a local mnist.npz file.
    Returns:
        X_train, y_train, X_test, y_test
    """
    print(f"Loading MNIST from {path} ...")
    data = np.load(path)
    X_train = data["x_train"]   # (60000, 28, 28)
    y_train = data["y_train"]   # (60000,)
    X_test = data["x_test"]     # (10000, 28, 28)
    y_test = data["y_test"]     # (10000,)
    print("Train:", X_train.shape, " Test:", X_test.shape)
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------
# 2. CNN feature extractor using TensorFlow
# ---------------------------------------------------------------------
def build_and_train_cnn_feature_extractor(
    X_train_raw,
    y_train,
    X_test_raw,
    y_test,
    epochs_cnn: int = 3,
    batch_size: int = 128,
):
    """
    Builds a small CNN in TensorFlow/Keras, trains it on MNIST,
    and returns dense feature vectors for train & test sets.
    Those features are then fed into the custom MLP.
    """

    # Prepare data for CNN: add channel dim and normalize
    X_train_cnn = X_train_raw.astype("float32") / 255.0
    X_test_cnn = X_test_raw.astype("float32") / 255.0
    X_train_cnn = np.expand_dims(X_train_cnn, axis=-1)  # (N, 28, 28, 1)
    X_test_cnn = np.expand_dims(X_test_cnn, axis=-1)

    print("Building CNN feature extractor ...")

    inputs = tf.keras.Input(shape=(28, 28, 1))

    x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    # This dense layer will be our "feature vector"
    x = layers.Dense(128, activation="relu", name="features")(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    cnn_model = models.Model(inputs=inputs, outputs=outputs)
    cnn_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(cnn_model.summary())

    print("Training CNN feature extractor ...")
    cnn_model.fit(
        X_train_cnn,
        y_train,
        epochs=epochs_cnn,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
    )

    # Build model that outputs the feature layer
    feature_model = models.Model(
        inputs=cnn_model.input,
        outputs=cnn_model.get_layer("features").output,
    )

    print("Extracting CNN features (train & test) ...")
    X_train_feat = feature_model.predict(X_train_cnn, batch_size=256, verbose=1)
    X_test_feat = feature_model.predict(X_test_cnn, batch_size=256, verbose=1)

    print("CNN feature shapes:", X_train_feat.shape, X_test_feat.shape)
    return X_train_feat, X_test_feat, y_train, y_test


# ---------------------------------------------------------------------
# 3. Preprocess CNN features for our MLP
# ---------------------------------------------------------------------
def preprocess_features_for_mlp(
    X_train_feat,
    y_train,
    X_test_feat,
    y_test,
    num_classes=10,
):
    """
    Takes CNN feature vectors and prepares them for our custom MLP:
    - Optional standardization
    - One-hot encoding for labels
    """

    # Standardize features (mean 0, std 1) for stability
    mean = X_train_feat.mean(axis=0, keepdims=True)
    std = X_train_feat.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train_feat - mean) / std
    X_test = (X_test_feat - mean) / std

    # One-hot labels
    Y_train = np.zeros((len(y_train), num_classes), dtype=np.float32)
    Y_train[np.arange(len(y_train)), y_train] = 1.0

    print("Preprocessed MLP inputs:", X_train.shape, "Y_train:", Y_train.shape)
    return X_train, Y_train, X_test, y_train, y_test


# ---------------------------------------------------------------------
# 4. Build and train our custom MLP on CNN features
# ---------------------------------------------------------------------
def train_mnist_mlp_on_features(
    X_train,
    Y_train,
    hidden_layer_sizes=(256, 128, 64),
    learning_rate=0.01,
    max_epochs=30,
    max_error=0.001,
    
):
    """
    Trains your NeuralNetwork on CNN feature vectors and returns (nn, error_history).
    """
    input_dim = X_train.shape[1]   # feature dimension from CNN
    output_dim = Y_train.shape[1]  # 10

    # Activation for hidden layers
    act_func, act_deriv = Activation.get_activation("ReLU")

    nn = NeuralNetwork(
        input_dim=input_dim,
        hidden_layer_sizes=list(hidden_layer_sizes),
        output_dim=output_dim,
        use_bias=True,
    )

    print(f"MLP architecture: {input_dim} -> {list(hidden_layer_sizes)} -> {output_dim}")
    print(f"Learning rate={learning_rate}, max_epochs={max_epochs}, max_error={max_error}")

    error_history = nn.train(
        X_train,
        Y_train,
        learning_rate=learning_rate,
        epochs=max_epochs,
        max_error=max_error,
        act_func=act_func,
        act_deriv=act_deriv,
        problem_type="classification",
    )

    print("MLP training finished. Epochs run:", len(error_history))
    return nn, error_history, act_func


# ---------------------------------------------------------------------
# 5. Evaluation helpers
# ---------------------------------------------------------------------
def compute_accuracy(preds, labels):
    return np.mean(preds == labels)


def confusion_matrix(preds, labels, num_classes=10):
    """
    Simple confusion matrix without sklearn.
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1
    return cm


# ---------------------------------------------------------------------
# 6. Plotting functions
# ---------------------------------------------------------------------
def plot_error(error_history, train_acc=None, test_acc=None ):
    plt.figure(figsize=(6, 4))

    epochs_range = np.arange(1, len(error_history) + 1)
    plt.plot(epochs_range, error_history, marker="o")

    final_error = error_history[-1]

    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("MNIST Training Error (MLP on CNN features)")
    plt.grid(True)

    # ✅ Text box with metrics
    metrics_text = f"Final Error: {final_error:.6f}"

    if train_acc is not None:
        metrics_text += f"\nTrain Accuracy: {train_acc * 100:.2f}%"

    if test_acc is not None:
        metrics_text += f"\nTest Accuracy: {test_acc * 100:.2f}%"

    plt.gca().text(
        0.98,
        0.98,
        metrics_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()



def plot_confusion_matrix(cm, title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    num_classes = cm.shape[0]
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()


def plot_sample_predictions(X_test_raw, y_test, preds, num_rows=3, num_cols=4):
    """
    Show a grid of test images with true & predicted labels.
    """
    plt.figure(figsize=(num_cols * 2, num_rows * 2))
    total = num_rows * num_cols
    indices = np.random.choice(len(X_test_raw), size=total, replace=False)

    for i, idx in enumerate(indices, start=1):
        img = X_test_raw[idx]
        true_label = y_test[idx]
        pred_label = preds[idx]

        plt.subplot(num_rows, num_cols, i)
        plt.imshow(img, cmap="gray")
        plt.title(f"T:{true_label} P:{pred_label}", fontsize=10)
        plt.axis("off")

    plt.suptitle("Sample MNIST Predictions (MLP on CNN features)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])


# ---------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------
def main():
    # 1. Load raw MNIST images
    X_train_raw, y_train, X_test_raw, y_test = load_mnist_npz("mnist.npz")

    # 2. Train CNN and extract feature vectors
    X_train_feat, X_test_feat, y_train_int, y_test_int = \
        build_and_train_cnn_feature_extractor(
            X_train_raw,
            y_train,
            X_test_raw,
            y_test,
            epochs_cnn=3,
            batch_size=128,
        )

    # 3. Preprocess features for our MLP
    X_train, Y_train, X_test, y_train_int, y_test_int = \
        preprocess_features_for_mlp(
            X_train_feat,
            y_train_int,
            X_test_feat,
            y_test_int,
            num_classes=10,
        )

    # 4. Train our custom NeuralNetwork on the CNN features
    nn, error_history, act_func = train_mnist_mlp_on_features(
        X_train,
        Y_train,
        hidden_layer_sizes=[256, 128, 64],
        learning_rate=0.01,
        max_epochs=50,
        max_error=0.001,
    )

    # 5. Evaluate
    train_preds = nn.predict_classes(X_train, act_func)
    test_preds = nn.predict_classes(X_test, act_func)

    train_acc = compute_accuracy(train_preds, y_train_int)
    test_acc = compute_accuracy(test_preds, y_test_int)

    print(f"Train accuracy (MLP on CNN features): {train_acc * 100:.2f}%")
    print(f"Test  accuracy (MLP on CNN features): {test_acc * 100:.2f}%")

    # 6. Plots
    plot_error(
        error_history,
        train_acc=train_acc,
        test_acc=test_acc,
    )


    cm = confusion_matrix(test_preds, y_test_int, num_classes=10)
    plot_confusion_matrix(cm, title="MNIST Confusion Matrix (Test)")

    plot_sample_predictions(X_test_raw, y_test_int, test_preds)

    # Show all figures
    plt.show()


if __name__ == "__main__":
    main()
