import numpy as np


class NeuralNetwork:
    """
    A simple Multi-Layer Perceptron (MLP) class that supports
    both classification (softmax + cross-entropy) and regression (linear + MSE).

    Public API:
        - __init__(input_dim, hidden_layer_sizes, output_dim, use_bias=True)
        - train(X_train, Y_train, learning_rate, epochs, max_error,
                act_func, act_deriv, problem_type="classification")
        - predict_classes(X, hidden_act_func)
        - predict_regression(X, hidden_act_func)
    """

    def __init__(self, input_dim, hidden_layer_sizes, output_dim, use_bias=True):
        # Layer sizes: [input] + hidden... + [output]
        self.layer_sizes = [input_dim] + list(hidden_layer_sizes) + [output_dim]
        self.use_bias = use_bias

        self.weights = []
        self.biases = []

        # Xavier / He-like initialization for stability
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            # Scale ~ sqrt(2 / fan_in) works well for ReLU / Tanh in practice
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            self.weights.append(w)

            if self.use_bias:
                b = np.zeros((1, fan_out))
                self.biases.append(b)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward_pass(self, X, act_func, output_activation="softmax"):
        """
        output_activation:
            - "softmax"  -> for classification
            - "linear"   -> for regression
        Returns:
            list of activations [a0, a1, ..., aL]
            where a0 = X and aL = network output.
        """
        activations = [X]
        a = X

        # Hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i])
            if self.use_bias:
                z += self.biases[i]
            a = act_func(z)
            activations.append(a)

        # Output layer
        z_out = np.dot(a, self.weights[-1])
        if self.use_bias:
            z_out += self.biases[-1]

        if output_activation == "softmax":
            # Softmax with numerical stability
            z_shifted = z_out - np.max(z_out, axis=1, keepdims=True)
            exp_z = np.exp(z_shifted)
            a_out = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        elif output_activation == "linear":
            a_out = z_out
        else:
            raise ValueError(f"Unknown output_activation: {output_activation}")

        activations.append(a_out)
        return activations

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------
    def predict_classes(self, X, hidden_act_func):
        """
        Classification prediction: returns class indices (0..C-1).
        """
        activations = self.forward_pass(
            X, hidden_act_func, output_activation="softmax"
        )
        output = activations[-1]
        return np.argmax(output, axis=1)

    def predict_regression(self, X, hidden_act_func):
        """
        Regression prediction: returns continuous outputs.
        """
        activations = self.forward_pass(
            X, hidden_act_func, output_activation="linear"
        )
        output = activations[-1]
        return output

    # ------------------------------------------------------------------
    # Unified training function
    # ------------------------------------------------------------------
    def train(
        self,
        X_train,
        Y_train,
        learning_rate,
        epochs,
        max_error,
        act_func,
        act_deriv,
        problem_type="classification",
    ):
        """
        Unified training function.

        Args:
            X_train (np.array): (N, D) input data.
            Y_train (np.array): (N, C) for classification (one-hot),
                                (N, 1) for regression.
            learning_rate (float)
            epochs (int): maximum number of epochs
            max_error (float): early stopping threshold
            act_func (callable): activation for hidden layers
            act_deriv (callable): derivative of activation (expects ACTIVATION a, not z)
            problem_type (str): "classification" or "regression"

        Returns:
            error_history (list of float): loss at each epoch.
        """
        problem_type = problem_type.lower()
        if problem_type not in ("classification", "regression"):
            raise ValueError(f"Unknown problem_type: {problem_type}")

        if problem_type == "classification":
            output_activation = "softmax"
        else:
            output_activation = "linear"

        N = X_train.shape[0]  # number of samples
        error_history = []

        for epoch in range(epochs):
            # -------- 1. Forward pass --------
            activations = self.forward_pass(
                X_train, act_func, output_activation=output_activation
            )
            output = activations[-1]

            # -------- 2. Compute loss and output delta --------
            if problem_type == "classification":
                # Cross-entropy with softmax
                # Clip to avoid log(0)
                eps = 1e-8
                current_error = -np.mean(
                    np.sum(Y_train * np.log(output + eps), axis=1)
                )
                # For softmax + cross-entropy:
                # gradient wrt z_out is (Y - output)
                delta = Y_train - output
            else:
                # Mean squared error for regression
                diff = Y_train - output
                current_error = np.mean(diff ** 2)
                # For linear output, derivative wrt z_out is diff
                delta = diff

            error_history.append(current_error)

            # -------- 3. Backpropagation --------
            deltas = [delta]  # delta for last layer is at index -1

            # Hidden layers backward: L-1 down to 1
            # layer_sizes: [l0, l1, ..., lL]
            # weights[i]: connects layer i -> i+1
            for i in range(len(self.layer_sizes) - 2, 0, -1):
                # Current delta corresponds to layer i+1
                delta_next = deltas[0]
                w_next = self.weights[i]  # W_i : i -> i+1
                a_i = activations[i]      # activation at layer i
                delta_i = (delta_next @ w_next.T) * act_deriv(a_i)
                deltas.insert(0, delta_i)

            # -------- 4. Gradient updates (with averaging over N) --------
            for i in range(len(self.weights)):
                # activations[i] has shape (N, layer_sizes[i])
                # deltas[i]      has shape (N, layer_sizes[i+1])
                grad_w = activations[i].T @ deltas[i] / N
                self.weights[i] += learning_rate * grad_w

                if self.use_bias:
                    grad_b = np.sum(deltas[i], axis=0, keepdims=True) / N
                    self.biases[i] += learning_rate * grad_b

            # -------- 5. Logging / early stopping --------
            tag = "CLS" if problem_type == "classification" else "REG"
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"[{tag}] Epoch {epoch+1}/{epochs} - Error: {current_error:.4f}")

            if current_error <= max_error:
                print(
                    f"[{tag}] Early stopping at epoch {epoch+1}, "
                    f"error = {current_error:.6f}"
                )
                break

        return error_history
