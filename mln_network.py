import numpy as np

class NeuralNetwork:
    """
    A simple Multi-Layer Perceptron (MLP) class that supports
    both classification (softmax + cross-entropy) and regression (linear + MSE).
    """

    def __init__(self, input_dim, hidden_layer_sizes, output_dim, use_bias=True):
        self.layer_sizes = [input_dim] + hidden_layer_sizes + [output_dim]
        self.use_bias = use_bias

        self.weights = []
        self.biases = []

        for i in range(len(self.layer_sizes) - 1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * 0.1
            self.weights.append(w)

            if self.use_bias:
                b = np.zeros((1, self.layer_sizes[i + 1]))
                self.biases.append(b)

    # --------------- Forward pass ---------------
    def forward_pass(self, X, act_func, output_activation="softmax"):
        """
        output_activation: "softmax" (classification) or "linear" (regression)
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
            exp_z = np.exp(z_out - np.max(z_out, axis=1, keepdims=True))
            a_out = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        elif output_activation == "linear":
            a_out = z_out
        else:
            raise ValueError(f"Unknown output_activation: {output_activation}")

        activations.append(a_out)
        return activations

    # --------------- Prediction helpers ---------------
    def predict_classes(self, X, hidden_act_func):
        """
        Classification prediction: returns class indices.
        """
        activations = self.forward_pass(X, hidden_act_func, output_activation="softmax")
        output = activations[-1]
        return np.argmax(output, axis=1)

    def predict_regression(self, X, hidden_act_func):
        """
        Regression prediction: returns continuous outputs.
        """
        activations = self.forward_pass(X, hidden_act_func, output_activation="linear")
        output = activations[-1]
        return output

    # --------------- Single train function ---------------
    def train(self, X_train, Y_train,
              learning_rate, epochs, max_error,
              act_func, act_deriv,
              problem_type="classification"):
        """
        Unified training function.

        problem_type:
            - "classification" → softmax + cross-entropy
            - "regression"     → linear + MSE
        """
        problem_type = problem_type.lower()
        if problem_type not in ("classification", "regression"):
            raise ValueError(f"Unknown problem_type: {problem_type}")

        error_history = []

        if problem_type == "classification":
            output_activation = "softmax"
        else:
            output_activation = "linear"

        for epoch in range(epochs):
            # Forward pass
            activations = self.forward_pass(X_train, act_func,
                                            output_activation=output_activation)
            output = activations[-1]

            if problem_type == "classification":
                # Error & delta for cross-entropy + softmax
                output_error = Y_train - output
                delta = output_error
            else:
                # Regression: MSE
                diff = Y_train - output
                mse = np.mean(diff ** 2)
                error_history.append(mse)

                # delta for linear output
                delta = diff

            # Build deltas list
            deltas = [delta]

            # Backprop through hidden layers
            for i in range(len(self.layer_sizes) - 2, 0, -1):
                delta = (deltas[0] @ self.weights[i].T) * act_deriv(activations[i])
                deltas.insert(0, delta)

            # Gradient updates
            for i in range(len(self.weights)):
                grad_w = activations[i].T @ deltas[i]
                self.weights[i] += learning_rate * grad_w
                if self.use_bias:
                    grad_b = np.sum(deltas[i], axis=0, keepdims=True)
                    self.biases[i] += learning_rate * grad_b

            # Compute error metric for logging / early stopping
            if problem_type == "classification":
                current_error = -np.mean(np.sum(Y_train * np.log(output + 1e-8), axis=1))
                error_history.append(current_error)
            else:
                current_error = mse  # already computed

            # Some debug every 20 epochs
            if (epoch + 1) % 20 == 0 or epoch == 0:
                tag = "CLS" if problem_type == "classification" else "REG"
                print(f"[{tag}] Epoch {epoch+1}/{epochs} - Error: {current_error:.4f}")

            # Early stopping
            if current_error <= max_error:
                tag = "CLS" if problem_type == "classification" else "REG"
                print(f"[{tag}] Early stopping at epoch {epoch+1}, error={current_error:.6f}")
                break

        return error_history
