import numpy as np

class Activation:
    """
    A collection of standard activation functions and their derivatives
    for use in neural networks.
    """

    @staticmethod
    def get_activation(name):
        """
        Helper to retrieve function pairs by string name (matching the UI dropdown).
        Returns: (function, derivative_function)
        """
        if name == "Sigmoid":
            return Activation.sigmoid, Activation.sigmoid_derivative
        elif name == "ReLU":
            return Activation.relu, Activation.relu_derivative
        elif name == "Tanh":
            return Activation.tanh, Activation.tanh_derivative
        elif name == "Linear":
            return Activation.linear, Activation.linear_derivative
        else:
            raise ValueError(f"Unknown activation function: {name}")

    # --- Sigmoid ---
    # Used often for binary classification output (0 to 1 range).
    @staticmethod
    def sigmoid(x):
        # Clip x to avoid overflow/underflow in exp
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(a):
        # FIX: 'a' is the activation (the output of sigmoid), not the input.
        # The derivative of sigmoid(z) is a*(1-a)
        return a * (1 - a)

    # --- ReLU (Rectified Linear Unit) ---
    # Very popular for hidden layers.
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(a):
        # FIX: 'a' is the activation.
        # The derivative is 1 if a > 0, else 0.
        return np.where(a > 0, 1, 0)

    # --- Tanh (Hyperbolic Tangent) ---
    # Similar to sigmoid but ranges from -1 to 1.
    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(a):
        # FIX: 'a' is the activation (the output of tanh), not the input.
        # The derivative of tanh(z) is 1 - a^2
        return 1 - a**2

    # --- Linear (Identity) ---
    # Used for regression output layers.
    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        # Derivative of x is always 1
        return np.ones_like(x)