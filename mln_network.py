import numpy as np

class NeuralNetwork:
    """
    A simple Multi-Layer Perceptron (MLP) class.
    """
    
    def __init__(self, input_dim, hidden_layer_sizes, output_dim, use_bias=True):
        """
        Initializes the network's architecture and parameters.

        Args:
            input_dim (int): Number of input features (e.g., 2 for X, Y).
            hidden_layer_sizes (list): A list of integers, where each integer is the
                                       number of neurons in a hidden layer.
                                       Example: [10, 5] means 2 hidden layers
                                       with 10 and 5 neurons.
            output_dim (int): Number of output neurons (e.g., number of classes).
            use_bias (bool): Whether to include bias terms.
        """
        self.layer_sizes = [input_dim] + hidden_layer_sizes + [output_dim]
        self.use_bias = use_bias
        
        # --- These are the `Weight` and `Bias` parameters ---
        # They are initialized here, to be updated during training.
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases for each layer
        # (from input_layer -> hidden_layer_1, hidden_1 -> hidden_2, ..., hidden_N -> output)
        for i in range(len(self.layer_sizes) - 1):
            # Weights are initialized randomly
            # Shape is (neurons_in_previous_layer, neurons_in_current_layer)
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.1
            self.weights.append(w)
            
            if self.use_bias:
                # Biases are initialized to zeros
                # Shape is (1, neurons_in_current_layer)
                b = np.zeros((1, self.layer_sizes[i+1]))
                self.biases.append(b)
                
    def forward_pass(self, X, act_func):
        activations = [X]
        a = X

        # Hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i])
            if self.use_bias:
                z += self.biases[i]
            a = act_func(z)
            activations.append(a)

        # --- Output Layer (Softmax for multi-class) ---
        output_z = np.dot(a, self.weights[-1])
        if self.use_bias:
            output_z += self.biases[-1]

        exp_z = np.exp(output_z - np.max(output_z, axis=1, keepdims=True))
        output_a = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        activations.append(output_a)
        return activations


    
    def predict(self, X, hidden_act_func):
        """
        Predicts the class labels for a given input X.
        
        Args:
            X (np.array): Input data.
            hidden_act_func (function): The activation function for hidden layers.
            
        Returns:
            np.array: A 1D array of predicted class indices (e.g., 0, 1, 2).
        """
        # Use the forward_pass to get layer activations
        activations = self.forward_pass(X, hidden_act_func)
        
        # Get the final output layer activations
        output_layer_activations = activations[-1]
        
        # The predicted class is the one with the highest activation/score
        predictions = np.argmax(output_layer_activations, axis=1)
        return predictions

    def train(self, X_train, Y_train, learning_rate, epochs, act_func, act_deriv):
        """
        Trains the network using backpropagation with Softmax + Cross-Entropy loss.
        """
        error_history = []

        for epoch in range(epochs):
            # --- 1. Forward Pass ---
            activations = self.forward_pass(X_train, act_func)
            output = activations[-1]

            # --- 2. Compute Error ---
            output_error = Y_train - output

            # --- 3. Output Layer Delta (Softmax + CrossEntropy) ---
            delta = output_error
            deltas = [delta]

            # --- 4. Backpropagate through hidden layers ---
            for i in range(len(self.layer_sizes) - 2, 0, -1):
                delta = (deltas[0] @ self.weights[i].T) * act_deriv(activations[i])
                deltas.insert(0, delta)

            # --- 5. Gradient Descent Updates ---
            for i in range(len(self.weights)):
                self.weights[i] += learning_rate * (activations[i].T @ deltas[i])
                if self.use_bias:
                    self.biases[i] += learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

            # --- 6. Cross-Entropy Loss for Monitoring ---
            current_error = -np.mean(np.sum(Y_train * np.log(output + 1e-8), axis=1))
            error_history.append(current_error)

        return error_history

