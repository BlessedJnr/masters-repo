import sys
import random
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QLabel
)
from PyQt6.QtCore import Qt

# Matplotlib imports
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# Import our other Python files
try:
    from activation_function import Activation
    from mln_network import NeuralNetwork
except ImportError:
    print("Error: Make sure 'activation_function.py' and 'mln_network.py' are in the same folder.")
    sys.exit(1)


# Global data storage
data_points = []
# Pre-define colors for the classes
class_colors = ['#FF0000', '#0000FF', '#00FF00', '#FFD700', '#FF00FF', '#00FFFF', '#FFA500', '#800080']


class NeuralNetworkUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network UI Builder")
        self.setGeometry(100, 100, 1200, 800)
        
        # To store the trained network
        self.trained_network = None

        # --- Main Layout ---
        # A central widget to hold everything
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        # A horizontal layout: [Input Plot] | [Controls & Error Plot]
        main_layout = QHBoxLayout(main_widget)

        # --- 1. Input Plot (Left Side) ---
        self.input_fig = Figure(figsize=(5, 5))
        self.input_canvas = FigureCanvas(self.input_fig)
        self.input_ax = self.input_fig.add_subplot(111)
        self.setup_input_plot()
        main_layout.addWidget(self.input_canvas)

        # --- 2. Right Panel (Controls + Error Plot) ---
        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)
        main_layout.addWidget(right_panel_widget)

        # --- 2a. Parameter Panel (Top Right) ---
        param_widget = QWidget()
        param_layout = QFormLayout(param_widget)
        param_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        
        # Problem type dropdown
        self.problem_type_combo = QComboBox()
        self.problem_type_combo.addItems(["Classification", "Regression"])
        param_layout.addRow("Problem Type:", self.problem_type_combo)

        # Number of classes spinbox
        self.num_classes_spin = QSpinBox()
        self.num_classes_spin.setRange(2, 8) # Max 8 classes (matches colors list)
        self.num_classes_spin.setValue(2)
        param_layout.addRow("Number of Classes:", self.num_classes_spin)

        # Class dropdown (will be populated dynamically)
        self.class_combo = QComboBox()
        param_layout.addRow("Current Class:", self.class_combo)

        # Learning rate
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(3)
        self.lr_spin.setRange(0.001, 1.0)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setSingleStep(0.001)
        param_layout.addRow("Learning Rate:", self.lr_spin)
        
        # Activation function dropdown
        self.activation_combo = QComboBox()
        self.activation_combo.addItems(["Sigmoid", "ReLU", "Tanh", "Linear"])
        param_layout.addRow("Activation Function:", self.activation_combo)

        # Epoch
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(10, 10000)
        self.epoch_spin.setValue(100)
        param_layout.addRow("Epochs:", self.epoch_spin)
        
        # Bias 
        self.bias_check = QCheckBox()
        self.bias_check.setChecked(True)
        param_layout.addRow("Use Bias:", self.bias_check)
        
        right_panel_layout.addWidget(param_widget)

        # --- Buttons ---
        self.train_button = QPushButton("Start Training")
        self.clear_button = QPushButton("Clear Input Points")
        right_panel_layout.addWidget(self.train_button)
        right_panel_layout.addWidget(self.clear_button)

        # --- 2b. Error Plot (Bottom Right) ---
        self.error_fig = Figure(figsize=(5, 4))
        self.error_canvas = FigureCanvas(self.error_fig)
        self.error_ax = self.error_fig.add_subplot(111)
        self.setup_error_plot()
        right_panel_layout.addWidget(self.error_canvas)

        # --- 3. Connecting Signals (Interactivity) ---
        
        # Connect "Number of classes" box to the "Class" dropdown
        self.num_classes_spin.valueChanged.connect(self.update_class_dropdown)
        
        # Connect mouse clicks on the input plot to our function
        self.input_canvas.mpl_connect('button_press_event', self.on_plot_click)
        
        # --- Connected training function ---
        self.train_button.clicked.connect(self.start_training)
        self.clear_button.clicked.connect(self.clear_input_plot)

        # --- Final Setup ---
        # Trigger the function once at the start to populate the class dropdown
        self.update_class_dropdown(self.num_classes_spin.value())

    def setup_input_plot(self):
        self.input_ax.clear()
        self.input_ax.set_title("Input Data")
        self.input_ax.set_xlabel("X-coordinate")
        self.input_ax.set_ylabel("Y-coordinate")
        self.input_ax.grid(True, linestyle='--', alpha=0.6) # Lighter grid
        # Add prominent axis lines
        self.input_ax.axhline(y=0, color='black', linewidth=1.2)
        self.input_ax.axvline(x=0, color='black', linewidth=1.2)
        # Set fixed axes for consistent plotting
        self.input_ax.set_xlim(-10, 10)
        self.input_ax.set_ylim(-10, 10)
        self.input_canvas.draw()

    def setup_error_plot(self):
        self.error_ax.clear()
        self.error_ax.set_title("Training Error")
        self.error_ax.set_xlabel("Epoch")
        self.error_ax.set_ylabel("Error")
        self.error_ax.grid(True)
        self.error_canvas.draw()

    def update_class_dropdown(self, num_classes):
        """
        Clears and repopulates the 'Class' dropdown based on the
        number selected in the 'Number of classes' spin box.
        """
        self.class_combo.clear()
        class_names = [f"Class {i+1}" for i in range(num_classes)]
        self.class_combo.addItems(class_names)

    def on_plot_click(self, event):
        """
        Called when the user clicks on the input plot.
        It gets the coordinates and the currently selected class
        and plots a new point with the correct color.
        """
        # Only register clicks inside the plot area
        if event.inaxes != self.input_ax:
            return

        x, y = event.xdata, event.ydata
        
        # Get the current class index (0 for "Class 1", 1 for "Class 2", etc.)
        class_index = self.class_combo.currentIndex()
        
        # Store the data
        data_points.append((x, y, class_index))
        
        # Get the color for this class
        color = class_colors[class_index]
        
        # Plot the new point with a black edge
        self.input_ax.plot(x, y, 'o', color=color, markersize=8, markeredgecolor='k')
        self.input_canvas.draw()

    def clear_input_plot(self):
        """Clears all data, resets plots, and clears the trained network."""
        global data_points
        data_points = []
        self.trained_network = None # Clear the trained model
        self.setup_input_plot()     # Reset input plot
        self.setup_error_plot()     # Reset error plot

    # --- Training Function ---
    def start_training(self):
        global data_points

        learning_rate = self.lr_spin.value()
        num_epochs = self.epoch_spin.value()
        num_classes = self.num_classes_spin.value()
        use_bias = self.bias_check.isChecked()

        act_name = self.activation_combo.currentText()
        act_func, act_deriv = Activation.get_activation(act_name)

        if not data_points:
            print("No data points to train on. Click on the grid to add data.")
            return

        # --- Prepare and normalize data ---
        X_train = np.array([[p[0], p[1]] for p in data_points])
        Y_train_indices = np.array([p[2] for p in data_points])
        Y_train = np.zeros((len(Y_train_indices), num_classes))
        Y_train[np.arange(len(Y_train_indices)), Y_train_indices] = 1

        # Normalize inputs for better training stability
        self.X_mean = X_train.mean(axis=0)
        self.X_std = X_train.std(axis=0) + 1e-8
        X_train = (X_train - self.X_mean) / self.X_std

        input_dim = 2
        output_dim = num_classes
        hidden_layer_sizes = [4]  # Simpler single hidden layer

        print(f"Starting training with architecture: {input_dim} -> {hidden_layer_sizes} -> {output_dim}")

        self.trained_network = NeuralNetwork(
            input_dim=input_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            output_dim=output_dim,
            use_bias=use_bias
        )

        error_history = self.trained_network.train(
            X_train, Y_train,
            learning_rate, num_epochs,
            act_func, act_deriv
        )

        # Plot error
        epochs = np.arange(1, num_epochs + 1)
        self.error_ax.clear()
        self.error_ax.set_title("Training Error")
        self.error_ax.set_xlabel("Epoch")
        self.error_ax.set_ylabel("Error")
        self.error_ax.grid(True)
        self.error_ax.plot(epochs, error_history, color='b')
        self.error_canvas.draw()
        print("Training complete.")

        # Plot decision boundary
        self.plot_decision_boundary(self.trained_network, act_func)

        
    # ---  Plot Discriminant Function ---
    def plot_decision_boundary(self, nn, hidden_act_func):
        """Plots the decision boundary (discriminant function) on the input plot."""
        global data_points

        self.setup_input_plot()

        x_min, x_max = self.input_ax.get_xlim()
        y_min, y_max = self.input_ax.get_ylim()

        # Create grid
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Apply normalization (same as training)
        grid_points = (grid_points - self.X_mean) / self.X_std

        # Predict class for each point
        Z = nn.predict(grid_points, hidden_act_func)
        Z = Z.reshape(xx.shape)

        num_classes = self.num_classes_spin.value()

        # --- Use discrete colormap and explicit class boundaries ---
        cmap_light = ListedColormap(class_colors[:num_classes])

        # Define boundaries between classes explicitly
        levels = np.arange(-0.5, num_classes + 0.5, 1)

        # Fill background (each class gets a distinct region color)
        self.input_ax.contourf(xx, yy, Z, levels=levels, cmap=cmap_light, alpha=0.5, antialiased=False)

        # Draw contour lines for decision boundaries
        self.input_ax.contour(xx, yy, Z, levels=np.arange(num_classes - 1) + 0.5, colors='k', linewidths=1.0)

        # Replot original points on top
        for x, y, class_index in data_points:
            color = class_colors[class_index]
            self.input_ax.plot(x, y, 'o', color=color, markersize=8, markeredgecolor='k')

        # Debug info
        unique_classes = np.unique(Z)
        print("Unique predicted classes on grid:", unique_classes)

        self.input_canvas.draw()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NeuralNetworkUI()
    window.show()
    sys.exit(app.exec())