import sys
import random
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QLabel
)
from PyQt6.QtCore import Qt

# Matplotlib imports for embedding plots in PyQt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# We'll use a dictionary to store our data points and their classes
# This will hold tuples of (x, y, class_index)
data_points = []
# Pre-define colors for the classes
class_colors = ['#FF0000', '#0000FF', '#00FF00', '#FFD700', '#FF00FF', '#00FFFF', '#FFA500', '#800080']


class NeuralNetworkUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network UI Builder")
        self.setGeometry(100, 100, 1200, 800)

        # --- Main Layout ---
        # A central widget to hold everything
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        # A horizontal layout: [Input Plot] | [Controls & Error Plot]
        main_layout = QHBoxLayout(main_widget)

        # --- 1. Input Plot (Left Side) ---
        # This is a Matplotlib Figure embedded in a PyQt Widget
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
        self.num_classes_spin.setRange(2, 8) # Let's set a reasonable max
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
        
        # Bias (using a checkbox as it's usually a boolean)
        self.bias_check = QCheckBox()
        self.bias_check.setChecked(True)
        param_layout.addRow("Use Bias:", self.bias_check)
        
        right_panel_layout.addWidget(param_widget)

        # Demo buttons
        self.train_button = QPushButton("Start Training (Demo)")
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
        
        # Connect number of classes to the dropdown update
        self.num_classes_spin.valueChanged.connect(self.update_class_dropdown)
        
        # Connect mouse clicks on the input plot
        self.input_canvas.mpl_connect('button_press_event', self.on_plot_click)
        
        # Connect demo buttons
        self.train_button.clicked.connect(self.plot_mock_error)
        self.clear_button.clicked.connect(self.clear_input_plot)

        # --- Final Setup ---
        self.update_class_dropdown(self.num_classes_spin.value())

    def setup_input_plot(self):
        """Sets up the initial state of the input plot with prominent axes."""
        self.input_ax.clear()
        self.input_ax.set_title("Input Data")
        self.input_ax.set_xlabel("X-coordinate")
        self.input_ax.set_ylabel("Y-coordinate")
        
        # Standard faint grid
        self.input_ax.grid(True, linestyle='--', alpha=0.6)
        
        # --- NEW: Prominent Axes Lines ---
        # axhline draws a horizontal line at y=0
        self.input_ax.axhline(y=0, color='black', linewidth=1.2)
        # axvline draws a vertical line at x=0
        self.input_ax.axvline(x=0, color='black', linewidth=1.2)
        
        # Set fixed axes for consistent plotting
        self.input_ax.set_xlim(-10, 10)
        self.input_ax.set_ylim(-10, 10)
        self.input_canvas.draw()

    def setup_error_plot(self):
        """Sets up the initial state of the error plot."""
        self.error_ax.clear()
        self.error_ax.set_title("Training Error")
        self.error_ax.set_xlabel("Epoch")
        self.error_ax.set_ylabel("Error")
        self.error_ax.grid(True)
        self.error_canvas.draw()

    def update_class_dropdown(self, num_classes):
        """Clears and repopulates the 'Class' dropdown."""
        self.class_combo.clear()
        class_names = [f"Class {i+1}" for i in range(num_classes)]
        self.class_combo.addItems(class_names)

    def on_plot_click(self, event):
        """Handles clicks on the input plot to add data points."""
        # Only register clicks inside the plot area
        if event.inaxes != self.input_ax:
            return

        x, y = event.xdata, event.ydata
        class_index = self.class_combo.currentIndex()
        
        # Store the data
        data_points.append((x, y, class_index))
        
        # Plot the new point
        color = class_colors[class_index]
        self.input_ax.plot(x, y, 'o', color=color, markersize=8)
        self.input_canvas.draw()

    def clear_input_plot(self):
        """Clears all data points and resets the plot."""
        global data_points
        data_points = []
        self.setup_input_plot()

    def plot_mock_error(self):
        """Demo function for error plotting."""
        num_epochs = self.epoch_spin.value()
        epochs = np.arange(1, num_epochs + 1)
        error = (1 / epochs) + (np.random.rand(num_epochs) * 0.1)
        error = np.sort(error)[::-1]
        
        self.error_ax.clear()
        self.error_ax.set_title("Training Error")
        self.error_ax.set_xlabel("Epoch")
        self.error_ax.set_ylabel("Error")
        self.error_ax.grid(True)
        self.error_ax.plot(epochs, error, color='b')
        self.error_canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NeuralNetworkUI()
    window.show()
    sys.exit(app.exec())