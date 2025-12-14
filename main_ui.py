import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QLabel, QGridLayout
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap

# Import our other Python files
try:
    from activation_function import Activation
    from mln_network import NeuralNetwork
except ImportError:
    print("Error: Make sure 'activation_function.py' and 'mln_network.py' are in the same folder.")
    sys.exit(1)

# Global data storage: (x, y, class_idx)
data_points = []
class_colors = ['#FF0000', '#0000FF', '#00FF00', '#FFD700',
                '#FF00FF', '#00FFFF', '#FFA500', '#800080']


class NeuralNetworkUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network UI Builder")
        self.setGeometry(100, 100, 1200, 800)

        self.trained_network = None
        self.X_mean = None
        self.X_std = None

        # === Main Layout ===
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --- Left: Input Plot ---
        self.input_fig = Figure(figsize=(5, 5))
        self.input_canvas = FigureCanvas(self.input_fig)
        self.input_ax = self.input_fig.add_subplot(111)
        self.setup_input_plot()
        main_layout.addWidget(self.input_canvas)

        # --- Right: Controls + Error Plot ---
        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)
        main_layout.addWidget(right_panel_widget)

        # === Parameter Panel ===
        param_widget = QWidget()
        param_layout = QVBoxLayout(param_widget)

        grid = QGridLayout()
        param_layout.addLayout(grid)

        # Problem type
        self.problem_type_combo = QComboBox()
        self.problem_type_combo.addItems(["Classification", "Regression"])

        # Number of classes
        self.num_classes_spin = QSpinBox()
        self.num_classes_spin.setRange(2, 8)
        self.num_classes_spin.setValue(2)

        # Current class (for classification)
        self.class_combo = QComboBox()

        # Learning rate
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(3)
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setSingleStep(0.001)

        # Activation function (for hidden layers)
        self.activation_combo = QComboBox()
        self.activation_combo.addItems(["Sigmoid", "ReLU", "Tanh", "Linear"])

        # Epochs
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 10000)
        self.epoch_spin.setValue(100)

        # Max error (early stopping)
        self.max_error_spin = QDoubleSpinBox()
        self.max_error_spin.setDecimals(4)
        self.max_error_spin.setRange(1e-6, 10.0)
        self.max_error_spin.setValue(0.01)
        self.max_error_spin.setSingleStep(0.0001)

        # Bias
        self.bias_check = QCheckBox()
        self.bias_check.setChecked(True)

        # ---- 2x2 grid placement ----
        grid.addWidget(QLabel("Problem Type:"),        0, 0)
        grid.addWidget(self.problem_type_combo,        0, 1)
        grid.addWidget(QLabel("Number of Classes:"),   0, 2)
        grid.addWidget(self.num_classes_spin,          0, 3)

        grid.addWidget(QLabel("Current Class:"),       1, 0)
        grid.addWidget(self.class_combo,               1, 1)
        grid.addWidget(QLabel("Learning Rate:"),       1, 2)
        grid.addWidget(self.lr_spin,                   1, 3)

        grid.addWidget(QLabel("Activation Function:"), 2, 0)
        grid.addWidget(self.activation_combo,          2, 1)
        grid.addWidget(QLabel("Epochs:"),              2, 2)
        grid.addWidget(self.epoch_spin,                2, 3)

        grid.addWidget(QLabel("Max Error:"),           3, 0)
        grid.addWidget(self.max_error_spin,            3, 1)
        grid.addWidget(QLabel("Use Bias:"),            3, 2)
        grid.addWidget(self.bias_check,                3, 3)

        # Hidden layers (full width)
        self.hidden_layers_spin = QSpinBox()
        self.hidden_layers_spin.setRange(1, 10)
        self.hidden_layers_spin.setValue(1)

        param_layout.addWidget(QLabel("Hidden Layers:"))
        param_layout.addWidget(self.hidden_layers_spin)

        self.hidden_layer_boxes = []
        self.hidden_layers_container = QVBoxLayout()
        self.hidden_layers_widget = QWidget()
        self.hidden_layers_widget.setLayout(self.hidden_layers_container)
        param_layout.addWidget(QLabel("Neurons per Layer:"))
        param_layout.addWidget(self.hidden_layers_widget)

        right_panel_layout.addWidget(param_widget)

        # Buttons
        self.train_button = QPushButton("Start Training")
        self.clear_button = QPushButton("Clear Input Points")
        right_panel_layout.addWidget(self.train_button)
        right_panel_layout.addWidget(self.clear_button)

        # Error plot
        self.error_fig = Figure(figsize=(5, 4))
        self.error_canvas = FigureCanvas(self.error_fig)
        self.error_ax = self.error_fig.add_subplot(111)
        self.setup_error_plot()
        right_panel_layout.addWidget(self.error_canvas)

        # ✅ Total error label (under graph)
        self.total_error_label = QLabel("Total Error: N/A")
        self.total_error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_panel_layout.addWidget(self.total_error_label)

        # Signals
        self.num_classes_spin.valueChanged.connect(self.update_class_dropdown)
        self.hidden_layers_spin.valueChanged.connect(self.update_hidden_layer_boxes)
        self.input_canvas.mpl_connect('button_press_event', self.on_plot_click)
        self.train_button.clicked.connect(self.start_training)
        self.clear_button.clicked.connect(self.clear_input_plot)

        self.update_class_dropdown(self.num_classes_spin.value())
        self.update_hidden_layer_boxes(self.hidden_layers_spin.value())

    # ====================== Plot setup ======================
    def setup_input_plot(self):
        self.input_ax.clear()
        self.input_ax.set_title("Input Data")
        self.input_ax.set_xlabel("X-coordinate")
        self.input_ax.set_ylabel("Y-coordinate")
        self.input_ax.grid(True, linestyle='--', alpha=0.6)
        self.input_ax.axhline(y=0, color='black', linewidth=1.2)
        self.input_ax.axvline(x=0, color='black', linewidth=1.2)
        self.input_ax.set_xlim(-10, 10)
        self.input_ax.set_ylim(-10, 10)
        self.input_canvas.draw_idle()

    def setup_error_plot(self):
        self.error_ax.clear()
        self.error_ax.set_title("Training Error")
        self.error_ax.set_xlabel("Epoch")
        self.error_ax.set_ylabel("Error")
        self.error_ax.grid(True)
        self.error_canvas.draw_idle()

    # ====================== UI helpers ======================
    def update_class_dropdown(self, num_classes):
        self.class_combo.clear()
        class_names = [f"Class {i+1}" for i in range(num_classes)]
        self.class_combo.addItems(class_names)

    def update_hidden_layer_boxes(self, count):
        for box in self.hidden_layer_boxes:
            box.deleteLater()
        self.hidden_layer_boxes = []

        for i in range(count):
            spin = QSpinBox()
            spin.setRange(1, 100)
            spin.setValue(4)
            spin.setPrefix(f"Layer {i+1}: ")
            self.hidden_layers_container.addWidget(spin)
            self.hidden_layer_boxes.append(spin)

    # ====================== Mouse interaction ======================
    def on_plot_click(self, event):
        if event.inaxes != self.input_ax:
            return

        x, y = event.xdata, event.ydata
        if self.problem_type_combo.currentText() == "Classification":
            class_index = self.class_combo.currentIndex()
            color = class_colors[class_index]
        else:
            class_index = 0
            color = 'k'

        data_points.append((x, y, class_index))
        self.input_ax.plot(x, y, 'o', color=color, markersize=8, markeredgecolor='k')
        self.input_canvas.draw_idle()

    def clear_input_plot(self):
        global data_points
        data_points = []
        self.trained_network = None
        self.setup_input_plot()
        self.setup_error_plot()
        self.total_error_label.setText("Total Error: N/A")


    # ====================== Training (single entry point) ======================
    def start_training(self):
        problem_type = self.problem_type_combo.currentText().lower()
        if problem_type == "classification":
            self._start_training_classification()
        else:
            self._start_training_regression()

    # ----------------- Classification -----------------
    def _start_training_classification(self):
        global data_points
        if not data_points:
            print("No data points to train on (classification).")
            return

        learning_rate = self.lr_spin.value()
        num_epochs = self.epoch_spin.value()
        num_classes = self.num_classes_spin.value()
        use_bias = self.bias_check.isChecked()
        max_error = self.max_error_spin.value()
        act_name = self.activation_combo.currentText()
        act_func, act_deriv = Activation.get_activation(act_name)

        X_train = np.array([[p[0], p[1]] for p in data_points])
        Y_indices = np.array([p[2] for p in data_points])
        Y_train = np.zeros((len(Y_indices), num_classes))
        Y_train[np.arange(len(Y_indices)), Y_indices] = 1

        self.X_mean = X_train.mean(axis=0)
        self.X_std = X_train.std(axis=0) + 1e-8
        X_norm = (X_train - self.X_mean) / self.X_std

        input_dim = 2
        output_dim = num_classes
        hidden_layer_sizes = [box.value() for box in self.hidden_layer_boxes]

        print(f"[CLS] Architecture: {input_dim} -> {hidden_layer_sizes} -> {output_dim}")
        print(f"[CLS] LR={learning_rate}, max_epochs={num_epochs}, max_error={max_error}")

        self.trained_network = NeuralNetwork(
            input_dim=input_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            output_dim=output_dim,
            use_bias=use_bias
        )

        error_history = self.trained_network.train(
            X_norm, Y_train,
            learning_rate=learning_rate,
            epochs=num_epochs,
            max_error=max_error,
            act_func=act_func,
            act_deriv=act_deriv,
            problem_type="classification"
        )

        self._plot_error_history(error_history)

        if self.trained_network is not None:
            self.plot_decision_boundary_classification(self.trained_network, act_func)

        # ----------------- Regression -----------------
    def _start_training_regression(self):
        global data_points
        if not data_points:
            print("No data points to train on (regression).")
            return

        learning_rate = self.lr_spin.value()
        num_epochs = self.epoch_spin.value()
        use_bias = self.bias_check.isChecked()
        max_error = self.max_error_spin.value()

        # ---- Get activation for HIDDEN layers ----
        act_name = self.activation_combo.currentText()

        # IMPORTANT FIX:
        # For regression we always want a NON-LINEAR hidden activation,
        # the output is already linear inside NeuralNetwork.train(...)
        if act_name == "Linear":
            print("[REG] 'Linear' activation selected; using ReLU for hidden layers instead.")
            act_name = "ReLU"

        act_func, act_deriv = Activation.get_activation(act_name)

        # For regression: X = [[x]], Y = [[y]]
        X_raw = np.array([[p[0]] for p in data_points])
        Y_train = np.array([[p[1]] for p in data_points])

        self.X_mean = X_raw.mean(axis=0)
        self.X_std = X_raw.std(axis=0) + 1e-8
        X_norm = (X_raw - self.X_mean) / self.X_std

        input_dim = 1
        output_dim = 1
        hidden_layer_sizes = [box.value() for box in self.hidden_layer_boxes]

        print(f"[REG] Architecture: {input_dim} -> {hidden_layer_sizes} -> {output_dim}")
        print(f"[REG] LR={learning_rate}, max_epochs={num_epochs}, max_error={max_error}")
        print(f"[REG] Hidden activation: {act_name}, output activation: Linear")

        self.trained_network = NeuralNetwork(
            input_dim=input_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            output_dim=output_dim,
            use_bias=use_bias
        )

        error_history = self.trained_network.train(
            X_norm, Y_train,
            learning_rate=learning_rate,
            epochs=num_epochs,
            max_error=max_error,
            act_func=act_func,
            act_deriv=act_deriv,
            problem_type="regression"
        )

        self._plot_error_history(error_history)

        if self.trained_network is not None:
            self.plot_regression_fit(X_raw, Y_train, self.trained_network, act_func)

    # ----------------- Shared error plot -----------------
    def _plot_error_history(self, error_history):
        print("Error history length:", len(error_history))
        self.error_ax.clear()
        self.error_ax.set_title("Training Error")
        self.error_ax.set_xlabel("Epoch")
        self.error_ax.set_ylabel("Error")
        self.error_ax.grid(True)

        if error_history:
            epochs_range = np.arange(1, len(error_history) + 1)
            self.error_ax.plot(epochs_range, error_history)

            # ✅ Show total (final) error
            final_error = error_history[-1]
            self.total_error_label.setText(f"Total Error (final): {final_error:.6f}")
        else:
            self.total_error_label.setText("Total Error: N/A")
            print("Warning: empty error history – nothing to plot.")

        self.error_canvas.draw_idle()


    # ====================== Classification decision boundary ======================
    def plot_decision_boundary_classification(self, nn, hidden_act_func):
        global data_points

        self.setup_input_plot()

        x_min, x_max = self.input_ax.get_xlim()
        y_min, y_max = self.input_ax.get_ylim()

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_norm = (grid_points - self.X_mean) / self.X_std

        Z = nn.predict_classes(grid_norm, hidden_act_func)
        Z = Z.reshape(xx.shape)

        num_classes = self.num_classes_spin.value()
        cmap_light = ListedColormap(class_colors[:num_classes])
        levels = np.arange(-0.5, num_classes + 0.5, 1)

        self.input_ax.contourf(
            xx, yy, Z,
            levels=levels,
            cmap=cmap_light,
            alpha=0.5,
            antialiased=False
        )

        self.input_ax.contour(
            xx, yy, Z,
            levels=np.arange(num_classes - 1) + 0.5,
            colors='k',
            linewidths=1.0
        )

        # Plot original points
        for x, y, class_index in data_points:
            color = class_colors[class_index]
            self.input_ax.plot(x, y, 'o', color=color,
                               markersize=8, markeredgecolor='k')

        unique_classes = np.unique(Z)
        print("Decision boundary unique predicted classes:", unique_classes)

        self.input_canvas.draw_idle()

    # ====================== Regression curve plotting ======================
    def plot_regression_fit(self, X_raw, Y_train, nn, hidden_act_func):
        """
        Draws:
        - training points (black x)
        - polynomial fit (red)
        - NN regression fit (green)
        """
        self.setup_input_plot()

        x_vals = X_raw.flatten()
        y_vals = Y_train.flatten()

        # Training points
        self.input_ax.plot(x_vals, y_vals, 'kx', label="Training data")

        # Polynomial baseline (degree up to 3)
        if len(x_vals) >= 2:
            degree = min(3, len(x_vals) - 1)
            coeffs = np.polyfit(x_vals, y_vals, degree)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
            y_poly = np.polyval(coeffs, x_line)
            self.input_ax.plot(x_line, y_poly, 'r', label=f"Poly deg {degree}")

        # NN prediction curve
        x_line = np.linspace(x_vals.min(), x_vals.max(), 200).reshape(-1, 1)
        x_line_norm = (x_line - self.X_mean) / self.X_std
        y_nn = nn.predict_regression(x_line_norm, hidden_act_func).flatten()
        self.input_ax.plot(x_line, y_nn, 'g', label="NN fit")

        self.input_ax.legend()
        self.input_canvas.draw_idle()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NeuralNetworkUI()
    window.show()
    sys.exit(app.exec())
