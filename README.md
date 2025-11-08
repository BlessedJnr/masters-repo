
# Neural Network UI Builder

## Features

* **Interactive Data Input:** Click on the left-hand grid to add data points (X, Y coordinates).
* **Dynamic Class Management:**
    * Change the "Number of Classes" (2-8).
    * The "Current Class" dropdown automatically updates.
    * Points are color-coded based on the currently selected class.
* **Hyperparameter Controls:** Set common parameters like Learning Rate, Activation Function, Epochs, and Bias.
* **Visualization:** Real-time plotting of input data and a placeholder specifically for training error graphs.

## Prerequisites

To run this application, you need to have **Python 3.8 or higher** installed on your system.

You can check your python version by running:
```bash
python --version
# or depending on your system:
python3 --version
````

## Installation

1.  **Save the Application Code:**
    Ensure you have saved the provided Python code into a file named `main_ui.py`.

2.  **Install Dependencies:**
    This project relies on `PyQt6` for the window framework and `matplotlib` for the graphing. Open your terminal or command prompt and run the following command:

    ```bash
    pip install PyQt6 matplotlib numpy
    ```

    *Note: If you are on Linux/macOS and standard `pip` doesn't work, try using `pip3`.*

## How to Run

1.  Navigate to the directory where you saved `nn_ui.py` using your terminal/command prompt.

    ```bash
    cd path/to/your/folder
    ```

2.  Execute the script:

    ```bash
    python main_ui.py
    ```

    *(On some systems, you might need to use `python3 main_ui.py`)*

3.  The application window should appear.

## Usage Guide

1.  **Set Classes:** Start by choosing how many classes you want in your dataset using the **Number of Classes** spin box (top right).
2.  **Select Active Class:** Use the **Current Class** dropdown to choose which class you want to add points for (e.g., "Class 1").
3.  **Add Data:** Click anywhere on the **Input Data** grid (left side) to add a point. It will be colored according to its class.
4.  **Switch & Repeat:** Change the **Current Class** to a different one (e.g., "Class 2") and add more points to see the colors change.
5.  **Test Controls:** Modify the Learning Rate, Activation Function, etc., to see how the UI elements interact.
6.  **Run Demo:** Click **Start Training (Demo)** to see the mock error plot animation on the bottom right.
7.  **Clear:** Use **Clear Input Points** to reset the grid and start over.

## Troubleshooting

  * **`ModuleNotFoundError: No module named 'PyQt6'`**:
    This means the installation failed. Ensure you ran `pip install PyQt6` successfully and that you are running the python interpreter that corresponds to where you installed the package.
  * **Graphics/Display Issues (Linux)**:
    Some Linux distributions require additional libraries for Qt6. If the window doesn't open, try installing standard Qt dependencies for your distribution (e.g., on Ubuntu: `sudo apt-get install libxcb-cursor0`).

<!-- end list -->
