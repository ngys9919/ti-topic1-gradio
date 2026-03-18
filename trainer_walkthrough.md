# MNIST CNN Training Walkthrough

I have created a single Python script `mnist_cnn_trainer.py` that fulfills all your requirements. Below is a summary of the implementation and how to run it.

## 🚀 Getting Started

### 1. Installation
Ensure you have the latest versions of the required libraries installed:
```bash
pip install keras>=3.0.0 torch torchvision gradio numpy matplotlib seaborn scikit-learn pillow
```

### 2. Execution
Run the script using Python:
```bash
python mnist_cnn_trainer.py
```
This will launch a local Gradio server. Look for the URL (e.g., `http://127.0.0.1:7860`) in your terminal to open the interface in your browser.

---

## 🛠️ Implementation Details

### Model Architecture
The script uses a standard CNN architecture suited for MNIST:
- **Input Layer**: `(28, 28, 1)`
- **Convolutional Layer 1**: 32 filters, 3x3 kernel, followed by MaxPooling
- **Convolutional Layer 2**: 64 filters, 3x3 kernel, followed by MaxPooling
- **Flatten & Dropout**: Reduces dimensionality and prevents overfitting
- **Dense Layers**: 128 units (configurable activation) followed by 10 units (Softmax)

### UI Components
- **Optimizer Select**: Supports `Adam`, `SGD`, `SGD + Momentum`, `RMSprop`, and `AdamW`.
- **Activation Select**: Choose from `relu`, `sigmoid`, `tanh`, `leaky_relu`, `elu`, and `swish`.
- **Hyperparameter Controls**: Sliders and inputs for Learning Rate, Epochs, and Batch Size.
- **Output Tabs**: Switch between Training History (plots) and Confusion Matrix.

### Progress Tracking
A custom Keras callback `GradioProgressCallback` is implemented to send epoch-by-epoch updates to the Gradio UI, including loss and accuracy metrics.

---

## 📈 Evaluation
Upon completion, the app provides:
1. **Accuracy/Loss Plots**: Visualizes training vs. validation performance.
2. **Confusion Matrix**: A heatmap showing where the model most commonly misclassifies digits.
3. **Training Summary**: A markdown breakdown of the configuration used and the final test results.
