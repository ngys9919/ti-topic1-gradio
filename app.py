import os

# Set Keras backend to PyTorch before importing Keras
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import gradio as gr
import io
from PIL import Image

def load_and_preprocess_data():
    """Loads and prepares the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension (MNIST is grayscale, so 1 channel)
    # Shape: (num_samples, 28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Train/Val split (10% validation)
    val_split = 0.1
    val_size = int(len(x_train) * val_split)
    
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def build_model(activation='relu'):
    """Builds a CNN model with the specified activation function."""
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation=activation),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation=activation),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation=activation),
        layers.Dense(10, activation="softmax"),
    ])
    return model

def get_optimizer(name, lr):
    """Returns the Keras optimizer based on selection."""
    if name == "Adam":
        return optimizers.Adam(learning_rate=lr)
    elif name == "SGD":
        return optimizers.SGD(learning_rate=lr)
    elif name == "SGD + Momentum":
        return optimizers.SGD(learning_rate=lr, momentum=0.9)
    elif name == "RMSprop":
        return optimizers.RMSprop(learning_rate=lr)
    elif name == "AdamW":
        return optimizers.AdamW(learning_rate=lr)
    else:
        return optimizers.Adam(learning_rate=lr)

class GradioProgressCallback(keras.callbacks.Callback):
    def __init__(self, progress_bar, epochs):
        self.progress_bar = progress_bar
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        if self.progress_bar:
            # Update progress bar: (progress_value, description)
            self.progress_bar((epoch + 1) / self.epochs, desc=f"Epoch {epoch+1}/{self.epochs} - loss: {logs.get('loss'):.4f}, acc: {logs.get('accuracy'):.4f}")

def plot_to_image(fig):
    """Converts a Matplotlib figure to a PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img

def train_and_evaluate(optimizer_name, activation_name, lr, epochs, batch_size, progress=gr.Progress()):
    """Main training and evaluation routine."""
    
    progress(0, desc="Loading data...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
    
    progress(0.1, desc="Building model...")
    model = build_model(activation=activation_name)
    optimizer = get_optimizer(optimizer_name, lr)
    
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    progress(0.15, desc="Starting training...")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=[GradioProgressCallback(progress, epochs)],
        verbose=0
    )
    
    progress(0.9, desc="Evaluating model...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    # Generate Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy Plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy', color='#4CAF50', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color='#2196F3', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Loss Plot
    ax2.plot(history.history['loss'], label='Train Loss', color='#F44336', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', color='#FF9800', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    metrics_plot = plot_to_image(fig)
    plt.close(fig)
    
    # Confusion Matrix
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('True Label')
    
    cm_plot = plot_to_image(fig_cm)
    plt.close(fig_cm)
    
    summary = (
        f"### Training Summary\n"
        f"- **Optimizer:** {optimizer_name}\n"
        f"- **Activation:** {activation_name}\n"
        f"- **Learning Rate:** {lr}\n"
        f"- **Epochs:** {epochs}\n"
        f"- **Batch Size:** {batch_size}\n"
        f"---\n"
        f"### Test Results\n"
        f"- **Test Accuracy:** {test_acc:.4%}\n"
        f"- **Test Loss:** {test_loss:.4f}"
    )
    
    return metrics_plot, cm_plot, summary

# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="MNIST CNN Trainer (Keras + PyTorch)") as demo:
    gr.Markdown(
        """
        # 🧠 MNIST CNN Trainer
        ### Powered by Keras 3 with PyTorch Backend
        Configure your model's hyperparameters and monitor the training process in real-time.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Configuration")
            opt_dropdown = gr.Dropdown(
                choices=["Adam", "SGD", "SGD + Momentum", "RMSprop", "AdamW"],
                value="Adam",
                label="Optimizer"
            )
            act_dropdown = gr.Dropdown(
                choices=["relu", "sigmoid", "tanh", "leaky_relu", "elu", "swish"],
                value="relu",
                label="Activation Function"
            )
            lr_slider = gr.Slider(
                minimum=0.0001,
                maximum=0.1,
                step=0.0001,
                value=0.001,
                label="Learning Rate"
            )
            epochs_input = gr.Number(value=5, label="Epochs", precision=0)
            batch_input = gr.Number(value=64, label="Batch Size", precision=0)
            
            train_btn = gr.Button("🚀 Start Training", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results")
            summary_output = gr.Markdown("Training results will appear here...")
            
            with gr.Tabs():
                with gr.TabItem("Accuracy & Loss"):
                    metrics_output = gr.Image(label="Training History")
                with gr.TabItem("Confusion Matrix"):
                    cm_output = gr.Image(label="Confusion Matrix")
    
    train_btn.click(
        fn=train_and_evaluate,
        inputs=[opt_dropdown, act_dropdown, lr_slider, epochs_input, batch_input],
        outputs=[metrics_output, cm_output, summary_output]
    )

if __name__ == "__main__":
    demo.launch()
