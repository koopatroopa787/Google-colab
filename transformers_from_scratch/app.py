"""
Gradio Web Interface for Transformer Training and Visualization

This interactive app allows you to:
- Configure and train a transformer model
- Visualize the architecture
- Generate text
- Monitor training progress
- Explore attention patterns
"""

import gradio as gr
import torch
import matplotlib.pyplot as plt
import pandas as pd
from models.llama import Llama
from models.components import *
from utils.data import prepare_dataset, get_batches
from utils.training import train, evaluate_loss, generate, get_model_info
from visualization.visualizer import (
    visualize_architecture,
    visualize_training_progress_interactive,
    visualize_model_size,
    create_transformer_flow_diagram
)

# Global variables to store model and dataset
current_model = None
current_dataset = None
current_config = None
encode_fn = None
decode_fn = None
training_history = None


def setup_model(vocab_size, d_model, n_layers, n_heads, context_window):
    """Setup model with given configuration"""
    global current_model, current_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_config = {
        'vocab_size': int(vocab_size),
        'd_model': int(d_model),
        'n_layers': int(n_layers),
        'n_heads': int(n_heads),
        'context_window': int(context_window),
        'device': device,
        'batch_size': 32,
        'epochs': 1000,
        'log_interval': 10
    }

    current_model = Llama(current_config)

    return f"✓ Model created successfully!\n\nConfiguration:\n{'-'*40}\n" + \
           "\n".join([f"{k}: {v}" for k, v in current_config.items()])


def load_dataset_fn(dataset_choice):
    """Load and prepare dataset"""
    global current_dataset, encode_fn, decode_fn, current_config

    try:
        if dataset_choice == "TinyShakespeare":
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            file_path = "tinyshakespeare.txt"
        else:
            return "Dataset not available"

        current_dataset, vocab, encode_fn, decode_fn = prepare_dataset(file_path, url)

        # Update vocab size in config if model exists
        if current_config:
            current_config['vocab_size'] = len(vocab)

        return f"✓ Dataset loaded!\n\nStats:\n{'-'*40}\n" + \
               f"Total characters: {len(current_dataset):,}\n" + \
               f"Vocabulary size: {len(vocab)}\n" + \
               f"Train size: {int(len(current_dataset)*0.8):,}\n" + \
               f"Val size: {int(len(current_dataset)*0.1):,}\n" + \
               f"Test size: {int(len(current_dataset)*0.1):,}"
    except Exception as e:
        return f"❌ Error loading dataset: {str(e)}"


def train_model_fn(epochs, learning_rate):
    """Train the model"""
    global current_model, current_dataset, current_config, training_history

    if current_model is None:
        return "Please create a model first!"

    if current_dataset is None:
        return "Please load a dataset first!"

    current_config['epochs'] = int(epochs)

    # Setup optimizer
    optimizer = torch.optim.Adam(current_model.parameters(), lr=float(learning_rate))

    # Train
    training_history = train(
        current_model,
        optimizer,
        current_dataset,
        get_batches,
        config=current_config,
        print_logs=True
    )

    return f"✓ Training complete!\n\nFinal Results:\n{'-'*40}\n" + \
           f"Train Loss: {training_history.iloc[-1]['train']:.4f}\n" + \
           f"Val Loss: {training_history.iloc[-1]['val']:.4f}"


def generate_text_fn(num_tokens, temperature, num_samples):
    """Generate text using the trained model"""
    global current_model, current_config, decode_fn

    if current_model is None:
        return "❌ Please create a model first!"

    if decode_fn is None:
        return "❌ Please load a dataset first!\n\nGo to the 'Data' tab and load TinyShakespeare dataset."

    try:
        # Generate
        generated_indices = generate(
            current_model,
            current_config,
            max_new_tokens=int(num_tokens),
            temperature=float(temperature),
            num_samples=int(num_samples)
        )

        # Decode
        texts = [decode_fn(indices.tolist()) for indices in generated_indices]

        result = "\n\n" + "="*60 + "\n\n"
        result += "\n\n".join([f"Sample {i+1}:\n{text}" for i, text in enumerate(texts)])

        return result
    except Exception as e:
        return f"❌ Error generating text: {str(e)}"


def visualize_architecture_fn():
    """Visualize model architecture"""
    global current_config

    if current_config is None:
        # Return empty figure with message
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No model configuration available.\nPlease create a model first.',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.axis('off')
        return fig

    try:
        fig = visualize_architecture(current_config)
        return fig
    except Exception as e:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error creating visualization:\n{str(e)}',
                ha='center', va='center', fontsize=12, color='red', transform=ax.transAxes)
        ax.axis('off')
        return fig


def visualize_training_fn():
    """Visualize training progress"""
    global training_history

    if training_history is None:
        # Return empty plot with message
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_annotation(
            text="No training history available.<br>Please train a model first.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=400
        )
        return fig

    try:
        fig = visualize_training_progress_interactive(training_history)
        return fig
    except Exception as e:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating visualization:<br>{str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig


def visualize_model_info_fn():
    """Visualize model information"""
    global current_model

    if current_model is None:
        # Return empty plot with message
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_annotation(
            text="No model available.<br>Please create a model first.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=400
        )
        return fig

    try:
        info = get_model_info(current_model)
        fig = visualize_model_size(info)
        return fig
    except Exception as e:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating visualization:<br>{str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig


def create_interface():
    """Create Gradio interface"""

    with gr.Blocks(title="Transformer from Scratch", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🤖 Transformer from Scratch: Interactive Training & Visualization

        Build, train, and understand transformer models with modern architecture (LLaMA-style):
        - **RMSNorm**: Efficient normalization
        - **RoPE**: Rotary Position Embeddings
        - **Multi-Head Attention**: Self-attention mechanism
        - **SwiGLU**: Advanced activation function

        """)

        with gr.Tabs():
            # Tab 1: Model Setup
            with gr.Tab("📐 Model Setup"):
                gr.Markdown("### Configure Your Transformer Model")

                with gr.Row():
                    with gr.Column():
                        vocab_size_input = gr.Number(label="Vocabulary Size", value=65)
                        d_model_input = gr.Slider(64, 1024, value=128, step=64,
                                                   label="Model Dimension (d_model)")
                        n_layers_input = gr.Slider(1, 12, value=4, step=1,
                                                    label="Number of Layers")
                        n_heads_input = gr.Slider(1, 16, value=8, step=1,
                                                   label="Number of Attention Heads")
                        context_window_input = gr.Slider(8, 256, value=16, step=8,
                                                          label="Context Window")

                        setup_btn = gr.Button("🚀 Create Model", variant="primary")

                    with gr.Column():
                        setup_output = gr.Textbox(label="Model Status", lines=15)

                setup_btn.click(
                    setup_model,
                    inputs=[vocab_size_input, d_model_input, n_layers_input,
                           n_heads_input, context_window_input],
                    outputs=setup_output
                )

            # Tab 2: Data Loading
            with gr.Tab("📚 Data"):
                gr.Markdown("### Load Training Dataset")

                dataset_choice = gr.Radio(
                    ["TinyShakespeare"],
                    label="Choose Dataset",
                    value="TinyShakespeare"
                )
                load_btn = gr.Button("📥 Load Dataset", variant="primary")
                data_output = gr.Textbox(label="Dataset Status", lines=10)

                load_btn.click(load_dataset_fn, inputs=dataset_choice, outputs=data_output)

            # Tab 3: Training
            with gr.Tab("🏋️ Training"):
                gr.Markdown("### Train Your Model")

                with gr.Row():
                    with gr.Column():
                        epochs_input = gr.Slider(100, 10000, value=1000, step=100,
                                                label="Training Epochs")
                        lr_input = gr.Number(label="Learning Rate", value=0.001)
                        train_btn = gr.Button("🎯 Start Training", variant="primary")

                    with gr.Column():
                        train_output = gr.Textbox(label="Training Status", lines=10)

                train_btn.click(
                    train_model_fn,
                    inputs=[epochs_input, lr_input],
                    outputs=train_output
                )

            # Tab 4: Generation
            with gr.Tab("✍️ Text Generation"):
                gr.Markdown("### Generate Text with Your Model")

                with gr.Row():
                    with gr.Column():
                        num_tokens_input = gr.Slider(10, 500, value=100, step=10,
                                                     label="Tokens to Generate")
                        temperature_input = gr.Slider(0.1, 2.0, value=1.0, step=0.1,
                                                     label="Temperature (randomness)")
                        num_samples_input = gr.Slider(1, 5, value=1, step=1,
                                                     label="Number of Samples")
                        generate_btn = gr.Button("🎲 Generate Text", variant="primary")

                    with gr.Column():
                        generated_output = gr.Textbox(label="Generated Text",
                                                     lines=20, max_lines=30)

                generate_btn.click(
                    generate_text_fn,
                    inputs=[num_tokens_input, temperature_input, num_samples_input],
                    outputs=generated_output
                )

            # Tab 5: Visualizations
            with gr.Tab("📊 Visualizations"):
                gr.Markdown("### Explore Model Architecture and Training")

                with gr.Tabs():
                    with gr.Tab("Architecture"):
                        arch_btn = gr.Button("🏗️ Show Architecture")
                        arch_plot = gr.Plot(label="Model Architecture")
                        arch_btn.click(visualize_architecture_fn, outputs=arch_plot)

                    with gr.Tab("Data Flow"):
                        flow_btn = gr.Button("🔄 Show Data Flow")
                        flow_plot = gr.Plot(label="Transformer Data Flow")
                        flow_btn.click(create_transformer_flow_diagram, outputs=flow_plot)

                    with gr.Tab("Training Progress"):
                        train_viz_btn = gr.Button("📈 Show Training History")
                        train_plot = gr.Plot(label="Training Progress")
                        train_viz_btn.click(visualize_training_fn, outputs=train_plot)

                    with gr.Tab("Model Info"):
                        info_btn = gr.Button("ℹ️ Show Model Info")
                        info_plot = gr.Plot(label="Parameter Distribution")
                        info_btn.click(visualize_model_info_fn, outputs=info_plot)

            # Tab 6: About
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## About This Implementation

                This is a from-scratch implementation of a transformer model using modern architectural choices
                inspired by LLaMA and other state-of-the-art models.

                ### Key Components:

                **1. RMSNorm (Root Mean Square Normalization)**
                - More efficient than LayerNorm
                - Used in LLaMA and other modern transformers
                - Formula: `RMS(x) = x / sqrt(mean(x²))`

                **2. RoPE (Rotary Position Embeddings)**
                - Encodes positional information through rotation
                - Preserves relative position information
                - Better than absolute position embeddings

                **3. Multi-Head Attention**
                - Multiple attention heads attend to different aspects
                - Scaled dot-product attention with causal masking
                - Formula: `Attention(Q,K,V) = softmax(QK^T / sqrt(d))V`

                **4. SwiGLU Activation**
                - Gated Linear Unit with Swish activation
                - Better than ReLU for transformers
                - Formula: `SwiGLU(x) = (xW_gate * σ(xW_gate)) ⊙ (xW)`

                ### Architecture Flow:
                1. **Input** → Token IDs
                2. **Embedding** → Dense vectors
                3. **Transformer Blocks** (repeated N times):
                   - RMSNorm
                   - Multi-Head Attention with RoPE
                   - Residual connection
                   - RMSNorm
                   - Feedforward + SwiGLU
                   - Residual connection
                4. **Output FFN** → Vocabulary predictions

                ### Training:
                - Next-token prediction (language modeling)
                - Cross-entropy loss
                - Adam optimizer

                ---

                **Dataset**: TinyShakespeare (1.1M characters from Shakespeare's works)

                **Model Size**: Configurable from ~30K to 141M+ parameters
                """)

    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(share=True)
