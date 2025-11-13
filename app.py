"""
Main Entry Point: AI/ML Projects Hub

This is a unified interface to access all four projects:
1. Transformer from Scratch - Build and train LLaMA-style transformers
2. Stable Diffusion - Text-to-image generation
3. Mistral RAG - Context-aware question answering
4. BipedalWalker RL - Reinforcement learning agent training
"""

import gradio as gr
import sys
import os

# Add project directories to path
sys.path.insert(0, os.path.dirname(__file__))


def create_main_interface():
    """Create the main hub interface"""

    with gr.Blocks(title="AI/ML Projects Hub", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🚀 AI/ML Projects Hub

        Welcome to the interactive AI/ML projects collection! Choose a project below to get started.

        ---
        """)

        with gr.Tabs():
            # Project 1: Transformers
            with gr.Tab("🤖 Transformer from Scratch"):
                gr.Markdown("""
                ## Build and Train LLaMA-Style Transformers

                **What you'll learn:**
                - Modern transformer architecture (RMSNorm, RoPE, Multi-Head Attention, SwiGLU)
                - Training language models from scratch
                - Text generation
                - Interactive visualizations

                **Features:**
                - Configure model architecture
                - Train on Shakespeare dataset
                - Generate text
                - Visualize attention patterns and architecture

                **Model Size:** ~30K to 141M+ parameters

                ---
                """)

                gr.Markdown("### Quick Start")
                gr.Markdown("""
                1. Run in terminal: `python transformers_from_scratch/app.py`
                2. Or import and use programmatically

```python
# Example: Train a small transformer
from transformers_from_scratch.models import Llama
from transformers_from_scratch.utils import prepare_dataset, train, generate
import torch

# Setup
config = {
    'vocab_size': 65,
    'd_model': 128,
    'n_layers': 4,
    'n_heads': 8,
    'context_window': 16,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'batch_size': 32,
    'epochs': 1000,
    'log_interval': 10
}

# Create model
model = Llama(config)

# Load data
dataset, vocab, encode, decode = prepare_dataset('tinyshakespeare.txt')

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
history = train(model, optimizer, dataset, get_batches, config=config)

# Generate
generated = generate(model, config, max_new_tokens=100)
text = decode(generated[0].tolist())
print(text)
```
                """)

            # Project 2: Stable Diffusion
            with gr.Tab("🎨 Stable Diffusion"):
                gr.Markdown("""
                ## Text-to-Image Generation

                **What you'll learn:**
                - How diffusion models work
                - Text-to-image generation
                - Prompt engineering

                **Features:**
                - Generate high-quality images from text
                - Multiple quality presets
                - Parameter controls (steps, guidance, resolution)
                - Batch generation

                **Model:** Stable Diffusion XL (SSD-1B)

                ---
                """)

                gr.Markdown("### Quick Start")
                gr.Markdown("""
                1. Run in terminal: `python stable_diffusion/app.py`
                2. Or use programmatically

```python
# Example: Generate an image
from stable_diffusion.core import StableDiffusionGenerator

# Initialize
generator = StableDiffusionGenerator()
generator.load_model()

# Generate
images = generator.generate_image(
    prompt="A beautiful sunset over mountains, 8K, photorealistic",
    num_inference_steps=50,
    guidance_scale=7.5,
    num_images=1
)

# Save
images[0].save("output.jpg")
```
                """)

            # Project 3: Mistral RAG
            with gr.Tab("💬 Mistral RAG"):
                gr.Markdown("""
                ## Retrieval Augmented Generation

                **What you'll learn:**
                - RAG architecture
                - Vector similarity search
                - Context-aware question answering
                - Document indexing

                **Features:**
                - Index web documents
                - Ask questions with context
                - Compare RAG vs direct LLM
                - See retrieved sources

                **Model:** Mistral-7B-Instruct (4-bit quantized)

                ---
                """)

                gr.Markdown("### Quick Start")
                gr.Markdown("""
                1. Run in terminal: `python mistral_rag/app.py`
                2. Or use programmatically

```python
# Example: RAG question answering
from mistral_rag.core import MistralRAGSystem

# Initialize
rag = MistralRAGSystem()
rag.load_model()

# Index documents
urls = ["https://example.com/article1", "https://example.com/article2"]
rag.index_documents(urls)
rag.setup_rag_chain()

# Ask questions
result = rag.ask("What is the main topic?")
print(result['answer'])
print("Sources:", result['context'])
```
                """)

            # Project 4: RL BipedalWalker
            with gr.Tab("🤸 BipedalWalker RL"):
                gr.Markdown("""
                ## Reinforcement Learning Agent

                **What you'll learn:**
                - Reinforcement learning with PPO
                - Training agents in continuous action spaces
                - Reward shaping
                - Policy optimization

                **Features:**
                - Train walking agent
                - Configurable hyperparameters
                - Model evaluation
                - Save/load trained models

                **Environment:** BipedalWalker-v3 (Gymnasium)
                **Algorithm:** PPO (Proximal Policy Optimization)

                ---
                """)

                gr.Markdown("### Quick Start")
                gr.Markdown("""
                1. Run in terminal: `python rl_bipedal_walker/app.py`
                2. Or use programmatically

```python
# Example: Train a walking agent
from rl_bipedal_walker.core import BipedalWalkerTrainer

# Create trainer
trainer = BipedalWalkerTrainer(n_envs=16)
trainer.create_environment()
trainer.create_model()

# Train
trainer.train(total_timesteps=1_000_000)

# Evaluate
mean_reward, std_reward = trainer.evaluate()
print(f"Performance: {mean_reward:.2f} +/- {std_reward:.2f}")

# Save
trainer.save_model("my_walker.zip")
```
                """)

        gr.Markdown("""
        ---

        ## 📚 Project Overview

        | Project | Description | Key Technologies |
        |---------|-------------|------------------|
        | **Transformers** | Build LLaMA-style models from scratch | PyTorch, RMSNorm, RoPE, SwiGLU |
        | **Stable Diffusion** | Generate images from text | Diffusers, SDXL, Gradio |
        | **Mistral RAG** | Context-aware Q&A system | LangChain, FAISS, Mistral-7B |
        | **BipedalWalker RL** | Train walking agent | Stable-Baselines3, PPO, Gymnasium |

        ## 🚀 Getting Started

        ### Installation
        ```bash
        pip install -r requirements.txt
        ```

        ### Run Individual Projects
        ```bash
        python transformers_from_scratch/app.py
        python stable_diffusion/app.py
        python mistral_rag/app.py
        python rl_bipedal_walker/app.py
        ```

        ## 📖 Documentation

        Each project has detailed documentation in its respective directory:
        - `transformers_from_scratch/` - Transformer implementation details
        - `stable_diffusion/` - Image generation guide
        - `mistral_rag/` - RAG architecture explanation
        - `rl_bipedal_walker/` - RL training guide

        ## 🤝 Contributing

        Contributions are welcome! Please feel free to submit issues or pull requests.

        ## 📄 License

        This project is for educational purposes.
        """)

    return app


if __name__ == "__main__":
    app = create_main_interface()
    app.launch(server_name="0.0.0.0", share=True)
