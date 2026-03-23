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
from textwrap import dedent

# Add project directories to path
sys.path.insert(0, os.path.dirname(__file__))


INTRO_MARKDOWN = dedent(
    """
    # 🚀 AI/ML Projects Hub

    Welcome to the interactive AI/ML projects collection! Choose a project below to get started.

    ---
    """
)

PROJECT_TABS = [
    {
        "label": "🤖 Transformer from Scratch",
        "overview": dedent(
            """
            ## Build and Train LLaMA-Style Transformers

            **What you'll learn:**
            - Modern transformer architecture (RMSNorm, RoPE, Multi-Head Attention, SwiGLU)
            - Training language models from scratch
            - Text generation and decoding strategies
            - Interactive architecture and attention visualizations

            **Features:**
            - Configure model architecture
            - Train on Shakespeare dataset
            - Generate text samples
            - Visualize attention patterns and architecture

            **Model Size:** ~30K to 141M+ parameters
            """
        ),
        "quick_start": dedent(
            """
            Run in a terminal:
            ```bash
            python transformers_from_scratch/app.py
            ```

            Or use programmatically:
            ```python
            import torch
            from transformers_from_scratch.models import Llama
            from transformers_from_scratch.utils import (
                prepare_dataset,
                get_batches,
                train,
                generate,
            )

            config = {
                'vocab_size': 65,
                'd_model': 128,
                'n_layers': 4,
                'n_heads': 8,
                'context_window': 16,
                'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                'batch_size': 32,
                'epochs': 200,
                'log_interval': 10,
            }

            dataset, vocab, encode, decode = prepare_dataset('tinyshakespeare.txt')
            model = Llama(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

            history = train(model, optimizer, dataset, get_batches, config=config, print_logs=True)
            sample = generate(model, config, max_new_tokens=100)
            print(decode(sample[0].tolist()))
            ```
            """
        ),
    },
    {
        "label": "🎨 Stable Diffusion",
        "overview": dedent(
            """
            ## Text-to-Image Generation

            **What you'll learn:**
            - How diffusion models work
            - Prompt engineering techniques
            - Balancing speed vs. quality

            **Features:**
            - Generate high-quality images from text
            - Multiple quality presets
            - Parameter controls (steps, guidance, resolution)
            - Batch generation

            **Model:** Stable Diffusion XL (SSD-1B)
            """
        ),
        "quick_start": dedent(
            """
            Run in a terminal:
            ```bash
            python stable_diffusion/app.py
            ```

            Or use programmatically:
            ```python
            from stable_diffusion.core import StableDiffusionGenerator, ImageGenerationPresets

            generator = StableDiffusionGenerator()
            generator.load_model()

            preset = ImageGenerationPresets.get_preset("Balanced")
            images = generator.generate_image(
                prompt="A beautiful sunset over mountains, 8K, photorealistic",
                num_inference_steps=preset["num_inference_steps"],
                guidance_scale=preset["guidance_scale"],
                num_images=1,
            )

            images[0].save("output.jpg")
            ```
            """
        ),
    },
    {
        "label": "💬 Mistral RAG",
        "overview": dedent(
            """
            ## Retrieval Augmented Generation

            **What you'll learn:**
            - RAG architecture
            - Vector similarity search
            - Context-aware question answering
            - Document indexing and source attribution

            **Features:**
            - Index web documents
            - Ask questions with context
            - Compare RAG vs direct LLM
            - See retrieved sources

            **Model:** Mistral-7B-Instruct (4-bit quantized)
            """
        ),
        "quick_start": dedent(
            """
            Run in a terminal:
            ```bash
            python mistral_rag/app.py
            ```

            Or use programmatically:
            ```python
            from mistral_rag.core import MistralRAGSystem

            rag = MistralRAGSystem()
            rag.load_model()

            urls = ["https://example.com/article1", "https://example.com/article2"]
            rag.index_documents(urls)
            rag.setup_rag_chain()

            result = rag.ask("What is the main topic?")
            print(result["answer"])
            print("Sources:", result["context"])
            ```
            """
        ),
    },
    {
        "label": "🤸 BipedalWalker RL",
        "overview": dedent(
            """
            ## Reinforcement Learning Agent

            **What you'll learn:**
            - Reinforcement learning with PPO
            - Training agents in continuous action spaces
            - Reward shaping and policy optimization

            **Features:**
            - Train walking agent
            - Configurable hyperparameters
            - Model evaluation
            - Save/load trained models

            **Environment:** BipedalWalker-v3 (Gymnasium)  
            **Algorithm:** PPO (Proximal Policy Optimization)
            """
        ),
        "quick_start": dedent(
            """
            Run in a terminal:
            ```bash
            python rl_bipedal_walker/app.py
            ```

            Or use programmatically:
            ```python
            from rl_bipedal_walker.core import BipedalWalkerTrainer

            trainer = BipedalWalkerTrainer(n_envs=4)
            trainer.create_model()
            trainer.train(total_timesteps=200_000)

            mean_reward, std_reward = trainer.evaluate()
            print(f"Performance: {mean_reward:.2f} +/- {std_reward:.2f}")

            trainer.save_model("my_walker.zip")
            ```
            """
        ),
    },
]

OVERVIEW_MARKDOWN = dedent(
    """
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
    """
)


def create_main_interface():
    """Create the main hub interface"""

    with gr.Blocks(title="AI/ML Projects Hub", theme=gr.themes.Soft()) as app:
        gr.Markdown(INTRO_MARKDOWN)

        with gr.Tabs():
            for project in PROJECT_TABS:
                with gr.Tab(project["label"]):
                    gr.Markdown(project["overview"])
                    gr.Markdown("### Quick Start")
                    gr.Markdown(project["quick_start"])

        gr.Markdown(OVERVIEW_MARKDOWN)

    return app


if __name__ == "__main__":
    app = create_main_interface()
    app.launch(server_name="0.0.0.0", share=True)
