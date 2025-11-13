# 🚀 AI/ML Projects Collection

A comprehensive collection of AI/ML projects with interactive web interfaces and detailed visualizations.

## 📋 Projects Overview

### 1. 🤖 Transformer from Scratch
Build and train LLaMA-style transformer models from the ground up.

**Features:**
- Complete transformer implementation with modern components:
  - **RMSNorm**: Efficient layer normalization
  - **RoPE**: Rotary Position Embeddings
  - **Multi-Head Attention**: Self-attention mechanism
  - **SwiGLU**: Advanced activation function
- Interactive architecture visualization
- Real-time training monitoring
- Text generation interface
- Attention pattern visualization
- Model parameter analysis

**Tech Stack:** PyTorch, Gradio, Plotly, Matplotlib

**Quick Start:**
```bash
python transformers_from_scratch/app.py
```

---

### 2. 🎨 Stable Diffusion - Text-to-Image
Generate stunning images from text descriptions using Stable Diffusion XL.

**Features:**
- Multiple quality presets (Fast, Balanced, Quality, Creative)
- Adjustable parameters:
  - Inference steps
  - Guidance scale
  - Image resolution
  - Seed control
- Batch generation
- Negative prompts
- Interactive web UI

**Tech Stack:** Diffusers, Transformers, Gradio

**Quick Start:**
```bash
python stable_diffusion/app.py
```

---

### 3. 💬 Mistral RAG - Context-Aware Q&A
Retrieval Augmented Generation system for answering questions based on custom documents.

**Features:**
- Web document scraping and indexing
- Vector similarity search with FAISS
- Context-aware question answering
- Source attribution
- Compare RAG vs direct LLM responses
- 4-bit quantized Mistral-7B for efficiency

**Tech Stack:** LangChain, FAISS, Mistral-7B, Sentence-Transformers

**Quick Start:**
```bash
python mistral_rag/app.py
```

---

### 4. 🤸 BipedalWalker RL - Reinforcement Learning
Train an AI agent to walk using PPO (Proximal Policy Optimization).

**Features:**
- PPO implementation with Stable-Baselines3
- Configurable hyperparameters
- Training progress tracking
- Model evaluation
- Save/load trained models
- Parallel environment training

**Tech Stack:** Gymnasium, Stable-Baselines3, PPO

**Quick Start:**
```bash
python rl_bipedal_walker/app.py
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for Transformers and Stable Diffusion)
- 8GB+ RAM (16GB+ recommended)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/koopatroopa787/Google-colab.git
cd Google-colab
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **For Stable Diffusion (optional - for faster generation):**
```bash
pip install xformers
```

4. **For Mistral RAG (install Playwright browsers):**
```bash
playwright install
playwright install-deps
```

---

## 🎯 Usage

### Main Hub Interface
Run the unified interface to access all projects:
```bash
python app.py
```

### Individual Projects

**Transformers:**
```python
from transformers_from_scratch.models import Llama
from transformers_from_scratch.utils import prepare_dataset, train
import torch

config = {
    'vocab_size': 65,
    'd_model': 128,
    'n_layers': 4,
    'n_heads': 8,
    'context_window': 16,
    'device': torch.device('cuda'),
    'batch_size': 32,
    'epochs': 1000,
    'log_interval': 10
}

model = Llama(config)
# ... train and generate
```

**Stable Diffusion:**
```python
from stable_diffusion.core import StableDiffusionGenerator

generator = StableDiffusionGenerator()
generator.load_model()

images = generator.generate_image(
    prompt="A beautiful landscape, 8K, detailed",
    num_inference_steps=50
)
images[0].save("output.jpg")
```

**Mistral RAG:**
```python
from mistral_rag.core import MistralRAGSystem

rag = MistralRAGSystem()
rag.load_model()
rag.index_documents(["https://example.com/doc"])
rag.setup_rag_chain()

result = rag.ask("What is the main topic?")
print(result['answer'])
```

**BipedalWalker RL:**
```python
from rl_bipedal_walker.core import BipedalWalkerTrainer

trainer = BipedalWalkerTrainer(n_envs=16)
trainer.create_model()
trainer.train(total_timesteps=1_000_000)

mean_reward, std_reward = trainer.evaluate()
trainer.save_model("walker.zip")
```

---

## 📂 Project Structure

```
Google-colab/
├── app.py                          # Main hub interface
├── requirements.txt                # Dependencies
├── README.md                       # This file
│
├── transformers_from_scratch/      # Transformer implementation
│   ├── app.py                      # Gradio interface
│   ├── models/
│   │   ├── llama.py               # Main model
│   │   └── components.py          # Building blocks
│   ├── utils/
│   │   ├── data.py                # Data loading
│   │   └── training.py            # Training utilities
│   └── visualization/
│       └── visualizer.py          # Visualization tools
│
├── stable_diffusion/               # Stable Diffusion
│   ├── app.py                      # Gradio interface
│   └── core/
│       └── generator.py           # Image generation
│
├── mistral_rag/                    # RAG system
│   ├── app.py                      # Gradio interface
│   └── core/
│       └── rag_system.py          # RAG implementation
│
└── rl_bipedal_walker/              # RL training
    ├── app.py                      # Gradio interface
    └── core/
        └── trainer.py             # PPO trainer
```

---

## 🎓 Learning Resources

### Transformers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [LLaMA Paper](https://arxiv.org/abs/2302.13971) - Modern architecture choices
- [Rotary Position Embeddings](https://arxiv.org/abs/2104.09864) - RoPE explanation

### Stable Diffusion
- [DDPM Paper](https://arxiv.org/abs/2006.11239) - Diffusion models
- [Stable Diffusion](https://arxiv.org/abs/2112.10752) - Latent diffusion

### RAG
- [RAG Paper](https://arxiv.org/abs/2005.11401) - Retrieval augmented generation
- [LangChain Docs](https://python.langchain.com/) - Framework documentation

### Reinforcement Learning
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Proximal Policy Optimization
- [Spinning Up in RL](https://spinningup.openai.com/) - OpenAI tutorial

---

## 🎨 Features & Highlights

### Interactive Visualizations
- **Transformer Architecture Diagrams**: See how data flows through the model
- **Attention Heatmaps**: Visualize what the model focuses on
- **Training Curves**: Monitor loss and performance in real-time
- **Parameter Distribution**: Understand model composition

### User-Friendly Interfaces
- **Gradio Web UIs**: No coding required to use the models
- **Progress Tracking**: See real-time updates during training/generation
- **Preset Configurations**: Quick start with optimized settings
- **Help & Documentation**: Built-in guides and examples

### Production-Ready Code
- **Modular Design**: Easy to extend and customize
- **Type Hints**: Better code clarity
- **Comprehensive Comments**: Learn as you code
- **Error Handling**: Robust and informative

---

## 💡 Tips & Best Practices

### For Transformers:
- Start with small models (d_model=128, n_layers=4) to understand the architecture
- Use GPU for training (100x faster than CPU)
- Monitor both train and validation loss
- Experiment with different architectures

### For Stable Diffusion:
- Be specific in prompts (include style, lighting, mood, colors)
- Use negative prompts to avoid unwanted elements
- Higher guidance scale = more adherence to prompt
- More steps = better quality but slower

### For Mistral RAG:
- Choose relevant, high-quality documents to index
- Use descriptive questions
- Compare RAG vs non-RAG to see the difference
- Experiment with chunk sizes for better retrieval

### For BipedalWalker RL:
- Start with 100K-1M timesteps for quick tests
- Use parallel environments for faster training
- Good walking typically appears after 1-5M steps
- Save models periodically

---

## 🖥️ Hardware Requirements

| Project | Min RAM | Recommended RAM | GPU |
|---------|---------|-----------------|-----|
| Transformers | 4GB | 8GB+ | Recommended |
| Stable Diffusion | 8GB | 16GB+ | Required |
| Mistral RAG | 8GB | 16GB+ | Recommended |
| BipedalWalker RL | 2GB | 4GB+ | Optional |

**GPU Recommendations:**
- NVIDIA GPUs with CUDA support
- 6GB+ VRAM for Stable Diffusion
- 4GB+ VRAM for Transformers and RAG

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

---

## 📄 License

This project is for educational purposes. Please respect the licenses of the underlying models and libraries.

---

## 🙏 Acknowledgments

- **PyTorch Team** - Deep learning framework
- **HuggingFace** - Transformers and Diffusers libraries
- **OpenAI** - Research and inspiration
- **Stability AI** - Stable Diffusion
- **Mistral AI** - Mistral models
- **LangChain** - RAG framework
- **Stable-Baselines3** - RL implementations

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Happy Learning! 🚀**
