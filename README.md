# 🚀 AI/ML Projects Collection

A hands-on playground of four self-contained AI/ML demos. Launch the unified Gradio hub or jump directly into each project to learn how the models work under the hood.

## Why this repo?
- One entry point (`python app.py`) to explore all demos
- Clean, modular Python code that is easy to extend
- Gradio interfaces plus programmatic APIs for every project
- Ready for quick experiments in local environments or Google Colab

## ⚡ Quickstart (local or Colab)
1. **Clone & enter the repo**
   ```bash
   git clone https://github.com/koopatroopa787/Google-colab.git
   cd Google-colab
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Optional extras**
   - Stable Diffusion speed-up: `pip install xformers`
   - Playwright browsers for RAG scraping: `playwright install && playwright install-deps`
4. **Launch the hub**
   ```bash
   python app.py
   ```
   A Gradio interface will open with tabs for every project.
5. **Run projects individually**
   ```bash
   python transformers_from_scratch/app.py
   python stable_diffusion/app.py
   python mistral_rag/app.py
   python rl_bipedal_walker/app.py
   ```

> **Hardware:** A CUDA GPU is recommended for transformers and RAG, and required for comfortable Stable Diffusion runs. CPU-only works for light experiments.

## 📋 Project Overview

| Project | What it does | Key technologies | UI entry |
|---------|--------------|------------------|----------|
| 🤖 Transformer from Scratch | Build and train LLaMA-style language models | PyTorch, RMSNorm, RoPE, SwiGLU, Plotly | `transformers_from_scratch/app.py` |
| 🎨 Stable Diffusion | Text-to-image generation with SDXL | Diffusers, Transformers, Gradio | `stable_diffusion/app.py` |
| 💬 Mistral RAG | Context-aware Q&A over custom docs | LangChain, FAISS, Mistral-7B | `mistral_rag/app.py` |
| 🤸 BipedalWalker RL | Train a walking agent with PPO | Gymnasium, Stable-Baselines3 | `rl_bipedal_walker/app.py` |

---

## 🧭 Project Walkthroughs

### 1) 🤖 Transformer from Scratch
Educational, from-scratch implementation of a LLaMA-style transformer with visualizations.

**Highlights**
- RMSNorm, RoPE, multi-head attention, SwiGLU blocks
- Attention and architecture visualizers
- Real-time training curves and parameter analysis

**Run it**
```bash
python transformers_from_scratch/app.py
```

**Programmatic snippet**
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

dataset, vocab, encode, decode = prepare_dataset("tinyshakespeare.txt")
model = Llama(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

history = train(model, optimizer, dataset, get_batches, config=config, print_logs=True)
generated = generate(model, config, max_new_tokens=100)
print(decode(generated[0].tolist()))
```

---

### 2) 🎨 Stable Diffusion - Text-to-Image
Generate high-quality images from text using Stable Diffusion XL (SSD-1B).

**Highlights**
- Quality presets (Fast, Balanced, Quality, Creative)
- Adjustable guidance, steps, resolution, seeds
- Negative prompt defaults to avoid common artifacts

**Run it**
```bash
python stable_diffusion/app.py
```

**Programmatic snippet**
```python
from stable_diffusion.core import StableDiffusionGenerator, ImageGenerationPresets

generator = StableDiffusionGenerator()
generator.load_model()

preset = ImageGenerationPresets.get_preset("Balanced")
images = generator.generate_image(
    prompt="A cinematic sunset over the ocean, 8K",
    num_inference_steps=preset["num_inference_steps"],
    guidance_scale=preset["guidance_scale"],
    num_images=1,
)

images[0].save("sunset.jpg")
```

---

### 3) 💬 Mistral RAG - Context-Aware Q&A
Retrieval Augmented Generation pipeline for answering questions using your documents.

**Highlights**
- Web/document scraping and indexing
- FAISS vector search with source attribution
- Side-by-side: RAG vs direct LLM responses

**Run it**
```bash
python mistral_rag/app.py
```

**Programmatic snippet**
```python
from mistral_rag.core import MistralRAGSystem

rag = MistralRAGSystem()
rag.load_model()
rag.index_documents(["https://example.com/doc1", "https://example.com/doc2"])
rag.setup_rag_chain()

response = rag.ask("What is the main idea?")
print(response["answer"])
print("Sources:", response["context"])
```

---

### 4) 🤸 BipedalWalker RL - Reinforcement Learning
Train a walking agent with PPO in the Gymnasium BipedalWalker-v3 environment.

**Highlights**
- Vectorized environments for faster experience collection
- Configurable PPO hyperparameters
- Evaluation, video frame sampling, and model save/load helpers

**Run it**
```bash
python rl_bipedal_walker/app.py
```

**Programmatic snippet**
```python
from rl_bipedal_walker.core import BipedalWalkerTrainer

trainer = BipedalWalkerTrainer(n_envs=4)
trainer.create_model()
trainer.train(total_timesteps=200_000)

mean_reward, std_reward = trainer.evaluate()
print(f"Performance: {mean_reward:.2f} +/- {std_reward:.2f}")
trainer.save_model("walker.zip")
```

---

## 🛠️ Testing & Verification
- Quick hub smoke test: launch `python app.py` and open each tab.
- Verify Box2D for RL: `python test_box2d.py`
- Targeted import checks: `python test_individual_project.py <transformers|stable_diffusion|mistral_rag|rl>`

---

## 📂 Project Structure

```
Google-colab/
├── app.py                          # Main hub interface
├── requirements.txt                # Dependencies
├── README.md                       # Documentation
│
├── transformers_from_scratch/      # Transformer implementation
│   ├── app.py                      # Gradio UI
│   ├── models/                     # Model + components
│   ├── utils/                      # Data + training utilities
│   └── visualization/              # Plotly/Matplotlib helpers
│
├── stable_diffusion/               # Stable Diffusion demo
│   └── core/                       # Generator + presets
│
├── mistral_rag/                    # RAG system
│   └── core/                       # RAG pipeline
│
└── rl_bipedal_walker/              # RL training
    └── core/                       # PPO trainer
```

---

## 🎓 Learning Resources

**Transformers**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LLaMA Paper](https://arxiv.org/abs/2302.13971)
- [Rotary Position Embeddings](https://arxiv.org/abs/2104.09864)

**Stable Diffusion**
- [DDPM Paper](https://arxiv.org/abs/2006.11239)
- [Stable Diffusion](https://arxiv.org/abs/2112.10752)

**RAG**
- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [LangChain Docs](https://python.langchain.com/)

**Reinforcement Learning**
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Spinning Up in RL](https://spinningup.openai.com/)

---

## 💡 Tips & Best Practices

**Transformers**
- Start with smaller configs (d_model=128, n_layers=4) to validate the pipeline
- Monitor both training and validation loss; enable GPU for meaningful speedups

**Stable Diffusion**
- Be specific in prompts (style, lighting, mood); negative prompts reduce artifacts
- More steps improve quality but slow generation; tune guidance for adherence

**Mistral RAG**
- Index focused, high-quality sources; experiment with chunking strategies
- Compare RAG vs non-RAG answers to validate retrieval quality

**BipedalWalker RL**
- Begin with shorter runs (100K–200K steps) before long training
- Save checkpoints periodically and evaluate with deterministic policies

---

## 🖥️ Hardware Requirements

| Project | Min RAM | Recommended RAM | GPU |
|---------|---------|-----------------|-----|
| Transformers | 4GB | 8GB+ | Recommended |
| Stable Diffusion | 8GB | 16GB+ | Required |
| Mistral RAG | 8GB | 16GB+ | Recommended |
| BipedalWalker RL | 2GB | 4GB+ | Optional |

**GPU recommendations**
- NVIDIA GPUs with CUDA support
- 6GB+ VRAM for Stable Diffusion
- 4GB+ VRAM for Transformers and RAG

---

## 🤝 Contributing
Contributions are welcome! Feel free to open issues, suggest features, or submit pull requests—documentation improvements are always appreciated.

## 📄 License
Educational use only. Please respect the licenses of the underlying models and libraries.

## 🙏 Acknowledgments
- **PyTorch Team** — Deep learning framework
- **HuggingFace** — Transformers and Diffusers libraries
- **OpenAI** — Research and inspiration
- **Stability AI** — Stable Diffusion
- **Mistral AI** — Mistral models
- **LangChain** — RAG framework
- **Stable-Baselines3** — RL implementations

## 📧 Contact
For questions or feedback, please open an issue on GitHub.

**Happy Learning! 🚀**
