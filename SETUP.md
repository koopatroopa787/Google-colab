# Setup and Testing Guide

This guide will help you set up and test all projects in this repository.

## 🚀 Quick Setup (Automated)

We provide automated setup scripts that will:
1. Create a virtual environment
2. Install all dependencies
3. Test each project
4. Generate a report

### Windows

Run the batch file:
```cmd
setup_and_test.bat
```

Or run the Python script directly:
```cmd
python setup_and_test.py
```

### Linux/macOS

Run the shell script:
```bash
./setup_and_test.sh
```

Or run the Python script directly:
```bash
python3 setup_and_test.py
```

---

## 🔧 Manual Setup

If you prefer to set up manually:

### 1. Create Virtual Environment

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install Additional Tools (Optional)

For better performance:
```bash
pip install xformers  # For faster Stable Diffusion (requires CUDA)
```

For Mistral RAG (web scraping):
```bash
playwright install
playwright install-deps
```

---

## 🧪 Testing Individual Projects

You can test each project individually:

```bash
python test_individual_project.py transformers
python test_individual_project.py stable_diffusion
python test_individual_project.py mistral_rag
python test_individual_project.py rl
```

---

## 📋 Project-Specific Setup

### Transformers from Scratch

**Requirements:**
- PyTorch
- NumPy, Pandas, Matplotlib
- Gradio, Plotly

**Test:**
```bash
cd transformers_from_scratch
python -c "from models.llama import Llama; print('✓ Works')"
```

**Run:**
```bash
python transformers_from_scratch/app.py
```

---

### Stable Diffusion

**Requirements:**
- PyTorch with CUDA (recommended)
- Diffusers, Transformers
- At least 8GB RAM (16GB+ recommended)
- GPU with 6GB+ VRAM (for faster generation)

**Test:**
```bash
cd stable_diffusion
python -c "from core.generator import StableDiffusionGenerator; print('✓ Works')"
```

**Run:**
```bash
python stable_diffusion/app.py
```

---

### Mistral RAG

**Requirements:**
- PyTorch
- LangChain packages (langchain, langchain-community, langchain-core)
- Sentence Transformers
- FAISS
- Playwright (for web scraping)

**Additional Setup:**
```bash
playwright install
playwright install-deps
```

**Test:**
```bash
cd mistral_rag
python -c "from core.rag_system import MistralRAGSystem; print('✓ Works')"
```

**Run:**
```bash
python mistral_rag/app.py
```

---

### BipedalWalker RL

**Requirements:**
- Gymnasium
- Stable-Baselines3
- PyTorch

**Test:**
```bash
cd rl_bipedal_walker
python -c "from core.trainer import BipedalWalkerTrainer; print('✓ Works')"
```

**Run:**
```bash
python rl_bipedal_walker/app.py
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**

If you get `ModuleNotFoundError`:
```bash
pip install --upgrade -r requirements.txt
```

**2. LangChain Import Errors**

Install the new LangChain packages:
```bash
pip install langchain langchain-community langchain-core langchain-text-splitters
```

**3. CUDA Not Available**

If PyTorch doesn't detect your GPU:
- Reinstall PyTorch with CUDA support: https://pytorch.org/get-started/locally/
- Check CUDA installation: `nvidia-smi`

**4. Memory Errors**

For Stable Diffusion and Mistral RAG:
- Close other applications
- Use smaller batch sizes
- Use CPU instead of GPU (slower but works)

**5. Playwright Errors**

Install browsers:
```bash
playwright install
playwright install-deps
```

---

## 📊 Test Report

After running `setup_and_test.py`, you'll see a report like:

```
============================================================
                        Test Report
============================================================

Total Projects: 4
Passed: 4
Failed: 0

  Transformers from Scratch: PASSED
  Stable Diffusion: PASSED
  Mistral RAG: PASSED
  BipedalWalker RL: PASSED

✓ All projects are working correctly! 🎉
```

---

## 🎯 Next Steps

Once setup is complete:

1. **Run the main hub:**
   ```bash
   python app.py
   ```

2. **Or run individual projects:**
   ```bash
   python transformers_from_scratch/app.py
   python stable_diffusion/app.py
   python mistral_rag/app.py
   python rl_bipedal_walker/app.py
   ```

3. **Access the web interface:**
   - Open browser to: http://127.0.0.1:7860
   - Or use the link shown in terminal

---

## 💡 Tips

- **Virtual Environment**: Always activate before working:
  - Windows: `venv\Scripts\activate`
  - Linux/macOS: `source venv/bin/activate`

- **GPU Usage**:
  - Transformers: Optional but 100x faster
  - Stable Diffusion: Highly recommended
  - Mistral RAG: Helpful for faster inference
  - RL: Optional

- **Memory Management**:
  - Close browsers and other apps when running heavy models
  - Use CPU mode if GPU memory is limited
  - Reduce batch sizes in configs

---

## 📞 Support

If you encounter issues:

1. Check this SETUP.md file
2. Review error messages carefully
3. Try running `setup_and_test.py` again
4. Check requirements.txt versions
5. Open an issue on GitHub

---

**Happy Coding! 🚀**
