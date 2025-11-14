# Box2D Installation Guide for Windows

Box2D is required for the BipedalWalker environment but is notoriously difficult to install on Windows. Here are **4 working solutions**, from easiest to hardest.

---

## ✅ Solution 1: Pre-built Wheel (EASIEST - RECOMMENDED)

This is the **fastest and most reliable** method for Windows.

### Step 1: Download the pre-built wheel

**For Python 3.11 (64-bit):**
```
https://download.lfd.uci.edu/pythonlibs/archived/Box2D-2.3.10-cp311-cp311-win_amd64.whl
```

**For Python 3.10 (64-bit):**
```
https://download.lfd.uci.edu/pythonlibs/archived/Box2D-2.3.10-cp310-cp310-win_amd64.whl
```

**For Python 3.9 (64-bit):**
```
https://download.lfd.uci.edu/pythonlibs/archived/Box2D-2.3.10-cp39-cp39-win_amd64.whl
```

**Check your Python version:**
```bash
python --version
```

### Step 2: Install the downloaded wheel

Navigate to your downloads folder and run:
```bash
cd Downloads
pip install Box2D-2.3.10-cp311-cp311-win_amd64.whl
```

### Step 3: Install gymnasium with box2d support

```bash
pip install gymnasium[box2d]
```

### Step 4: Verify installation

```bash
python -c "import Box2D; print('Box2D installed successfully!')"
```

---

## ✅ Solution 2: Use Conda (ALTERNATIVE)

Conda handles the compilation automatically.

### Step 1: Install Conda

Download from: https://docs.conda.io/en/latest/miniconda.html

### Step 2: Create conda environment

```bash
conda create -n ai_ml python=3.11
conda activate ai_ml
```

### Step 3: Install Box2D

```bash
conda install -c conda-forge box2d-py
```

### Step 4: Install other dependencies

```bash
pip install -r requirements.txt
```

---

## ✅ Solution 3: Use CartPole Instead (NO BOX2D NEEDED)

If you can't install Box2D, use a simpler environment that doesn't require it.

### Modify the code to use CartPole:

When creating the trainer in the Gradio UI or code:

```python
from rl_bipedal_walker.core import BipedalWalkerTrainer

trainer = BipedalWalkerTrainer(n_envs=16)
trainer.create_environment('CartPole-v1')  # Use CartPole instead
trainer.create_model()
trainer.train(total_timesteps=100_000)
```

**CartPole-v1** is much simpler but still demonstrates RL concepts!

---

## ✅ Solution 4: Install SWIG and Build from Source (ADVANCED)

Only use this if you want to build from source.

### Step 1: Install SWIG

**Option A: Using Chocolatey (Recommended)**
```bash
# Install Chocolatey first: https://chocolatey.org/install
choco install swig
```

**Option B: Manual Installation**
1. Download SWIG from: http://www.swig.org/download.html
2. Extract to `C:\swigwin-4.1.1`
3. Add to PATH: `C:\swigwin-4.1.1`

### Step 2: Install Visual C++ Build Tools

Download and install:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

Select "Desktop development with C++"

### Step 3: Install Box2D

```bash
pip install box2d-py
pip install gymnasium[box2d]
```

---

## 🧪 Test Your Installation

After installation, test with:

```bash
python test_box2d.py
```

Or manually:

```python
import gymnasium as gym
import Box2D

# Test environment creation
env = gym.make('BipedalWalker-v3')
obs, info = env.reset()
print(f"✓ BipedalWalker works! Observation shape: {obs.shape}")
env.close()
```

---

## 🚀 Quick Commands for Your Situation

Based on your error, here's what to do:

### For Python 3.11 (your version):

1. **Download the wheel:**
   Open browser and download from:
   ```
   https://download.lfd.uci.edu/pythonlibs/archived/Box2D-2.3.10-cp311-cp311-win_amd64.whl
   ```

2. **Install it:**
   ```bash
   cd E:\vs_code\Google-colab
   pip install C:\Users\yashk\Downloads\Box2D-2.3.10-cp311-cp311-win_amd64.whl
   ```

3. **Install gymnasium[box2d]:**
   ```bash
   pip install gymnasium[box2d]
   ```

4. **Test it:**
   ```bash
   python -c "import Box2D; print('Success!')"
   ```

---

## ❌ Common Errors & Fixes

### Error: "No matching distribution found"
- Make sure you downloaded the correct wheel for your Python version
- Check with `python --version`

### Error: "is not a supported wheel on this platform"
- You downloaded wrong Python version or architecture
- Download the correct one (cp311 for Python 3.11, cp310 for 3.10, etc.)

### Error: "SWIG failed with exit code 1"
- Don't build from source on Windows - use pre-built wheel instead!

---

## 📞 Still Not Working?

If all else fails:

1. **Use CartPole** (no Box2D needed):
   ```python
   trainer.create_environment('CartPole-v1')
   ```

2. **Use Docker** (all dependencies pre-installed):
   ```bash
   docker run -it python:3.11
   pip install gymnasium[box2d]
   ```

3. **Use Google Colab** (cloud-based, GPU included):
   - Upload the notebook versions
   - Everything works out of the box

---

## 📚 Additional Resources

- **Christoph Gohlke's Wheels**: https://www.lfd.uci.edu/~gohlke/pythonlibs/
- **Gymnasium Docs**: https://gymnasium.farama.org/
- **Box2D Python**: https://github.com/pybox2d/pybox2d
- **SWIG Download**: http://www.swig.org/download.html

---

**Last Updated:** 2024-11-13

**Tested On:** Windows 10/11, Python 3.9-3.11
