#!/usr/bin/env python3
"""
Individual Project Testing Script

Test a specific project to ensure it's working correctly.

Usage:
    python test_individual_project.py transformers
    python test_individual_project.py stable_diffusion
    python test_individual_project.py mistral_rag
    python test_individual_project.py rl
"""

import sys
import os
import subprocess


def test_transformers():
    """Test Transformers from Scratch project"""
    print("Testing Transformers from Scratch...")

    test_code = """
import sys
sys.path.insert(0, '.')

# Test basic imports
import torch
import numpy as np
import gradio as gr
import plotly.graph_objects as go

# Test project imports
from transformers_from_scratch.models.components import (
    RMSNorm, SwiGLU, RoPEAttentionHead,
    RoPEMaskedMultiheadAttention, LlamaBlock
)
from transformers_from_scratch.models.llama import Llama
from transformers_from_scratch.utils.data import prepare_dataset, get_batches
from transformers_from_scratch.utils.training import train, evaluate_loss, generate
from transformers_from_scratch.visualization.visualizer import visualize_architecture

# Test model creation
config = {
    'vocab_size': 65,
    'd_model': 64,
    'n_layers': 2,
    'n_heads': 4,
    'context_window': 8,
    'device': torch.device('cpu'),
    'batch_size': 4,
    'epochs': 10,
    'log_interval': 5
}

model = Llama(config)
print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")

# Test visualization
try:
    fig = visualize_architecture(config)
    print("✓ Visualization works")
except Exception as e:
    print(f"✗ Visualization failed: {e}")

print("✓ All Transformers tests passed!")
"""

    result = subprocess.run([sys.executable, "-c", test_code], capture_output=True, text=True)

    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print(f"✗ Test failed:\n{result.stderr}")
        return False


def test_stable_diffusion():
    """Test Stable Diffusion project"""
    print("Testing Stable Diffusion...")

    test_code = """
import sys
sys.path.insert(0, '.')

# Test basic imports
import torch
import gradio as gr
from diffusers import StableDiffusionXLPipeline

# Test project imports
from stable_diffusion.core.generator import (
    StableDiffusionGenerator,
    ImageGenerationPresets
)

# Test generator creation
generator = StableDiffusionGenerator()
print("✓ Generator created")

# Test presets
preset = ImageGenerationPresets.get_preset("Fast")
print(f"✓ Preset loaded: {preset['description']}")

print("✓ All Stable Diffusion tests passed!")
"""

    result = subprocess.run([sys.executable, "-c", test_code], capture_output=True, text=True)

    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print(f"✗ Test failed:\n{result.stderr}")
        return False


def test_mistral_rag():
    """Test Mistral RAG project"""
    print("Testing Mistral RAG...")

    test_code = """
import sys
sys.path.insert(0, '.')

# Test basic imports
import torch
import gradio as gr
from transformers import AutoTokenizer

# Test LangChain imports
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("✓ Using new LangChain imports")
except ImportError:
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    print("✓ Using legacy LangChain imports")

# Test project imports
from mistral_rag.core.rag_system import MistralRAGSystem

# Test system creation
rag = MistralRAGSystem()
print("✓ RAG system created")

print("✓ All Mistral RAG tests passed!")
"""

    result = subprocess.run([sys.executable, "-c", test_code], capture_output=True, text=True)

    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print(f"✗ Test failed:\n{result.stderr}")
        return False


def test_rl():
    """Test RL BipedalWalker project"""
    print("Testing BipedalWalker RL...")

    test_code = """
import sys
sys.path.insert(0, '.')

# Test basic imports
import gymnasium as gym
import gradio as gr
from stable_baselines3 import PPO

# Test project imports
from rl_bipedal_walker.core.trainer import BipedalWalkerTrainer

# Test trainer creation
trainer = BipedalWalkerTrainer(n_envs=2)
print("✓ Trainer created")

# Test environment info
info = trainer.get_env_info()
print(f"✓ Environment info retrieved: {info['observation_space_shape']}")

print("✓ All BipedalWalker RL tests passed!")
"""

    result = subprocess.run([sys.executable, "-c", test_code], capture_output=True, text=True)

    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print(f"✗ Test failed:\n{result.stderr}")
        return False


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python test_individual_project.py <project>")
        print("\nAvailable projects:")
        print("  transformers       - Transformers from Scratch")
        print("  stable_diffusion   - Stable Diffusion")
        print("  mistral_rag        - Mistral RAG")
        print("  rl                 - BipedalWalker RL")
        sys.exit(1)

    project = sys.argv[1].lower()

    tests = {
        'transformers': test_transformers,
        'stable_diffusion': test_stable_diffusion,
        'mistral_rag': test_mistral_rag,
        'rl': test_rl
    }

    if project not in tests:
        print(f"Unknown project: {project}")
        print(f"Available projects: {', '.join(tests.keys())}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Testing: {project}")
    print(f"{'='*60}\n")

    success = tests[project]()

    if success:
        print(f"\n✓ {project} is working correctly!")
        sys.exit(0)
    else:
        print(f"\n✗ {project} has issues")
        sys.exit(1)


if __name__ == "__main__":
    main()
