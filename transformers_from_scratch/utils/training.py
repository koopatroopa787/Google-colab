"""
Training utilities for transformer models

This module provides:
- Training loop with evaluation
- Loss evaluation
- Text generation
- Learning rate scheduling
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import pandas as pd
from matplotlib import pyplot as plt


@torch.no_grad()
def evaluate_loss(model, dataset, get_batches_fn, config):
    """
    Evaluate model loss on train and validation sets

    Args:
        model: The model to evaluate
        dataset: Full dataset tensor
        get_batches_fn: Function to generate batches
        config: Configuration dictionary

    Returns:
        Dictionary with train and validation losses
    """
    out = {}
    model.eval()  # Set to evaluation mode

    for split in ["train", "val"]:
        losses = []

        # Evaluate on multiple batches for stable estimate
        for _ in range(10):
            xb, yb = get_batches_fn(dataset, split, config['batch_size'], config['context_window'])
            xb, yb = xb.to(config['device']), yb.to(config['device'])

            # Forward pass
            _, loss = model(xb, yb)
            losses.append(loss.item())

        # Average loss for this split
        out[split] = np.mean(losses)

    model.train()  # Set back to training mode
    return out


def train(model, optimizer, dataset, get_batches_fn, scheduler=None, config=None, print_logs=False):
    """
    Train the transformer model

    Args:
        model: Model to train
        optimizer: Optimizer (e.g., Adam)
        dataset: Training dataset
        get_batches_fn: Function to generate batches
        scheduler: Optional learning rate scheduler
        config: Training configuration
        print_logs: Whether to print progress

    Returns:
        DataFrame with training history
    """
    losses = []
    start_time = time.time()

    model.to(model.device)
    print(f"\n{'='*60}")
    print(f"Starting training for {config['epochs']} epochs")
    print(f"{'='*60}\n")

    for epoch in range(config['epochs']):
        # Zero gradients
        optimizer.zero_grad()

        # Get batch
        xs, ys = get_batches_fn(dataset, 'train', config['batch_size'], config['context_window'])
        xs, ys = xs.to(model.device), ys.to(model.device)

        # Forward pass
        logits, loss = model(xs, targets=ys)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update learning rate if scheduler provided
        if scheduler:
            scheduler.step()

        # Log progress
        if epoch % config['log_interval'] == 0:
            batch_time = time.time() - start_time
            x = evaluate_loss(model, dataset, get_batches_fn, config)
            losses.append(x)

            if print_logs:
                lr_info = f" | LR: {scheduler.get_last_lr()[0]:.6f}" if scheduler else ""
                print(f"Epoch {epoch:4d} | Train: {x['train']:.4f} | Val: {x['val']:.4f} | "
                      f"Time: {batch_time:.2f}s{lr_info}")

            start_time = time.time()

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Final validation loss: {losses[-1]['val']:.4f}")
    print(f"{'='*60}\n")

    return pd.DataFrame(losses)


def generate(model, config, max_new_tokens=100, temperature=1.0, num_samples=1, prompt=None):
    """
    Generate text using the trained model

    Args:
        model: Trained model
        config: Configuration dictionary
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        num_samples: Number of samples to generate
        prompt: Optional text prompt to start generation

    Returns:
        List of generated text samples
    """
    model.eval()

    if prompt is not None:
        # TODO: Encode prompt if provided
        idx = torch.zeros(num_samples, 1).long().to(config['device'])
    else:
        # Start with zeros (will be replaced)
        idx = torch.zeros(num_samples, 1).long().to(config['device'])

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get predictions (use only last context_window tokens)
            idx_cond = idx[:, -config['context_window']:]
            logits = model(idx_cond)

            # Focus on last time step
            logits = logits[:, -1, :] / temperature

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=-1)

    model.train()
    return idx


def plot_training_history(history_df, save_path=None):
    """
    Plot training and validation loss curves

    Args:
        history_df: DataFrame with training history
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history_df.index * 10, history_df['train'], label='Train Loss', linewidth=2)
    plt.plot(history_df.index * 10, history_df['val'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return plt.gcf()


def get_model_info(model):
    """
    Get detailed information about the model

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'layers': {}
    }

    # Get layer information
    for name, param in model.named_parameters():
        info['layers'][name] = {
            'shape': list(param.shape),
            'parameters': param.numel()
        }

    return info
