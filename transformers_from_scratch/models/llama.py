"""
LLaMA-style Transformer Model

This module implements a complete transformer model with modern architectural choices:
- Token embeddings
- Multiple transformer blocks (LlamaBlock)
- Final feedforward network for predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .components import LlamaBlock, SwiGLU


class Llama(nn.Module):
    """
    Complete LLaMA-style transformer model

    Architecture:
    1. Token Embedding Layer
    2. Stack of N Transformer Blocks (LlamaBlock)
    3. Final Feedforward Network with SwiGLU
    4. Output projection to vocabulary

    This model can be trained for next-token prediction (language modeling).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embeddings: converts token IDs to dense vectors
        self.embeddings = nn.Embedding(config['vocab_size'], config['d_model'])

        # Stack of transformer blocks
        self.llama_blocks = nn.Sequential(
            OrderedDict([(f"llama_{i}", LlamaBlock(config)) for i in range(config['n_layers'])])
        )

        # Final feedforward network for output projection
        self.ffn = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

        # Device configuration
        self.device = config['device']
        self.to(self.device)

        # Print model statistics
        total_params = sum([m.numel() for m in self.parameters()])
        print(f"✓ Model initialized with {total_params:,} parameters")
        print(f"  - Layers: {config['n_layers']}")
        print(f"  - Heads: {config['n_heads']}")
        print(f"  - Dimension: {config['d_model']}")
        print(f"  - Vocab size: {config['vocab_size']}")

    def forward(self, idx, targets=None):
        """
        Forward pass through the model

        Args:
            idx: Token IDs (batch_size, seq_len)
            targets: Target token IDs for training (batch_size, seq_len)

        Returns:
            If targets provided: (logits, loss)
            If targets not provided: logits only
        """
        # Convert token IDs to embeddings
        x = self.embeddings(idx.to(self.device))

        # Pass through all transformer blocks
        x = self.llama_blocks(x)

        # Final projection to vocabulary
        logits = self.ffn(x)

        if targets is None:
            return logits
        else:
            # Calculate cross-entropy loss for training
            targets = targets.to(self.device)
            loss = F.cross_entropy(
                logits.view(-1, self.config['vocab_size']),
                targets.view(-1)
            )
            return logits, loss

    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), subtract position and token embeddings.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embeddings.weight.numel()
        return n_params


class SimpleBrokenModel(nn.Module):
    """
    Simple baseline model without advanced features

    This is a basic neural network for comparison purposes.
    It doesn't use attention or any advanced transformer components.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Simple embedding layer
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])

        # Basic feedforward layers
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

        self.device = config['device']
        print(f"Simple model parameters: {sum([m.numel() for m in self.parameters()]):,}")

    def forward(self, idx, targets=None):
        # Embed tokens
        x = self.embedding(idx)

        # Simple feedforward
        logits = self.linear(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config['vocab_size']),
                targets.view(-1)
            )
            return logits, loss
        else:
            return logits
