"""
Transformer Components: Building blocks for the transformer architecture
This module contains implementations of:
- RMSNorm: Root Mean Square Layer Normalization
- RoPE: Rotary Position Embeddings
- SwiGLU: Swish-Gated Linear Unit activation
- Attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    RMSNorm is a simpler and more efficient alternative to LayerNorm.
    It normalizes the inputs using the root mean square statistic.
    Used in LLaMA and other modern transformers for better performance.
    """
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Normalized tensor of the same shape
        """
        # Calculate RMS: sqrt(mean(x^2))
        ff_rms = torch.linalg.norm(x, dim=(1, 2)) * x[0].numel() ** -0.5
        # Normalize and scale
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)
        return self.scale[:x.shape[1], :].unsqueeze(0) * raw


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) activation function

    SwiGLU is used as an activation function in the feedforward layers.
    Paper: https://arxiv.org/pdf/2002.05202v1.pdf

    Formula: SwiGLU(x) = (x * W_gate) * σ(β * (x * W_gate)) ⊙ (x * W)
    where σ is sigmoid and ⊙ is element-wise multiplication
    """
    def __init__(self, size):
        super().__init__()
        self.linear_gate = nn.Linear(size, size)  # Gate transformation
        self.linear = nn.Linear(size, size)  # Main transformation
        self.beta = nn.Parameter(torch.ones(1))  # Learnable parameter

    def forward(self, x):
        """
        Args:
            x: Input tensor
        Returns:
            Activated tensor using SwiGLU
        """
        # Swish gate: x * sigmoid(beta * x)
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        # Element-wise multiplication with main branch
        out = swish_gate * self.linear(x)
        return out


def get_rotary_matrix(context_window, embedding_dim):
    """
    Generate Rotary Position Embedding (RoPE) matrix

    RoPE encodes position information by rotating embeddings in a way that
    preserves relative position information between tokens.

    Args:
        context_window: Maximum sequence length
        embedding_dim: Dimension of embeddings (must be even)
    Returns:
        Rotation matrix of shape (context_window, embedding_dim, embedding_dim)
    """
    R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)

    for position in range(context_window):
        for i in range(embedding_dim // 2):
            # Calculate rotation angle based on position and dimension
            theta = 10000. ** (-2. * (i - 1) / embedding_dim)
            m_theta = position * theta

            # Fill rotation matrix with sine and cosine values
            R[position, 2 * i, 2 * i] = np.cos(m_theta)
            R[position, 2 * i, 2 * i + 1] = -np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i] = np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i + 1] = np.cos(m_theta)

    return R


class RoPEAttentionHead(nn.Module):
    """
    Single attention head with Rotary Position Embeddings

    This implements scaled dot-product attention with RoPE for position encoding.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Linear transformations for Q, K, V
        self.w_q = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.w_k = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.w_v = nn.Linear(config['d_model'], config['d_model'], bias=False)

        # Rotary position embedding matrix
        self.R = get_rotary_matrix(config['context_window'], config['d_model']).to(config['device'])

    def forward(self, x, return_attn_weights=False):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            return_attn_weights: Whether to return attention weights for visualization
        Returns:
            Attention output and optionally attention weights
        """
        b, m, d = x.shape  # batch size, sequence length, dimension

        # Compute Q, K, V
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Apply rotary position embeddings to Q and K
        q_rotated = (torch.bmm(q.transpose(0, 1), self.R[:m])).transpose(0, 1)
        k_rotated = (torch.bmm(k.transpose(0, 1), self.R[:m])).transpose(0, 1)

        # Scaled dot-product attention with causal mask
        activations = F.scaled_dot_product_attention(
            q_rotated, k_rotated, v, dropout_p=0.1, is_causal=True
        )

        if return_attn_weights:
            # Calculate attention weights for visualization
            attn_mask = torch.tril(torch.ones((m, m)), diagonal=0)
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1, 2)) / np.sqrt(d) + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            return activations, attn_weights

        return activations


class RoPEMaskedMultiheadAttention(nn.Module):
    """
    Multi-head attention with RoPE

    Combines multiple attention heads and projects the output.
    Each head can attend to different aspects of the input.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create multiple attention heads
        self.heads = nn.ModuleList([
            RoPEAttentionHead(config) for _ in range(config['n_heads'])
        ])

        # Output projection
        self.linear = nn.Linear(config['n_heads'] * config['d_model'], config['d_model'])
        self.dropout = nn.Dropout(.1)

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        Returns:
            Multi-head attention output
        """
        # Apply all attention heads
        heads = [h(x) for h in self.heads]

        # Concatenate heads and project
        x = torch.cat(heads, dim=-1)
        x = self.linear(x)
        x = self.dropout(x)

        return x


class LlamaBlock(nn.Module):
    """
    Transformer block with RMSNorm, RoPE attention, and SwiGLU

    This is a single transformer layer that combines:
    - RMSNorm for normalization
    - Multi-head attention with RoPE
    - Feedforward network with SwiGLU activation
    - Residual connections
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # RMSNorm layers
        self.rms = RMSNorm((config['context_window'], config['d_model']))

        # Multi-head attention with RoPE
        self.attention = RoPEMaskedMultiheadAttention(config)

        # Feedforward network with SwiGLU
        self.feedforward = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        Returns:
            Transformed tensor
        """
        # Attention block with residual connection and pre-normalization
        x = x + self.attention(self.rms(x))

        # Feedforward block with residual connection and pre-normalization
        x = x + self.feedforward(self.rms(x))

        return x
