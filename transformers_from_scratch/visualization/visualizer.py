"""
Visualization utilities for transformer models

This module provides interactive visualizations for:
- Model architecture
- Attention patterns
- Training progress
- Token embeddings
- Layer activations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def visualize_architecture(config):
    """
    Create a visual diagram of the transformer architecture

    Args:
        config: Model configuration dictionary

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.axis('off')

    # Title
    ax.text(5, 14, 'LLaMA-Style Transformer Architecture',
            ha='center', fontsize=16, fontweight='bold')

    # Color scheme
    colors = {
        'embedding': '#FF6B6B',
        'attention': '#4ECDC4',
        'ffn': '#45B7D1',
        'norm': '#FFA07A',
        'output': '#98D8C8'
    }

    y_pos = 12

    # Input Embedding
    rect = FancyBboxPatch((2, y_pos), 6, 0.8,
                          boxstyle="round,pad=0.1",
                          facecolor=colors['embedding'],
                          edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, y_pos + 0.4, 'Token Embeddings', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(8.5, y_pos + 0.4, f"[{config.get('vocab_size', 'V')}, {config.get('d_model', 'D')}]",
            ha='left', va='center', fontsize=9, style='italic')

    y_pos -= 1.5

    # Transformer Blocks
    n_layers = config.get('n_layers', 4)
    ax.text(5, y_pos + 0.5, f'{n_layers} × Transformer Blocks',
            ha='center', fontsize=12, fontweight='bold')

    for layer in range(min(2, n_layers)):  # Show first 2 layers
        y_pos -= 1

        # RMSNorm
        rect = FancyBboxPatch((2.5, y_pos), 5, 0.5,
                              boxstyle="round,pad=0.05",
                              facecolor=colors['norm'],
                              edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(5, y_pos + 0.25, 'RMSNorm', ha='center', va='center', fontsize=9)

        y_pos -= 0.8

        # Multi-Head Attention
        rect = FancyBboxPatch((1.5, y_pos), 7, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['attention'],
                              edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(5, y_pos + 0.4, f'Multi-Head Attention (RoPE)', ha='center', va='center',
                fontsize=10, fontweight='bold')
        ax.text(8.5, y_pos + 0.4, f"{config.get('n_heads', 'H')} heads",
                ha='left', va='center', fontsize=8, style='italic')

        y_pos -= 1

        # RMSNorm
        rect = FancyBboxPatch((2.5, y_pos), 5, 0.5,
                              boxstyle="round,pad=0.05",
                              facecolor=colors['norm'],
                              edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(5, y_pos + 0.25, 'RMSNorm', ha='center', va='center', fontsize=9)

        y_pos -= 0.8

        # FFN with SwiGLU
        rect = FancyBboxPatch((1.5, y_pos), 7, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['ffn'],
                              edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(5, y_pos + 0.4, 'Feedforward + SwiGLU', ha='center', va='center',
                fontsize=10, fontweight='bold')

        y_pos -= 1.2

    if n_layers > 2:
        ax.text(5, y_pos + 0.5, '...', ha='center', fontsize=20, fontweight='bold')
        y_pos -= 0.5

    # Output FFN
    rect = FancyBboxPatch((2, y_pos), 6, 0.8,
                          boxstyle="round,pad=0.1",
                          facecolor=colors['output'],
                          edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, y_pos + 0.4, 'Output FFN + SwiGLU', ha='center', va='center',
            fontsize=11, fontweight='bold')

    y_pos -= 1.2

    # Final projection
    rect = FancyBboxPatch((2, y_pos), 6, 0.8,
                          boxstyle="round,pad=0.1",
                          facecolor=colors['output'],
                          edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, y_pos + 0.4, 'Vocabulary Projection', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(8.5, y_pos + 0.4, f"[B, T, {config.get('vocab_size', 'V')}]",
            ha='left', va='center', fontsize=9, style='italic')

    # Add legend
    legend_y = 0.5
    ax.text(1, legend_y, 'Legend:', fontsize=10, fontweight='bold')
    legend_items = [
        ('Embedding', colors['embedding']),
        ('Normalization', colors['norm']),
        ('Attention', colors['attention']),
        ('Feedforward', colors['ffn'])
    ]

    for i, (label, color) in enumerate(legend_items):
        x = 2 + (i % 2) * 3
        y = legend_y - (i // 2) * 0.4
        rect = FancyBboxPatch((x, y - 0.15), 0.3, 0.3,
                              boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x + 0.5, y, label, va='center', fontsize=8)

    plt.tight_layout()
    return fig


def visualize_attention_pattern(attention_weights, tokens=None, layer_name="Attention"):
    """
    Visualize attention weights as a heatmap

    Args:
        attention_weights: Attention weight matrix (seq_len, seq_len)
        tokens: Optional list of token strings
        layer_name: Name of the layer

    Returns:
        plotly Figure object
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().detach().numpy()

    fig = go.Figure(data=go.Heatmap(
        z=attention_weights,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Attention<br>Weight")
    ))

    fig.update_layout(
        title=f'{layer_name} - Attention Pattern',
        xaxis_title='Key Position',
        yaxis_title='Query Position',
        width=600,
        height=600,
        yaxis=dict(autorange='reversed')  # Reverse y-axis for better readability
    )

    if tokens:
        fig.update_xaxis(ticktext=tokens, tickvals=list(range(len(tokens))))
        fig.update_yaxis(ticktext=tokens, tickvals=list(range(len(tokens))))

    return fig


def visualize_training_progress_interactive(history_df):
    """
    Create interactive training progress visualization

    Args:
        history_df: DataFrame with training history

    Returns:
        plotly Figure object
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Loss Over Time', 'Loss Distribution'),
        vertical_spacing=0.12
    )

    epochs = history_df.index * 10

    # Loss over time
    fig.add_trace(
        go.Scatter(x=epochs, y=history_df['train'],
                  name='Train Loss', mode='lines+markers',
                  line=dict(color='#FF6B6B', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history_df['val'],
                  name='Val Loss', mode='lines+markers',
                  line=dict(color='#4ECDC4', width=2)),
        row=1, col=1
    )

    # Loss distribution
    fig.add_trace(
        go.Histogram(x=history_df['train'], name='Train Loss Distribution',
                    marker_color='#FF6B6B', opacity=0.7, nbinsx=20),
        row=2, col=1
    )
    fig.add_trace(
        go.Histogram(x=history_df['val'], name='Val Loss Distribution',
                    marker_color='#4ECDC4', opacity=0.7, nbinsx=20),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Loss Value", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    fig.update_layout(
        height=800,
        title_text="Training Progress Dashboard",
        showlegend=True,
        barmode='overlay'
    )

    return fig


def visualize_model_size(model_info):
    """
    Visualize model parameter distribution

    Args:
        model_info: Dictionary with model information

    Returns:
        plotly Figure object
    """
    # Group parameters by layer type
    layer_data = {}
    for layer_name, layer_info in model_info['layers'].items():
        layer_type = layer_name.split('.')[0]
        if layer_type not in layer_data:
            layer_data[layer_type] = 0
        layer_data[layer_type] += layer_info['parameters']

    labels = list(layer_data.keys())
    values = list(layer_data.values())

    fig = go.Figure(data=[
        go.Pie(labels=labels, values=values, hole=.3,
               marker=dict(colors=px.colors.qualitative.Set3))
    ])

    fig.update_layout(
        title=f"Model Parameters Distribution<br>"
              f"Total: {model_info['total_parameters']:,} params "
              f"({model_info['model_size_mb']:.2f} MB)",
        height=500
    )

    return fig


def create_transformer_flow_diagram():
    """
    Create an interactive flow diagram showing how data moves through transformer

    Returns:
        plotly Figure object
    """
    fig = go.Figure()

    # Define nodes
    nodes = [
        "Input Tokens",
        "Embeddings",
        "RMSNorm",
        "Multi-Head<br>Attention",
        "Residual +",
        "RMSNorm",
        "FFN +<br>SwiGLU",
        "Residual +",
        "Output<br>Logits"
    ]

    y_positions = [8, 7, 6, 5, 4.5, 4, 3, 2.5, 1]
    node_colors = ['#FFE5E5', '#FFD1D1', '#FFA07A', '#4ECDC4', '#98D8C8',
                   '#FFA07A', '#45B7D1', '#98D8C8', '#FFE5E5']

    # Add nodes
    for i, (node, y, color) in enumerate(zip(nodes, y_positions, node_colors)):
        fig.add_trace(go.Scatter(
            x=[5], y=[y],
            mode='markers+text',
            marker=dict(size=60, color=color, line=dict(color='black', width=2)),
            text=node,
            textposition='middle center',
            textfont=dict(size=10, color='black', family='Arial Black'),
            showlegend=False,
            hoverinfo='text',
            hovertext=f'{node} Layer'
        ))

    # Add arrows
    for i in range(len(y_positions) - 1):
        fig.add_annotation(
            x=5, y=y_positions[i],
            ax=5, ay=y_positions[i+1],
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor='black'
        )

    fig.update_layout(
        title="Transformer Data Flow",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[3, 7]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 9]),
        height=700,
        plot_bgcolor='white'
    )

    return fig
