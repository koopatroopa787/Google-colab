"""
Data loading and preprocessing utilities

This module handles:
- Text dataset loading
- Tokenization (character-level)
- Batch generation for training
"""

import torch
import urllib.request
import os


def download_dataset(url, file_name):
    """
    Download a text dataset from URL

    Args:
        url: URL to download from
        file_name: Local file name to save to
    """
    if not os.path.exists(file_name):
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, file_name)
        print(f"✓ Dataset saved to {file_name}")
    else:
        print(f"✓ Dataset already exists at {file_name}")


def load_text_data(file_path):
    """
    Load text data from file

    Args:
        file_path: Path to text file

    Returns:
        Text content as string
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def create_char_tokenizer(text):
    """
    Create character-level tokenizer

    Args:
        text: Input text

    Returns:
        vocab: Sorted list of unique characters
        stoi: Character to index mapping
        itos: Index to character mapping
        encode: Function to encode text to indices
        decode: Function to decode indices to text
    """
    # Create vocabulary from unique characters
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)

    # Create mappings
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}

    # Encode and decode functions
    encode = lambda s: [stoi[ch] for ch in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    print(f"✓ Vocabulary created")
    print(f"  - Vocab size: {vocab_size}")
    print(f"  - Sample characters: {vocab[:10]}")

    return vocab, stoi, itos, encode, decode


def prepare_dataset(file_path, url=None):
    """
    Complete dataset preparation pipeline

    Args:
        file_path: Path to dataset file
        url: Optional URL to download from if file doesn't exist

    Returns:
        dataset: PyTorch tensor of encoded text
        vocab: List of characters
        encode: Encoding function
        decode: Decoding function
    """
    # Download if needed
    if url and not os.path.exists(file_path):
        download_dataset(url, file_path)

    # Load text
    text = load_text_data(file_path)
    print(f"✓ Loaded {len(text):,} characters")

    # Create tokenizer
    vocab, stoi, itos, encode, decode = create_char_tokenizer(text)

    # Encode entire dataset
    dataset = torch.tensor(encode(text), dtype=torch.int8)
    print(f"✓ Dataset encoded to tensor of shape {dataset.shape}")

    return dataset, vocab, encode, decode


def get_batches(data, split, batch_size, context_window, config=None):
    """
    Generate random batches for training/validation/testing

    Args:
        data: Full dataset tensor
        split: 'train', 'val', or 'test'
        batch_size: Number of sequences in batch
        context_window: Length of each sequence
        config: Optional config dict (not used currently)

    Returns:
        x: Input sequences (batch_size, context_window)
        y: Target sequences (batch_size, context_window)
    """
    # Split dataset: 80% train, 10% val, 10% test
    train_data = data[:int(.8 * len(data))]
    val_data = data[int(.8 * len(data)): int(.9 * len(data))]
    test_data = data[int(.9 * len(data)):]

    # Select appropriate split
    batch_data = train_data
    if split == 'val':
        batch_data = val_data
    elif split == 'test':
        batch_data = test_data

    # Generate random starting indices
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))

    # Create input (x) and target (y) sequences
    # Target is input shifted by one position (next token prediction)
    x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()

    return x, y


def get_dataset_stats(dataset):
    """
    Get statistics about the dataset

    Args:
        dataset: Encoded dataset tensor

    Returns:
        Dictionary with dataset statistics
    """
    return {
        'total_chars': len(dataset),
        'train_size': int(len(dataset) * 0.8),
        'val_size': int(len(dataset) * 0.1),
        'test_size': int(len(dataset) * 0.1),
        'unique_tokens': len(torch.unique(dataset))
    }
