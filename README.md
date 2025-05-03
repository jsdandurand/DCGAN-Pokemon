# ViTGAN-MNIST

A Vision Transformer-based GAN implementation for generating MNIST digits, featuring both conditional generation and spatial attention mechanisms.

## Features

- **Vision Transformer (ViT) Architecture**: Implements both generator and discriminator using transformer blocks
- **Conditional Generation**: Generate specific digits by conditioning on class labels
- **Flexible Attention Mechanisms**: 
  - Standard Multi-Head Attention
  - Spatial Attention with local feature extraction
- **MNIST Dataset**: Trained on the classic MNIST handwritten digit dataset

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
matplotlib>=3.3.0
```

## Project Structure

```
.
├── src/
│   ├── main.py        # Main training script
│   ├── train.py       # Training utilities and visualization
│   └── vit.py         # Model architecture implementation
└── README.md
```

## Usage

### Training the Model

```python
python src/main.py
```

### Model Configuration

The ViTGAN can be configured with different attention mechanisms:

```python
# Standard attention
model = ViTGAN(
    embed_dim=256,
    discriminator="ViT",
    attention_type="normal"
)

# Spatial attention
model = ViTGAN(
    embed_dim=256,
    discriminator="ViT",
    attention_type="spatial"
)
```

### Conditional Generation

Generate specific digits by providing class labels:

```python
# Generate images of digit '7'
labels = torch.tensor([7] * batch_size)
fake_images = model.generate(noise, labels)
```

## Architecture Details

### Generator
- Vision Transformer-based architecture
- Supports both standard and spatial attention
- Class conditioning through learnable embeddings
- Patch-based image generation

### Discriminator
- Vision Transformer or MLP-based options
- CLS token classification
- Flexible attention mechanisms

## Training Process

The model uses:
- BCE loss for GAN objective
- Adam optimizer
- Patch size of 7x7 for 28x28 MNIST images
- Conditional generation with class embeddings

## Visualization

The training process includes visualization of:
- Generated samples
- Training losses
- Class-conditional generation results

## License

This project is open-source and available under the MIT License. 