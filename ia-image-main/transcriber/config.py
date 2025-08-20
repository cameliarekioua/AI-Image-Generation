import torch

config = {
    "device": "mps",

    "img_size": 256,
    "img_channels": 3,
    "n_classes": -1,
    "mean": torch.tensor([0.0, 0.0, 0.0]),
    "std": torch.tensor([1.0, 1.0, 1.0]),

    "batch_size": 128,
    "n_epochs": 1000,
    "learning_rate": 3e-4,

    "T": 200,

    "latent_img_size": 32,
    "latent_img_channels": 4,
    "patch_size": 2,

    "n_blocks": 12,
    "n_heads": 12,
    "embd_dim": 12 * 64,
    "dropout": 0.0
}
