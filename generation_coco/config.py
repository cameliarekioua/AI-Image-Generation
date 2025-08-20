import torch

config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "img_size": 128, #256,
    "img_channels": 3,
    "n_classes": -1,
    "mean": torch.tensor([0.5, 0.5, 0.5]),
    "std": torch.tensor([0.5, 0.5, 0.5]),

    "batch_size": 128,
    "n_epochs": 1000,
    "learning_rate": 1e-4,

    "T": 200,

    "latent_img_size": 16, #32,
    "latent_img_channels": 4,
    "patch_size": 2,

    "n_blocks": 12,
    "n_heads": 12,
    "embd_dim": 12 * 64,
    "dropout": 0.0
}
