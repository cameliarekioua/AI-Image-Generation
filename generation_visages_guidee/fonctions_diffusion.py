import torch
from config import config

device = config["device"]
img_channels = config["latent_img_channels"]
img_size = config["latent_img_size"]
T = config["T"]

def noise_imgs(x1, t):
    t = t.view(-1, 1, 1, 1)
    x0 = torch.randn_like(x1, device=device)
    xt = (1 - t) * x0 + t * x1
    return xt, x0

def sample(n_imgs, model, mixed_precision, labels=None, text_cond=None):
    model.eval()
    with torch.no_grad():
        xt = torch.randn((n_imgs, img_channels, img_size, img_size), device=device)
        for t in torch.linspace(0, 1, T):
            t = t.expand(xt.shape[0]).view(-1, 1).to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=mixed_precision):
                xt = xt + 1/T * model(xt, t, labels, text_cond)
    model.train()
    return xt
