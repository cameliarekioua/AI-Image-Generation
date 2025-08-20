import torch
from tqdm import tqdm
from get_data import train_loader, val_loader
from config import config
from vae import VAE
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = config["device"]
img_channels = config["img_channels"]
mean, std = config["mean"], config["std"]
n_epochs = 30
learning_rate = 1e-4

model = VAE().to(device)
print(f"Num params: {(sum(p.numel() for p in model.parameters())) / 1e6} M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')

def sample(mu, log_var):
    sigma = torch.exp(1/2 * log_var)
    noise = torch.randn_like(mu)
    z = noise * sigma + mu
    out = model.decoder(z)
    return out

train_loss_list, val_loss_list = [], []

mixed_precision = True
scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)

epoch_init = 18
if epoch_init != 0:
    checkpoint = torch.load(f"./sauvegardes_entrainement/vae_checkpoint_epoch_{epoch_init}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #scaler.load_state_dict(checkpoint["scaler"])


for epoch in range(epoch_init+1, epoch_init+n_epochs):
    for i, (imgs, _) in tqdm(enumerate(train_loader), total=len(train_loader)):

        imgs = imgs.to(device)
        
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=mixed_precision):
            reconstructed_imgs, mu, log_var = model(imgs)

            reconstruction_loss = criterion(reconstructed_imgs, imgs)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = (reconstruction_loss + kl_loss) / imgs.shape[0]

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        if i%30 == 0:
            train_loss_list.append(loss.item())

            with torch.no_grad():
                imgs, _ = next(iter(val_loader))
                imgs = imgs.to(device)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=mixed_precision):
                    reconstructed_imgs, mu, log_var = model(imgs)
                    reconstruction_loss = criterion(reconstructed_imgs, imgs)
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                    val_loss = (reconstruction_loss + kl_loss) / imgs.shape[0]
                val_loss_list.append(val_loss.item())

            

    plt.figure()
    plt.plot(train_loss_list, c="blue")
    plt.plot(val_loss_list, c="red")
    plt.savefig("training_loss_vae.png")
    plt.close()

    plt.figure()
    plt.imshow(imgs[0].permute(1, 2, 0).detach().cpu() * std + mean)
    plt.savefig("image.png")
    plt.close()

    plt.figure()
    plt.imshow(F.sigmoid(reconstructed_imgs[0]).permute(1, 2, 0).detach().cpu() * std + mean)
    plt.savefig("image_reconstruite.png")
    plt.close()



    if epoch%5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            'loss': loss.item(),
        }, f"./sauvegardes_entrainement/vae_checkpoint_epoch_{epoch}.pt")
