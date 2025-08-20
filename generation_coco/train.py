import torch
import torch.nn as nn
from DiT import DiT
from config import config
from get_data import train_loader, val_loader, get_embedding, get_tokens
from fonctions_diffusion import noise_imgs, sample
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import time


mean = config["mean"]
std = config["std"]
device = config["device"]
n_epochs = config["n_epochs"]
learning_rate = config["learning_rate"]


model = DiT().to(device)
print(f"Num params: {(sum(p.numel() for p in model.parameters())) / 1e6} M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()

train_loss_list, val_loss_list = [], []

mixed_precision = True
scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)


# pour reprendre l'entrainement à partir des sauvegardes
current_epoch = 5
if current_epoch != 0:
    checkpoint = torch.load(f"./sauvegardes_entrainement/dit_checkpoint2_epoch_{current_epoch}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint["scaler"])


# VAE pré-entraîné
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
vae.eval()
vae = vae.to(device)


# compiler le modèle pour optimiser sur gpu
model = torch.compile(model)
vae = torch.compile(vae)


# Training loop
for epoch in range(current_epoch+1, current_epoch+n_epochs):
    for i, (imgs, textes) in tqdm(enumerate(train_loader), total=len(train_loader)):

        imgs = imgs.to(device)
        labels = get_embedding(textes)
        text_cond = get_tokens(textes)

        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=mixed_precision):
                posterior = vae.encode(imgs)
            x1 = posterior.latent_dist.sample() * 0.18215   # x1 est en float32

        t = torch.rand((x1.shape[0], 1), device=device)
        xt, x0 = noise_imgs(x1, t)
        targets = x1 - x0

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=mixed_precision):
            preds = model(xt, t, labels, text_cond)   # preds est en float16
            loss = mse(preds, targets)   # loss est en float32

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()



        if i%50 == 0:
            train_loss_list.append(loss.item())
            
            model.eval()
            with torch.no_grad():
                imgs, textes = next(iter(val_loader))
                imgs = imgs.to(device)
                labels = get_embedding(textes)
                text_cond = get_tokens(textes)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=mixed_precision):
                    posterior = vae.encode(imgs)
                x1 = posterior.latent_dist.sample() * 0.18215
                t = torch.rand((x1.shape[0], 1), device=device)
                xt, x0 = noise_imgs(x1, t)
                targets = x1 - x0
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=mixed_precision):
                    preds = model(xt, t, labels, text_cond)
                    val_loss = mse(preds, targets)
            model.train()
            val_loss_list.append(val_loss.item())


        if epoch % 1 == 0 and i%500 == 0:
            s = 10
            train_loss_list_moyennée = torch.tensor(train_loss_list[:len(train_loss_list)//s*s]).view(-1, s).mean(1)
            val_loss_list_moyennée = torch.tensor(val_loss_list[:len(val_loss_list)//s*s]).view(-1, s).mean(1)
            textes = ["A sheep and its lamb in a fenced grass enclosure.", "A group of elephants that are standing in the dirt.", "a man", "a cute cat"]
            labels = get_embedding(textes)
            text_cond = get_tokens(textes)
            n_samples = len(textes)
            sampled_imgs = sample(n_samples, model, mixed_precision, labels, text_cond) / 0.18215
            with torch.no_grad():
                sampled_imgs = vae.decode(sampled_imgs).sample

            fig = plt.figure(figsize=(20, 8))
            gs = GridSpec(2, n_samples, figure=fig)

            # Graphiques sur la première ligne
            ax1 = fig.add_subplot(gs[0, :n_samples//2])  # Fusionne les premières colonnes
            ax1.plot(train_loss_list, c="blue")
            ax1.plot(val_loss_list, c="red")
            ax1.set_title("Loss List")

            ax2 = fig.add_subplot(gs[0, n_samples//2:])  # Fusionne les colonnes restantes
            ax2.plot(train_loss_list_moyennée, c="blue")
            ax2.plot(val_loss_list_moyennée, c="red")
            ax2.set_title("Loss List Moyennée")

            # Images sur la deuxième ligne
            for j in range(n_samples):
                ax = fig.add_subplot(gs[1, j])
                img = torch.clamp(sampled_imgs[j].detach().cpu() * std.view(-1, 1, 1) + mean.view(-1, 1, 1), 0, 1).permute(1, 2, 0)
                ax.imshow(img)
                ax.set_title(textes[j], fontsize=10)
                ax.axis('off')  # Désactive les axes pour une meilleure présentation

            plt.tight_layout()  # Assure que tout s'affiche bien
            plt.savefig("graph2.png")  # Sauvegarde le graphe dans un fichier
            plt.close(fig)  # Ferme la figure pour libérer la mémoire



    if epoch % 8 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model._orig_mod.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "loss": loss.item(),
        }, f"./sauvegardes_entrainement/dit_checkpoint2_epoch_{epoch}.pt")
