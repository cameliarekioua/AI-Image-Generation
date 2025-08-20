import torch
import torch.nn as nn
from DiT import DiT
from config import config
from get_data import train_loader, get_embedding
from fonctions_diffusion import noise_imgs, sample
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import time

mean = config["mean"]
std = config["std"]
device = config["device"]
batch_size = config["batch_size"]
n_epochs = config["n_epochs"]
learning_rate = config["learning_rate"]

model = DiT().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()

print(f"Num params: {(sum(p.numel() for p in model.parameters())) / 1e6} M")

loss_list = []


# pour reprendre l'entrainement à partir des sauvegardes
current_epoch = 0
if current_epoch > 0:
    model.load_state_dict(torch.load(f"./sauvegardes_entrainement/model_epoch_{current_epoch}.pth"))
    optimizer.load_state_dict(torch.load(f"./sauvegardes_entrainement/optimizer_epoch_{current_epoch}.pth"))
    loss_list = torch.load(f"./sauvegardes_entrainement/loss_list_epoch_{current_epoch}.pth")


# Training loop
for epoch in range(current_epoch, n_epochs):
    for i, (train_images, textes) in tqdm(enumerate(train_loader), total=len(train_loader)):

        x1 = train_images.to(device)
        # textes = textes[0]   # pour coco
        textes = [str(texte.item()) for texte in textes]   # pour mnist

        labels = get_embedding(textes)

        t = torch.rand((x1.shape[0], 1), device=device)
        xt, x0 = noise_imgs(x1, t)
        targets = x1 - x0

        preds = model(xt, t, labels)

        loss = mse(preds, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if i%10 == 0:
            loss_list.append(loss.item())


        if epoch % 1 == 0 and i==100:
            s = 5
            loss_list_moyennée = torch.tensor(loss_list[:len(loss_list)//s*s]).view(-1, s).mean(1)
            textes = ['0', '1', '2', '3', '4', '5']
            labels = get_embedding(textes)
            sampled_imgs = sample(len(textes), model, labels)

            if len(loss_list_moyennée) > 0:
                print(f"last loss_moyennée : {loss_list_moyennée[-1]}")

            fig = plt.figure(figsize=(20, 8))
            gs = GridSpec(2, len(textes), figure=fig)

            # Graphiques sur la première ligne
            ax1 = fig.add_subplot(gs[0, :len(textes)//2])  # Fusionne les premières colonnes
            ax1.plot(loss_list, c="blue")
            ax1.set_title("Loss List")

            ax2 = fig.add_subplot(gs[0, len(textes)//2:])  # Fusionne les colonnes restantes
            ax2.plot(loss_list_moyennée, c="blue")
            ax2.set_title("Loss List Moyennée")

            # Images sur la deuxième ligne
            for j in range(len(textes)):
                ax = fig.add_subplot(gs[1, j])
                img = torch.clamp(sampled_imgs[j].detach().cpu() * std.view(-1, 1, 1) + mean.view(-1, 1, 1), 0, 1).permute(1, 2, 0)
                ax.imshow(img)
                ax.axis('off')  # Désactive les axes pour une meilleure présentation

            plt.tight_layout()  # Assure que tout s'affiche bien
            plt.savefig("graph_mnist.png")  # Sauvegarde le graphe dans un fichier
            plt.close(fig)  # Ferme la figure pour libérer la mémoire


    if (epoch+1) % 1 == 0:
        torch.save(model.state_dict(), f"./sauvegardes_entrainement/model_epoch_{epoch+1}.pth")
        torch.save(optimizer.state_dict(), f"./sauvegardes_entrainement/optimizer_epoch_{epoch+1}.pth")
        torch.save(loss_list, f"./sauvegardes_entrainement/loss_list_epoch_{epoch+1}.pth")

