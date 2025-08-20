import torch
import torch.nn as nn
from unet import UNetConditionnel
from config import config
from get_data import train_loader, get_embedding
from fonctions_diff_unet import noise_imgs, sample
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import os
from piq import ssim

mean = config["mean"]
std = config["std"]
device = config["device"]
batch_size = config["batch_size"]
n_epochs = config["n_epochs"]
learning_rate = config["learning_rate"]
img_channels = config["img_channels"] 

model = UNetConditionnel(in_channels=img_channels, out_channels=img_channels).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()
print(f"Num params: {(sum(p.numel() for p in model.parameters())) / 1e6} M")


#reprendre l entrainement a partir des sauvegardes
current_epoch = 4
if current_epoch != 0:
    checkpoint = torch.load(f"./sauvegardes_entrainement2/unet_checkpoint_epoch_{current_epoch}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    

loss_list = []
for epoch in range(current_epoch+1, current_epoch+n_epochs+1):
    for i, (train_images, textes) in tqdm(enumerate(train_loader), total=len(train_loader)):
        train_images = train_images.to(device)
        textes = [str(texte) for texte in textes]

        labels = get_embedding(textes).to(device)
        t = torch.rand((train_images.shape[0], 1), device=device)

        # Bruitage des images (pas de latent car unet!)
        xt, x0 = noise_imgs(train_images, t)
        targets = train_images - x0 

   
        preds = model(xt, t, labels)

        loss = mse(preds, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if i%10 == 0:
            loss_list.append(loss.item())



        if epoch%1 == 0 and i%300 == 0:
            s = 5
            loss_list_moy = torch.tensor(loss_list[:len(loss_list)//s*s]).view(-1, s).mean(1)

            model.eval()
            with torch.no_grad():
                test_texts = [
                    "un triangle petit vert en haut à gauche, un triangle petit noir en bas à gauche, un losange moyen jaune au milieu, un disque grand jaune en bas à droite et un cercle petit violet en haut à droite",
                    "un carré grand rouge en bas à droite"
                ]
                test_labels = get_embedding(test_texts).to(device)
                sampled_imgs = sample(len(test_texts), model, test_labels)
                
                sampled_imgs = torch.clamp(sampled_imgs.cpu() * std.view(-1, 1, 1) + mean.view(-1, 1, 1), 0, 1) #denormaliser
                
            model.train()
            fig = plt.figure(figsize=(20, 8))
            gs = GridSpec(2, len(test_texts), figure=fig)

            ax1 = fig.add_subplot(gs[0, :len(test_texts)//2])
            ax1.plot(loss_list, c="blue")
            ax1.set_title("Loss brute")

            ax2 = fig.add_subplot(gs[0, len(test_texts)//2:])
            ax2.plot(loss_list_moy, c="blue")
            ax2.set_title("Loss Moyennée")

            for j in range(len(test_texts)):
                ax = fig.add_subplot(gs[1, j])
                img = sampled_imgs[j].detach().permute(1, 2, 0)
                ax.imshow(img)
                ax.axis('off')

            plt.tight_layout()
            os.makedirs("outputs2", exist_ok=True)
            plt.savefig(f"outputs2/graph_epoch_{epoch}_iter_{i}.png")
            plt.close()

    # Sauvegarde
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, f"./sauvegardes_entrainement2/unet_checkpoint_epoch_{epoch}.pt")

    