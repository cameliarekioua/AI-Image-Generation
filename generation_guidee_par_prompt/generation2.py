import torch
from DiT2 import DiT
from config import config
from get_data import get_embedding, get_tokens
from fonctions_diffusion import sample
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = config["device"]
mean = config["mean"]
std = config["std"]

model = DiT().to(device)
print(f"Num params: {(sum(p.numel() for p in model.parameters())) / 1e6} M")

# pour reprendre l'entrainement à partir des sauvegardes
current_epoch = 60
if current_epoch != 0:
    checkpoint = torch.load(f"./sauvegardes_entrainement/dit_checkpoint3_epoch_{current_epoch}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])


# importer le vae
from vae import VAE
vae = VAE().to(device)
epoch_vae = 35
checkpoint = torch.load(f"./sauvegardes_entrainement/vae_checkpoint_epoch_{epoch_vae}.pt")
vae.load_state_dict(checkpoint['model_state_dict'])



textes = ["un triangle petit vert en haut à gauche, un triangle petit noir en bas à gauche, un losange moyen jaune au milieu, un disque grand jaune en bas à droite et un cercle petit violet en haut à droite", "un losange violet petit en bas à droite, un losange violet grand en haut à droite", "un cercle rouge au milieu"]
labels = get_embedding(textes)
text_cond = get_tokens(textes)
sampled_imgs = sample(len(textes), model, labels, text_cond)
sampled_imgs = vae.decoder(sampled_imgs)

fig = plt.figure(figsize=(30, 10))

for i in range(len(textes)):
    ax = fig.add_subplot(1, len(textes), i + 1)
    # ax.text(0.5, 0.5, textes[i], fontsize=20, ha='center', va='center')
    img = torch.clamp(F.sigmoid(sampled_imgs[i]).detach().cpu() * std.view(-1, 1, 1) + mean.view(-1, 1, 1), 0, 1).permute(1, 2, 0)
    ax.imshow(img)
    ax.axis('off')  # Désactive les axes pour une meilleure présentation

plt.tight_layout()  # Assure que tout s'affiche bien
plt.savefig("generation2.png")  # Sauvegarde le graphe dans un fichier
plt.close(fig)  # Ferme la figure pour libérer la mémoire
