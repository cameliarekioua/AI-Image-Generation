import torch
from DiT import DiT
from config import config
from fonctions_diffusion import sample
import matplotlib.pyplot as plt
import torch.nn.functional as F
import clip

device = config["device"]
mean = config["mean"]
std = config["std"]

model = DiT().to(device)
print(f"Num params: {(sum(p.numel() for p in model.parameters())) / 1e6} M")


# pour reprendre l'entrainement à partir des sauvegardes
current_epoch = 40
if current_epoch != 0:
    checkpoint = torch.load(f"./sauvegardes_entrainement/dit_checkpoint_epoch_{current_epoch}.pt", map_location=torch.device('mps'))
    model.load_state_dict(checkpoint['model_state_dict'])


# importer le vae
from vae import VAE
vae = VAE().to(device)
epoch_vae = 15
checkpoint = torch.load(f"./sauvegardes_entrainement/vae_checkpoint_epoch_{epoch_vae}.pt", map_location=torch.device('mps'))
vae.load_state_dict(checkpoint['model_state_dict'])



textes = ["un rond bleu au milieu", "un carré rouge en bas à gauche", "un rond jaune en haut à droite, un disque vert au milieu, un losange grand violet en bas à droite"]
labels = get_embedding(textes)
sampled_imgs = sample(len(textes), model, labels)
sampled_imgs = vae.decoder(sampled_imgs)

fig = plt.figure(figsize=(30, 10))

for i in range(len(textes)):
    ax = fig.add_subplot(1, len(textes), i + 1)
    ax.text(0.5, 0.5, textes[i], fontsize=20, ha='center', va='center')
    img = torch.clamp(F.sigmoid(sampled_imgs[i]).detach().cpu() * std.view(-1, 1, 1) + mean.view(-1, 1, 1), 0, 1).permute(1, 2, 0)
    ax.imshow(img)
    ax.axis('off')  # Désactive les axes pour une meilleure présentation

plt.tight_layout()  # Assure que tout s'affiche bien
#plt.show("generation.png")  # Sauvegarde le graphe dans un fichier
#plt.close(fig)  # Ferme la figure pour libérer la mémoire
plt.show()  # Affiche le graphe