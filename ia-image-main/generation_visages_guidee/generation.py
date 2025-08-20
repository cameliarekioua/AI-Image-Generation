import torch
from DiT import DiT
from config import config
from fonctions_diffusion import sample
from get_data import get_embedding
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

mean = config["mean"]
std = config["std"]
device = config["device"]

model = DiT().to(device)
print(f"Num params: {(sum(p.numel() for p in model.parameters())) / 1e6} M")

mixed_precision = True
scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)

# pour reprendre l'entrainement à partir des sauvegardes
current_epoch = 39
if current_epoch != 0:
    checkpoint = torch.load(f"./sauvegardes_entrainement/dit_checkpoint_epoch_{current_epoch}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

# VAE pré-entraîné
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
vae = vae.to(device)
vae.eval()

# compiler les modèles pour accélérer le code
model = torch.compile(model)
vae = torch.compile(vae)


textes = ["A young man with black hair, straight hair, a chubby face, earrings, and a necklace.", "A smiling attractive young woman with wavy hair, high cheekbones, a big nose, a slightly open mouth, a hat, and lipstick.", "A bald man."]
labels = get_embedding(textes)
n_samples = len(textes)
sampled_imgs = sample(n_samples, model, mixed_precision=mixed_precision) / 0.18215
with torch.no_grad():
    sampled_imgs = vae.decode(sampled_imgs).sample


fig = plt.figure(figsize=(20, 8))
gs = GridSpec(1, n_samples, figure=fig)

# Images sur la deuxième ligne
for j in range(n_samples):
    ax = fig.add_subplot(gs[0, j])
    img = torch.clamp(sampled_imgs[j].detach().cpu() * std.view(-1, 1, 1) + mean.view(-1, 1, 1), 0, 1).permute(1, 2, 0)
    ax.imshow(img)
    ax.axis('off')  # Désactive les axes pour une meilleure présentation

plt.tight_layout()  # Assure que tout s'affiche bien
plt.savefig("generation.png")  # Sauvegarde le graphe dans un fichier
plt.close(fig)  # Ferme la figure pour libérer la mémoire

