import torch
from clip_a_la_main import ImgToEmbdNetwork, TextToEmbdNetwork
from config import config
from get_data import train_loader, val_loader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import CLIPTokenizer

device = config["device"]

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

model_img_to_embd = ImgToEmbdNetwork().to(device)
print(f"Num params model_img_to_embd: {(sum(p.numel() for p in model_img_to_embd.parameters())) / 1e6} M")

model_text_to_embd = TextToEmbdNetwork().to(device)
print(f"Num params model_text_to_embd: {(sum(p.numel() for p in model_text_to_embd.parameters())) / 1e6} M")

optimizer = torch.optim.AdamW(list(model_img_to_embd.parameters()) + list(model_text_to_embd.parameters()), lr=3e-4)

train_loss_list, val_loss_list = [], []


# pour reprendre l'entrainement à partir des sauvegardes
current_epoch = 18
if current_epoch != 0:
    checkpoint = torch.load(f"./sauvegardes_entrainement/clip_checkpoint_epoch_{current_epoch}.pt")
    model_img_to_embd.load_state_dict(checkpoint['model_img_to_embd_state_dict'])
    model_text_to_embd.load_state_dict(checkpoint['model_text_to_embd_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])




def clip_loss(image_embeds, text_embeds, temperature=0.07):
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    logits = image_embeds @ text_embeds.T / temperature
    labels = torch.arange(len(logits)).to(logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


for epoch in range(current_epoch+1, current_epoch+100):
    for i, (imgs, textes) in tqdm(enumerate(train_loader), total=len(train_loader)):

        imgs = imgs.to(device)
        textes = [str(texte) for texte in textes]
        labels = tokenizer(textes, padding="max_length", truncation=True, max_length=77, return_tensors="pt")["input_ids"].to(torch.float32).to(device)

        img_embd = model_img_to_embd(imgs)
        text_embd = model_text_to_embd(labels)

        loss = clip_loss(img_embd, text_embd)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


        if i%10 == 0:
            train_loss_list.append(loss.item())

            model_img_to_embd.eval()
            model_text_to_embd.eval()
            imgs, textes = next(iter(val_loader))
            imgs = imgs.to(device)
            textes = [str(texte) for texte in textes]
            labels = tokenizer(textes, padding="max_length", truncation=True, max_length=77, return_tensors="pt")["input_ids"].to(torch.float32).to(device)
            img_embd = model_img_to_embd(imgs)
            text_embd = model_text_to_embd(labels)
            val_loss = clip_loss(img_embd, text_embd)
            val_loss_list.append(val_loss.item())

    
        if epoch % 1 == 0 and i%200 == 0:

            # Mise en eval
            model_img_to_embd.eval()
            model_text_to_embd.eval()

            # Récupère un batch
            images, textes = next(iter(val_loader))
            images = images.to(device)
            textes = [str(t) for t in textes]
            labels = tokenizer(textes, padding="max_length", truncation=True, max_length=77, return_tensors="pt")["input_ids"].to(torch.float32).to(device)

            # Embeddings
            with torch.no_grad():
                image_embeds = model_img_to_embd(images)
                text_embeds = model_text_to_embd(labels)

            # Similarité
            image_embeds = F.normalize(image_embeds, dim=-1)
            text_embeds = F.normalize(text_embeds, dim=-1)
            similarity_matrix = image_embeds @ text_embeds.T  # [B, B]

            # 🎨 Figure combinée
            plt.figure(figsize=(12, 5))

            # Subplot 1 : courbe de loss
            plt.subplot(1, 2, 1)
            plt.plot(train_loss_list, c="blue")
            plt.plot(val_loss_list, c="red")
            plt.title("Évolution de la loss")
            plt.xlabel("Itérations (x10)")
            plt.ylabel("Loss")

            # Subplot 2 : heatmap de similarité
            plt.subplot(1, 2, 2)
            plt.imshow(similarity_matrix.cpu(), cmap='viridis')
            plt.title("Similarité image / texte")
            plt.xlabel("Texte")
            plt.ylabel("Image")
            plt.colorbar(fraction=0.046, pad=0.04)

            # Sauvegarde
            plt.tight_layout()
            plt.savefig("entraînement_clip.png")

            # Remet en mode train
            model_img_to_embd.train()
            model_text_to_embd.train()



    if epoch%3 == 0:
        torch.save({
            'epoch': epoch,
            'model_img_to_embd_state_dict': model_img_to_embd.state_dict(),
            'model_text_to_embd_state_dict': model_text_to_embd.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, f"./sauvegardes_entrainement/clip_checkpoint_epoch_{epoch}.pt")
