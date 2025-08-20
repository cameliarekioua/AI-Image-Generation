import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import clip
from config import config
import os
from PIL import Image

img_size = config["img_size"]
mean = config["mean"]
std = config["std"]
batch_size = config["batch_size"]
device = config["device"]

class CustomDataset(Dataset):

    def __init__(self, image_dir, prompt_dir, transform=None):
        self.image_dir = image_dir
        self.prompt_dir = prompt_dir
        self.transform = transform

        self.image_files = sorted(os.listdir(image_dir))
        self.prompts = {}

        for img_file in self.image_files:
            prompt_file = os.path.splitext(img_file)[0] + ".txt"
            with open(os.path.join(prompt_dir, prompt_file), "r", encoding="utf-8") as f:
                self.prompts[img_file] = f.read().strip()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_filename)

        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        prompt = self.prompts[image_filename]

        return image, prompt

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),  # Convertit vers RGB seulement si besoin
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_dataset = CustomDataset("data/formes_geometriques/images", "data/formes_geometriques/prompts", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

val_dataset = CustomDataset("data/formes_geometriques/images_test", "data/formes_geometriques/prompts_test", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)




# pour obtenir des tokens à partir des noms des textures
clip_model, _ = clip.load("ViT-B/32", device=device)

def get_embedding(prompt):
    with torch.no_grad():
        text_tokens = clip.tokenize(prompt).to(device)
        text_embedding = clip_model.encode_text(text_tokens)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    return text_embedding.to(torch.float32)




from transformers import CLIPTokenizer, CLIPTextModel

# Charge le tokenizer et le modèle texte CLIP (comme dans Stable Diffusion)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
text_encoder.eval().to(device)

def get_tokens(prompts, max_length=77):
    if isinstance(prompts, str): prompts = [prompts]
    with torch.no_grad():
        tokens = tokenizer(prompts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)
        outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state  # (B, 77, 768)




# from transformers import CLIPTokenizer
# from clip_a_la_main import TextToEmbdNetwork

# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
# model_text_to_embd = TextToEmbdNetwork().to(device)

# # pour reprendre l'entrainement à partir des sauvegardes
# current_epoch = 12
# if current_epoch != 0:
#     checkpoint = torch.load(f"./sauvegardes_entrainement/clip_checkpoint_epoch_{current_epoch}.pt")
#     model_text_to_embd.load_state_dict(checkpoint['model_text_to_embd_state_dict'])

# def get_embedding(prompt):
#     with torch.no_grad():
#         tokens = tokenizer(prompt, padding="max_length", truncation=True, max_length=77, return_tensors="pt")["input_ids"].to(torch.float32).to(device)
#         text_embd = model_text_to_embd(tokens)
#     return text_embd
