import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from config import config
import clip


img_size = config["img_size"]
mean = config["mean"]
std = config["std"]
batch_size = config["batch_size"]
device = config["device"]


transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    captions = [item[1][0] for item in batch]  # première caption de chaque image
    return images, captions

train_dataset = datasets.CocoCaptions(root="./data/coco/train2017", annFile="./data/coco/annotations/captions_train2017.json", transform=transform)
val_dataset = datasets.CocoCaptions(root="./data/coco/val2017", annFile="./data/coco/annotations/captions_val2017.json", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=3)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=3)



clip_model, _ = clip.load("ViT-B/32", device=device)
clip_model.eval()

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

