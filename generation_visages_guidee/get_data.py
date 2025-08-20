from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from config import config
import torch
import clip

# Configs
img_size = config["img_size"]
mean = config["mean"]
std = config["std"]
batch_size = config["batch_size"]
device = config["device"]

# Transforms
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Catégorisation des attributs
def create_prompt(attr_vector, attr_names):
    attrs = {name: (val == 1) for name, val in zip(attr_names, attr_vector)}

    # Âge et genre
    age = "young" if attrs["Young"] else "middle-aged"
    gender = "man" if attrs["Male"] else "woman"

    base = "A"
    if attrs["Smiling"]:
        base += " smiling"
    if attrs["Bald"]:
        base += " bald"
    if attrs["Attractive"]:
        base += " attractive"
    base += f" {age} {gender}"

    # Cheveux
    hair = []
    for color in ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"]:
        if attrs[color]:
            hair.append(color.split("_")[0].lower() + " hair")
    if attrs["Bangs"]:
        hair.append("bangs")
    if attrs["Receding_Hairline"]:
        hair.append("a receding hairline")
    if attrs["Wavy_Hair"]:
        hair.append("wavy hair")
    if attrs["Straight_Hair"]:
        hair.append("straight hair")

    # Visage & yeux
    face = []
    if attrs["High_Cheekbones"]:
        face.append("high cheekbones")
    if attrs["Oval_Face"]:
        face.append("an oval face")
    if attrs["Pointy_Nose"]:
        face.append("a pointy nose")
    if attrs["Big_Nose"]:
        face.append("a big nose")
    if attrs["Big_Lips"]:
        face.append("big lips")
    if attrs["Mouth_Slightly_Open"]:
        face.append("a slightly open mouth")
    if attrs["Bushy_Eyebrows"]:
        face.append("bushy eyebrows")
    if attrs["Arched_Eyebrows"]:
        face.append("arched eyebrows")
    if attrs["Narrow_Eyes"]:
        face.append("narrow eyes")
    if attrs["Bags_Under_Eyes"]:
        face.append("bags under the eyes")
    if attrs["Chubby"]:
        face.append("a chubby face")
    if attrs["Double_Chin"]:
        face.append("a double chin")
    if attrs["Pale_Skin"]:
        face.append("pale skin")
    if attrs["Rosy_Cheeks"]:
        face.append("rosy cheeks")

    # Poils du visage
    facial_hair = []
    # if attrs["5_o_Clock_Shadow"]:
    #     facial_hair.append("5 o'clock shadow")
    if attrs["Goatee"]:
        facial_hair.append("a goatee")
    if attrs["Mustache"]:
        facial_hair.append("a mustache")
    if not attrs["No_Beard"]:
        facial_hair.append("a beard")
    if attrs["Sideburns"]:
        facial_hair.append("sideburns")

    # Accessoires / vêtements
    accessories = []
    if attrs["Eyeglasses"]:
        accessories.append("glasses")
    if attrs["Wearing_Hat"]:
        accessories.append("a hat")
    if attrs["Wearing_Lipstick"]:
        accessories.append("lipstick")
    if attrs["Wearing_Earrings"]:
        accessories.append("earrings")
    if attrs["Wearing_Necklace"]:
        accessories.append("a necklace")
    if attrs["Wearing_Necktie"]:
        accessories.append("a necktie")
    if attrs["Heavy_Makeup"]:
        accessories.append("heavy makeup")

    # Mélanger tout ça proprement
    details = hair + face + facial_hair + accessories
    if details:
        base += " with " + ", ".join(details[:-1])
        if len(details) > 1:
            base += ", and " + details[-1]
        else:
            base += details[0]

    base += "."
    return base

# Dataset personnalisé
class CelebADatasetWithPrompts(CelebA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attr_names = self.attr_names

    def __getitem__(self, index):
        img, attr = super().__getitem__(index)
        prompt = create_prompt(attr, self.attr_names)
        return img, prompt

# Chargement des datasets
train_dataset = CelebADatasetWithPrompts(root="./data/", split='train', download=True, transform=transform)
val_dataset = CelebADatasetWithPrompts(root="./data/", split='test', download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)


# pour obtenir des tokens à partir des prompts
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

