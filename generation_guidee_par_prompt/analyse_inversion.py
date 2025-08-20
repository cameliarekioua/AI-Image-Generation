import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer
from clip_a_la_main import TextToEmbdNetwork
import clip

# Multitudes de prompts
color_only = [
    ("un carré rouge au milieu", "un carré bleu au milieu"),
    ("un disque vert en haut à droite", "un disque jaune en haut à droite"),
    ("un losange violet en bas à gauche", "un losange orange en bas à gauche"),
    ("un cercle bleu en haut à gauche", "un cercle rouge en haut à gauche"),
]

size_only = [
    ("un petit carré bleu au milieu", "un grand carré bleu au milieu"),
    ("un disque moyen rouge en bas à droite", "un disque petit rouge en bas à droite"),
    ("un cercle grand vert en haut à gauche", "un cercle petit vert en haut à gauche"),
    ("un losange petit violet au milieu", "un losange grand violet au milieu"),
]

position_only = [
    ("un carré rouge au milieu", "un carré rouge en haut à gauche"),
    ("un disque bleu en bas à droite", "un disque bleu en haut à droite"),
    ("un cercle vert en bas à gauche", "un cercle vert en haut à droite"),
    ("un losange jaune en haut à droite", "un losange jaune en bas à gauche"),
]

shape_and_position = [
    ("un disque rouge au milieu", "un carré rouge en haut à gauche"),
    ("un cercle bleu en bas à droite", "un losange bleu en haut à droite"),
    ("un carré vert en haut à gauche", "un disque vert en bas à droite"),
    ("un losange jaune au milieu", "un cercle jaune en haut à droite"),
]

shape_only = [
    ("un cercle rouge moyen en haut à gauche", "un disque rouge moyen en haut à gauche"),
    ("un carré bleu petit en bas à droite", "un losange bleu petit en bas à droite"),
    ("un losange vert grand en bas à gauche", "un cercle vert grand en bas à gauche"),
    ("un disque jaune moyen au milieu", "un carré jaune moyen au milieu"),
]

color_and_position = [
    ("un disque rouge en haut à gauche", "un disque bleu en bas à droite"),
    ("un carré vert au milieu", "un carré jaune en haut à gauche"),
    ("un cercle violet en bas à gauche", "un cercle rouge en haut à droite"),
    ("un losange bleu en haut à droite", "un losange rouge en bas à gauche"),
]

all_different = [
    ("un petit disque rouge en bas à gauche", "un grand carré bleu en haut à droite"),
    ("un losange violet en haut à gauche", "un cercle jaune en bas à droite"),
    ("un carré moyen vert au milieu", "un disque petit orange en haut à droite"),
    ("un cercle rouge en haut à gauche", "un losange bleu en bas à droite"),
]

many_objects = [
    ("un cercle rouge en haut à gauche et un carré bleu en bas à droite", "un carré bleu en bas à droite et un cercle rouge en haut à gauche"),
    ("un losange vert en bas à gauche, un disque jaune en haut à gauche, un carré jaune au milieu et un disque vert en bas à droite", "un disque jaune en haut et un losange vert en bas"),
    ("un cercle rouge au milieu et un carré bleu en bas à droite", "un cercle bleu au milieu et un carré rouge en bas à droite"),
    ("un cercle rouge en haut à gauche et un carré bleu en bas à droite", "un carré rouge en haut à gauche et un cercle bleu en bas à droite"),
    ("un cercle rouge en haut à gauche et un disque vert en bas à droite", "un cercle rouge en bas à droite et un disque vert en haut à gauche"),
]

categories = {
    "color_only": color_only,
    "shape_only": shape_only,
    "size_only": size_only,
    "position_only": position_only,
    "shape_and_position": shape_and_position,
    "color_and_position": color_and_position,
    "many_objects": many_objects,
    "all_different": all_different,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialise tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Charge modèle perso unique (checkpoint 18)
model_2 = TextToEmbdNetwork().to(device)
checkpoint_2 = torch.load("clip_checkpoint_epoch_18.pt", map_location=device)
model_2.load_state_dict(checkpoint_2["model_text_to_embd_state_dict"])
model_2.eval()

# Charge modèle OpenAI CLIP
model_openai, preprocess = clip.load("ViT-B/32", device=device)
model_openai.eval()

def get_embedding_perso(prompt, model):
    with torch.no_grad():
        tokens = tokenizer(prompt, padding="max_length", truncation=True, max_length=77, return_tensors="pt")["input_ids"]
        tokens = tokens.to(device).to(torch.float32)
        emb = model(tokens)
        emb = F.normalize(emb, p=2, dim=-1)
    return emb

def get_embedding_openai(prompt):
    tokens = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        emb = model_openai.encode_text(tokens)
        emb = F.normalize(emb, p=2, dim=-1)
    return emb

def cosine_similarity(tensor1, tensor2):
    return (tensor1 @ tensor2.T).item()

for cat, pairs in categories.items():
    print(f"\n=== Catégorie : {cat} ===")
    for p1, p2 in pairs:
        emb1_openai = get_embedding_openai(p1)
        emb2_openai = get_embedding_openai(p2)
        sim_openai = cosine_similarity(emb1_openai, emb2_openai)

        emb1_perso = get_embedding_perso(p1, model_2)
        emb2_perso = get_embedding_perso(p2, model_2)
        sim_perso = cosine_similarity(emb1_perso, emb2_perso)

        print(f"Prompts: \"{p1}\" <-> \"{p2}\"")
        print(f" OpenAI CLIP similarity : {sim_openai:.4f}")
        print(f" Clip à la main similarity : {sim_perso:.4f}\n")
