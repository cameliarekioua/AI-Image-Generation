import io
import base64
import torch
import torch.nn.functional as F
from flask import Flask, request, render_template
from PIL import Image

from DiT import DiT
from config import config
from fonctions_diffusion import sample
from vae import VAE
import clip

# Setup
device = config["device"]
mean = config["mean"]
std = config["std"]

# Load models
model = DiT().to(device)
checkpoint_dit = torch.load(f"./generation_guidee_par_prompt/sauvegardes_entrainement/dit_checkpoint_epoch_13.pt", map_location='cpu')
model.load_state_dict(checkpoint_dit['model_state_dict'])
model.eval()

vae = VAE().to(device)
checkpoint_vae = torch.load(f"./generation_guidee_par_prompt/sauvegardes_entrainement/vae_checkpoint_epoch_15.pt", map_location='cpu')
vae.load_state_dict(checkpoint_vae['model_state_dict'])
vae.eval()

clip_model, preprocess = clip.load("ViT-B/32", device=device)

def get_embedding(prompt):
    with torch.no_grad():
        text_tokens = clip.tokenize(prompt).to(device)
        text_embedding = clip_model.encode_text(text_tokens)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    return text_embedding.to(torch.float32)

def generate_image(prompt):
    labels = get_embedding([prompt])
    sampled_imgs = sample(1, model, labels)
    sampled_imgs = vae.decoder(sampled_imgs)

    img_tensor = torch.clamp(
        torch.sigmoid(sampled_imgs[0]).detach().cpu() * std.view(-1, 1, 1) + mean.view(-1, 1, 1),
        0, 1
    ).permute(1, 2, 0)

    img = Image.fromarray((img_tensor.numpy() * 255).astype("uint8"))

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_b64

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    image_data = None
    prompt = ""
    if request.method == "POST":
        prompt = request.form["prompt"]
        try:
            image_data = generate_image(prompt)
        except Exception as e:
            return f"Erreur: {e}"
    return render_template("index.html", image_data=image_data, prompt=prompt)

if __name__ == "__main__":
    app.run(port=5001, debug=True)
