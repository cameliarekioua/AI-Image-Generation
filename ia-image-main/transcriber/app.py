from flask import Flask, request, render_template, jsonify
import requests
import os
import subprocess
import io
import base64
import torch
from PIL import Image
from werkzeug.utils import secure_filename
from tempfile import NamedTemporaryFile

# Image generation imports
from DiT import DiT
from config import config
from fonctions_diffusion import sample
from vae import VAE
import clip

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Whisper API configuration
WHISPER_API_KEY = 'sk-ok5Do040vHjWVctSsEeuoBh1p6iMTXZoaefzLf6cMe0='
WHISPER_SERVER_URL = 'https://image.rezel.net/whispercpp/inference'

# Setup for image generation
device = config["device"]
mean = config["mean"]
std = config["std"]

# Load models
model = DiT().to(device)
checkpoint_dit = torch.load("./transcriber/sauvegardes_entrainement/dit_checkpoint_epoch_13.pt", map_location='cpu')
model.load_state_dict(checkpoint_dit['model_state_dict'])
model.eval()

vae = VAE().to(device)
checkpoint_vae = torch.load("./transcriber/sauvegardes_entrainement/vae_checkpoint_epoch_15.pt", map_location='cpu')
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        audio_file = request.files.get('file')
        if not audio_file:
            return jsonify({"error": "No audio file provided"}), 400

        with NamedTemporaryFile(delete=True, suffix=".webm") as temp_input:
            audio_file.save(temp_input.name)

            with NamedTemporaryFile(delete=True, suffix=".wav") as temp_wav:
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i", temp_input.name,
                    "-ar", "16000",
                    "-ac", "1",
                    temp_wav.name
                ]
                subprocess.run(ffmpeg_cmd, check=True)

                with open(temp_wav.name, "rb") as wav_file:
                    files = {'file': ('recording.wav', wav_file, 'audio/wav')}
                    data = {
                        'temperature': '0.0',
                        'temperature_inc': '0.2',
                        'response_format': 'json'
                    }
                    headers = {
                        'Authorization': f'Bearer {WHISPER_API_KEY}'
                    }

                    response = requests.post(
                        WHISPER_SERVER_URL,
                        files=files,
                        data=data,
                        headers=headers,
                        timeout=60
                    )

                if response.status_code != 200:
                    return jsonify({
                        "error": f"Whisper server returned status {response.status_code}",
                        "details": response.text
                    }), 502

                transcription_result = response.json()
                prompt_text = transcription_result.get("text", "").strip()

                if not prompt_text:
                    return jsonify({"error": "Transcription succeeded but no text found"}), 500

                # Generate image from prompt
                image_b64 = generate_image(prompt_text)

                return jsonify({
                    "text": prompt_text,
                    "image_base64": image_b64
                })

    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Audio conversion failed", "details": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5003, debug=True)
