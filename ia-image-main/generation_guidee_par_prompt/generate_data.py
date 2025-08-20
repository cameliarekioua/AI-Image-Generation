import random
import os
from PIL import Image, ImageDraw

CHIFFRES_EN_MOTS = {
    1: "un",
    2: "deux",
    3: "trois",
    4: "quatre",
    5: "cinq"
}

FORMES = ["cercle", "disque", "carré", "losange", "triangle"]

POSITIONS = {
    "en haut à gauche": (20, 20),
    "en haut à droite": (176, 20),
    "en bas à gauche": (20, 176),
    "en bas à droite": (176, 176),
    "au milieu": (98, 98)
}

TAILLES = {
    "petit": 30,
    "moyen": 50,
    "grand": 70
}

COULEURS = {
    "rouge": (255, 0, 0),
    "bleu": (0, 0, 255),
    "vert": (0, 255, 0),
    "jaune": (255, 255, 0),
    "violet": (128, 0, 128),
    "noir": (0, 0, 0)
}

os.makedirs("data/formes_geometriques/images", exist_ok=True)
os.makedirs("data/formes_geometriques/prompts", exist_ok=True)

def dessiner_forme(draw, forme, centre, taille, couleur):
    x, y = centre
    w = taille
    h = taille
    bbox = (x, y, x + w, y + h)

    if forme == "cercle":
        draw.ellipse(bbox, outline=couleur, width=2)
    elif forme == "disque":
        draw.ellipse(bbox, fill=couleur)
    elif forme == "carré":
        draw.rectangle(bbox, outline=couleur, width=2)
    elif forme == "losange":
        cx = x + w / 2
        cy = y + h / 2
        dx = w * 0.4
        dy = h * 0.6
        points = [(cx, cy - dy), (cx + dx, cy), (cx, cy + dy), (cx - dx, cy)]
        draw.polygon(points, outline=couleur, width=2)
    elif forme == "triangle":
        cx = x + w / 2
        cy = y + h / 2
        dx = w / 2
        dy = h / 2
        points = [(cx, cy - dy), (cx - dx, cy + dy), (cx + dx, cy + dy)]
        draw.polygon(points, outline=couleur, width=2)

def format_prompt(forme_position_list):
    parts = []
    for forme, position, taille_nom, couleur_nom in forme_position_list:
        mot_nombre = CHIFFRES_EN_MOTS[1]  # toujours "un"
        parts.append(f"{mot_nombre} {forme} {taille_nom} {couleur_nom} {position}")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " et " + parts[-1]

def generer_image(index, taille_image=256):
    image = Image.new("RGB", (taille_image, taille_image), "white")
    draw = ImageDraw.Draw(image)

    positions = list(POSITIONS.items())
    random.shuffle(positions)
    nb_formes = random.randint(1, 5)
    formes_choisies = random.sample(FORMES * 2, nb_formes)
    forme_position_list = []

    for i in range(nb_formes):
        forme = formes_choisies[i]
        pos_nom, (x, y) = positions[i]
        taille_nom = random.choice(list(TAILLES.keys()))
        couleur_nom = random.choice(list(COULEURS.keys()))
        taille_px = TAILLES[taille_nom]
        couleur_rgb = COULEURS[couleur_nom]

        dessiner_forme(draw, forme, (x, y), taille_px, couleur_rgb)
        forme_position_list.append((forme, pos_nom, taille_nom, couleur_nom))

    prompt = format_prompt(forme_position_list)
    image.save(f"data/formes_geometriques/images_test/forme_{index}.png")

    with open(f"data/formes_geometriques/prompts_test/forme_{index}.txt", "w") as f_txt:
        f_txt.write(f"{prompt}\n")

    print(f"{index} : {prompt}")
    return prompt

def generer_dataset(nb_images):
    for i in range(nb_images):
        generer_image(i)

generer_dataset(5000)
