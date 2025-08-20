# Artishow : Modèles d'IA générative pour la création d'images simples

## Objectif

Ce projet vise à explorer les modèles d’IA générative pour créer des images et il inclut une interface web permettant deux types de génération :

- À partir de prompts textuels (ou de voix convertie en texte) : on peut créer des images de formes géométriques (carrés, cerles, triangles, losanges, disques) à partir de descriptions textuelles simples où on peut adapter leurs tailles, couleurs et positions.
- De manière non guidée  : on peut générer aléatoirement des visages.

## Architecture 
- `generation_coco` : génération guidée par prompts à partir des images du jeu de données generation_coco (vae+dit)
- `generation_guidee_par_prompt` : génération guidée par prompts avec unet et vae+dit à partir du jeu de données des formes géométriques
- `génération_non_guidée` : génération avec le jeu de données celebA (unet et vae+dit)
- `génération_visages_guidée` : génération guidée par prompts sur le jeu de données celebA (vae+dit) ( les prompts ont été générés manuellement à partir ces caractéristiques de celebA)
- `notebooks` : notebooks pédagogiques expliquant des concepts théoriques utilisés
- `implementations` : implementation de modèles sur le jeu de données MNIST
- `main_website` : le site web contenant les notebooks sous format html (`notebooks_html) et différentes générations 
- `modele_clip` : notebook sur le modele clip de openai avec documentation
- `transcriber` : modèle speech to text de IBM sur un site web
- `ressources.txt` : de la documentation sur plusieurs modèles / techniques utilisés
- `Suivi.md` : suivi des contributions de chaque membre de l'équipe tout au long du semestre
- `ia-image` : rapport sur les enjeux environnementaux et sociétaux du projet
- `planning.pdf` : le planning pour le semestre
- `poster-ia-image` : poster du projet (format A1 portait)
- `slides_ia_image.pdf` : slides de l'évaluation intermédiaire
- `requirements.txt` : bibliothèques requises pour le projet

## Interface Web

L'interface web comprend :
- de la génération par prompt (formes géométriques)
- de la génération non guidée (visages)
- de la génération par commande vocale (speech-to-text)
- des notebooks pédagogiques.



## Contenu pédagogique

Le projet est aussi un support d’apprentissage. Nous avons documenté :
- Le fonctionnement des réseaux de neurones et de la backpropagation
- Les principes des GANs (Generative Adversarial Networks)
- Les modèles de diffusion (inspirés de Stable Diffusion)
- Les architectures UNet, DiT (Diffusion Transformer), CLIP, VAE, etc.
- L’usage de PyTorch, avec des notebooks interactifs

Tout est accessible dans le dossier `notebooks/`, ou `ressources.txt` ou via l’interface web.

## Installation


**Serveur Web :**

Attention : Le programme requiert énormément de ram ~16gb par programme. Il est recommandé de modifier start_servers.sh pour ne lancer que la page principal et la génération que vous voulez.
 
```bash

git clone git@gitlab.enst.fr:proj104/2024-2025/ia-image.git
cd ia-image
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
chmod +x start_servers.sh
chmod +x stop_servers.sh
./start_servers.sh
```

Le serveur est accessible à http://localhost:5000

Pour arreter les serveurs :
```bash
./stop_servers.sh
```


Vous devez également générer un dossier data avec les formes géométriques et les prompts associés : 
```bash
cd generation_guidee_par_prompt
python3 generate_data.py
```
## Limites

- Difficultés en cas de prompts contenant plusieurs formes : risque de permutation des attributs (couleur, taille, position).


##  Perspectives

- Améliorer la gestion des permutations (via meilleure séparation des embeddings).


## Encadrants :
-  M. Pascal BIANCHI
-  Mme. Charlotte LACLAU

## Équipe
- Mahir ASSANE-MOUSSABAY
- Sarah DAHER
- Charles HERR
- Camelia REKIOUA

