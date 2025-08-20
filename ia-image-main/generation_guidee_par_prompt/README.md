Ce dossier contient les 2 méthodes utilisées pour la génération guidée par prompt (VAE + DiT) et UNET:
- `config.py` : les paramètres du code
- `generate_data.py` : le code pour générer notre propre jeu de données de formes géométriques
- `get_data` : charger les données et appliquer clip dessus
- `DiT.py`,`vae.py` et `unet.py`, `clip_a_la_main.py` : l'architecture des modèles implémentés.
- `train_unet.py`, `train_vae.py`, `train_clip_a_la_main.py` et `train.py` : les boucles d'entrainement de ces modèles, 
- `fonctions_diffusion.py` et `fonctions_diff_unet.py` : les fonctions pour la diffusion pour bruiter et débruiter des images
- `analyse_inversion.py` : multitude de prompts pour analyser le comportement de clip dessus suite à un problème dans la génération d'images par prompts(inversion des caractértistiques dans l'image générée par ex. quand on demande un carré bleu et un cercle rouge, les couleurs peuvent être inversés, avoir 2 carrés etc.) (cf `log_analyse_inversion.txt` pour les résultats de similarité obtenus)
- `*.png` : divers images dans l'exécution des codes
