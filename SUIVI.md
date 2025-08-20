# Suivi Projet

# Séance 1
## 14/02/2025

Tout le groupe se documente sur la génération d'image.
- **Charles**: étude de la théorie derrière les score-based generative models et de leur lien avec la diffusion, notamment à travers la vidéo https://www.youtube.com/watch?v=B4oHJpEJBAA&ab_channel=Outlier.

- **Sarah, Mahir et Camelia** : apprentissage du fonctionnement théorique des réseaux de neurones avec la playlist suivante : https://www.youtube.com/watch?v=XUFLq6dKQok&list=PLO_fdPEVlfKoanjvTJbIbd9V5d9Pzp8Rw 
    - **Résumé** : initialisation puis boucler (forward propagation z= W.X+b, a (activation) = 1/(1+e^(-z)); sigmoide), cost (calcul de l'erreur) avec logloss, backpropagation (descente de gradient) on calcule les gradients puis on mets a jour les parametres en utilisant un learning rate fixé) -> élargir à la structure de plusieurs couches de neurones. (**TODO** Un jupyter notebook détaillant cette théorie sera implémenté ultérieurement.)


# Séance 2
## 18/02/2025
- **Tous** : 
    - réflexion sur le planning et la répartition des taches ; le planning sera envoyé cette semaine une fois finalisé.
    - Réunion avec les encadrants pour discuter des attendus.
    - Cours par M. Bianchi sur l'apprentissage supervisé notamment la regression logistique et non supervisé Convolutional Neural Network


# Entre Séance 2 et Séance 3
## 20/02/2025
- **Tous** : 
    - Participation à un séminaire sur l'ia "Principles of Large-Scale Foundation Models" par Jhony H. Giraldo (enseignant-chercheur à Télécom Paris)

## 21/02/2025
- Le planning est finalisé et envoyé aux encadrants 

## 23/02/2025 
- **Sarah :** Début du travail sur le notebook1 (classification binaire)

## 27/02/2025
- **Sarah :** Ajout de l'implémentation en python du premier neurone suivant la théorie expliquée dans la partie 1 du notebook 1


# Séance 3
- **Charles**: étude de la théorie derrière les modèles à diffusion, notamment à travers le blog https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ qui décrit les motivations mathématiques derrière le processus de bruitage et débruitage.
- **Camelia :** Ajout de l'explication de réseaux de neurones et de forward/backward propagation dans la partie 3 du notebook 1.
- **Sarah** : Ajout des expressions des dérivées dans le Notebook1 suite aux explications et formules de Camelia.
- **Mahir** : Recherche sur le guidage par prompt sur la génération d'image par diffusion et leurs limites par le papier de recherche https://arxiv.org/pdf/2503.08250 

# Séance 4
- **Charles**: début d'une page web expliquant les modèles à diffusion, l'objectif est le même que pour les notebooks mais sous un format légèrement différent
- **Sarah** : début de la documentation sur les GANs et prise de notes sur de la théorie dans le Notebook2 à travers la ressource suivante : https://youtu.be/r9mfSrRIFnM?si=ubSwmfF5TfzdLcvC (cf notebooks/notebook_GAN.ipynb)


# Séance 5
- **Charles**: poursuite de la page web expliquant le fonctionnement des modèles à diffusion et implémentation d'un modèle à diffusion dans un jupyter notebook
- **Sarah et Camelia** : présentation de la théorie des GANs à l'encadrant et mise à jour du notebook2 en fonction de ses retours 
- **Mahir** : Présentation du modèle de réseaux de neuronnes CLIP nécéssaire à la génération d'image guidée par prompt 


# Séance 6
- **Charles**: étude du flow matching qui généralise les modèles à diffusion, ces modèles ont été popularisé par le papier https://arxiv.org/pdf/2210.02747. Dans l'implémentation le flow matching est très proche des modèles à diffusion mais bruite les images d'une autre façon (interpolation linéaire entre les images et le bruit).
- **Mahir, Charles et Sarah** : réunion en visio avec l'encadrante afin de fixer le projet et de faire un point sur l'avancement.
- **Sarah** : tuto sur pytorch pour pouvoir commcer à implémenter les GANs sur MNIST


# Entre Séance 6 et Séance 7
- **Charles** : Etude des DiT (diffusion transformer) qui remplacent le UNet usuellement utilisé en diffusion par un transformer
- **Sarah** : essaie de faire de la convolution pour compresser les données de MNIST pour les GANs, mais pas nécessaire car les données MNIST ne sont pas très grandes et début de l'implémentation des GANs

# Séance 7
- **Charles**: implémentation d'un DiT (diffusion transformer) (évoqué précédemment) 
- **Sarah** : Finalisation de l'implémentation des GANs sur MNIST avec pytorch.
- **Mahir** : Implémentation du modèle CLIP

# Séance 8

# Entre Séance 8 et Séance 9
- **Mahir** : Reflexion sur l'intégration du guidage par prompt (modèle CLIP) via un embedding dans le modèle à diffusion et exploration de différentes manières de faire de l'embedding (conditionnement via ajout d'une colonne de pixel, modification aléatoire de pixels, ...).
- **Mahir** : Premier test de la technique de cosinus similarity avec le modèle CLIP dans le cadre de la classification d'image
- **Sarah** : documentation sur l'usage des autoencodeurs en général, et début de la doc sur les VAEs en particulier dont on utilisera le décodeur ultérieurement dans le projet pour reconstruire des images à partir du vecteur (prompt -> CLIP) sur lequel on a appliqué la diffusion (cf ressources.txt)
- **Camelia** : Compréhension du code de l'implémentation des gans et de l'usage de pytorch pour.
- **Charles** : Création d'un code pour accéder aux paires d'images et prompts du jeu de données et normalisation des données

# Séance 9
- **Mahir** : Ajout de la documentation expliquant le fonctionnement du modèle CLIP (résumé de la présentation de la séance 5)
- **Sarah** : Suite de la documentation sur la théorie derrière les VAEs (cf ressources.txt)
- **Camelia**: Préparation de la présentation pour le livrable de mi-projet.
- **Charles** : Début du code de la boucle d'entraînement pour le projet de génération par prompt

# Séance 10
- **Sarah, Charles, Mahir et Camelia** : réunion avec l'encadrant et présentation du travail accompli jusqu'à présent, fixation de la date de la soutenance et des attendus, on a aussi à commencer préparer les slides de la présentation
- **Mahir** : Recherche de dataset alternative à COCO comme base pour la génération d'image par prompt. Ex: https://huggingface.co/datasets/shirsh10mall/Image_Captioning_Dataset ou https://huggingface.co/datasets/itshemantkmr/2D_Shape_Image_Datasets
- **Mahir** : Exploration de l'utilisation d'une fonction inverse dans le modèle CLIP pour faire de la génération sans stable diffusion. Lecture d'un papier de recherche à ce sujet (https://arxiv.org/pdf/2403.02580)
- **À faire** : relire et finaliser les notebooks pédagogiques, appliquer le modèle stable diffusion sur MNIST pour s'assurer du fonctionnement du modèle, générer notre propre jeu de données de formes géometriques avec python et l'utiliser pour la génération par prompts.

# Entre Séance 10 et 11
- **Sarah** : remise en forme du notebook GANs (plus lisible) et écriture du code de génération de notre jeu de données sur les formes géométriques, **mais pb** : collisions d'images à régler (cf branche generation_par_prompt : generate_data.py)
- **Charles** : Assemblage des différents composants du code d'entraînement et premiers tests d'entraînement du modèle

# Séance 11
- **Sarah et Camelia** :
    -  régler le code de génération d'images : OK pour 256\*256 mais problème pour le passage vers 32\*32 images flous
    -  préparation des slides de la présentation intermédiaire
- **Mahir** : Remplissage du fichier de présentation pour l'évaluation intermédiaire. Préparation et réfléxion autour des impacts sociaux et environnementaux du projet suite au mail de Marc Jeanmougin du 02/05 
- **Charles** : finalisation du site/notebook sur la diffusion

# Entre Séance 11 et 12
- **Tous** : Présentation des slides récapitulant le travail de la P3 à l'encadrant (évaluation intermédiaire)
- **Mahir** : Documentation sur les differentes manière de faire du Contrastive Representation Learning qui est à la base de l'entrainement des modèles comme CLIP (https://lilianweng.github.io/posts/2021-05-31-contrastive/)

# Séance 12
- **Charles** : étude de la théorie derrière les vaes 
- **Mahir et Sarah** : rédaction du fichier sur l'impact social et environnemental du projet
- **Camelia** : documentation sur les unets pour pouvoir l'implémenter pour la prochaine séance

# Séance 13
Séance audit de projet. 

# Entre Séance 13 et 14 :
- réunion du groupe, lecture du feedback de la séance précédente et prise en compte des points à améliorer : 
    - on va améliorer le dataset d'entraînement (ajout de couleurs, tailles)
    - on va expliciter sur l'interface de génération d'images le type d'images pouvant être générées.
    - ajout de séances avant la fin au cas où il faut corriger et débuguer
    - mettre à disposition les notebooks sur un site.
    
# Séance 14
- **Sarah** : documentation sur le fonctionnements des unets (reseaux de neurones convolutionnels, cf. ressources.txt), test du nouveau dataset avec les couleurs et les tailles OK
- **Camelia** : implémentation du unet et intégration au code pour pouvoir changer de modèle facilement et amélioration du data set d'entraineemnt pour y ajouter la taille et les couleurs des formes.
- **Charles** : début d'implémentation d'un VAE pour remplacer le modèle utilisé précédemment (modèle pré-entrainé, tiré de stable diffusion)
- **Mahir** : Absence (maladie)

# Entre séance 14 et 15 :
- **Charles** : implémentation et entraînement du VAE évoqué précédemment et rajout de ce dernier dans la boucle d'entraînement principale

# Séance 15
- **Tous** : Réunion avec Laclau (CR: on doit commencer le poster, potentiellement implémenter une 2eme fonctionalité (site web ou son ou dictée))
- **Mahir** : Expérimentation de création de site regroupant le modèle et les notebook pour la démo des projets artishow 
- **Sarah et Camelia** : Commencer à réaliser le poster format A1 portrait de notre présentation finale
- **Charles** : résolution d'un bug dans la mesure du temps d'exécution sur GPUs des différentes parties du programme

# Entre séance 15 et 16 :
- **Sarah, Camelia** : Réalisation du poster du projet 
- **Charles** : entraînement du code sur la base de données d'images de visages celebA

# Séance 16 :
- **Mahir** : Création du site web pour la génération d'image par prompt
- **Mahir** : Création du site web pour la génération d'image de visage non guidée
- **Mahir** : Création du site web pour la dictée vocale
- **Sarah** :
    - refaire le unet, retirer le latent, ajouter le conditionnement concaténé à l'image, skip connections, réduire sa taille car trop lourd pour le gpus
    - boucle d'entrainement et tests : pas très concluant, du bruit sur les images.
- **Camelia** : préparation du pitch pour vendredi.
- **Charles** : optimisations gpus (notamment utilisation de float16), entraînement du modèle sur le jeu de données d'images générales coco

# Séance 17
- **Sarah** : 
    - entrainement unet dans la génération non guidée
    -  correction de la boucle d'entrainement dans la generation guidée
- **Charles** : recréation d'une version simplifiée de CLIP pour être mieux adaptée à notre jeu de données de formes géométriques 

# Séance 18
- **Camelia** :  Rédaction du readme.
- **Tous** : Réunion avec notre encadrante pour lui montrer notre avancée et avoir les derniers conseils avant l'évaluation.
- **Sarah** : analyse du probleme d'inversion des figures dans les images (conclusion : pb vient de clip cf log dans generation_guidee_par_prompt), rédaction des readme par dossier du projet

# Séance 19 
- **Camelia** : amélioration du readme, ajout de css pour faire le design de l'interface web.
- **Sarah** : merge les différentes branches du projet.
- **Mahir** : finalisation de l'interface web.
