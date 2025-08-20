# Explication du fonctionnement du modèle CLIP

## Rappel sur les réseaux de neuronnes

Principe général : Un réseau de neuronnes prend en entrée un vecteur et renvoie un autre vecteur.

## Ce que CLIP fait

CLIP est un modèle qui est multimodale. Il prend en entrée à la fois du texte et des images. Il tokenize le texte et l'image (avec l'information des pixels).

Le modèle est entrainé sur un jeu d'image avec label et il doit minimiser le produit scalaire entre le label et l'image.

De manière conceptuelle, il fait en sorte que les label et les images soit proche dans l'espace de sortie.

## Quelques considérations

Pour que le modèle soit pertinent, il faut un bon équilibre entre le nombre de paramètres du modèle et le dataset. En effet, un réseau trop grand peut créer un overfitting et lorsque le modèle verra une nouvelle image il ne saura pas lui attribué un label cohérent (la distance avec les autres images/label est égale). Un réseau trop petit fait en sorte que la classification n'est pas pertinente.

## Utilisation de CLIP

Clip est principalement utilisé dans Stable Diffusion. Il est utilisé pour faire de l'embbeding de prompt. Il permet en effet de faire un conditionnement pertinent.

## Schéma récapitulatif 

<img src="https://miro.medium.com/v2/resize:fit:3662/format:webp/1*tg7akErlMSyCLQxrMtQIYw.png">

## Sources

https://medium.com/data-science/simple-implementation-of-openai-clip-model-a-tutorial-ace6ff01d9f2