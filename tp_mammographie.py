import os
import heapq
from collections import Counter
import numpy as np
import cv2
import matplotlib.pyplot as plt

OUTPUT_DIR = "image_resultat"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# partie 1: chargement et analyse de l'image
def load_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found: {filename}")
    return img

def show_image(img, title="Image"):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

def analyze_histogram(img):
    """Affiche les statistiques basiques et l'histogramme de l'image."""
    print(f"--- Analyse Statistiques ---")
    print(f"Min: {img.min()}, Max: {img.max()}, Moyenne: {img.mean():.2f}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Image d'origine")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.hist(img.ravel(), bins=256, range=[0, 256], color="black")
    plt.title("Histogramme")
    plt.xlabel("Niveaux de gris")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "partie1_histogramme.png"))
    plt.show()



# partie 2: segmentation de l'image








"""===================================================================================================="""


img = load_image("mammogram.png")
show_image(img, title="Image Mammographie")
analyze_histogram(img)