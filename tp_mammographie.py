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

def analyze_histogram(img, title = "Analyse de l'image"):
    """Affiche les statistiques basiques et l'histogramme de l'image."""
    print(f"--- Analyse Statistiques ---")
    print(f"Min: {img.min()}, Max: {img.max()}, Moyenne: {img.mean():.2f}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.hist(img.ravel(), bins=256, range=[0, 256], color="black")
    plt.title("Histogramme")
    plt.xlabel(f"Niveaux de gris \nMin: {img.min()} | Max: {img.max()} | Moy: {img.mean():.1f}")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "partie1_histogramme.png"))
    plt.show()


#interpretation:
"""
Zones de l'image :Fond (air) : 
    Noir (valeur proche de $0$).   
    Tissu mammaire sain : Nuances de gris moyen.   
    Anomalies (masses, microcalcifications) : Zones très claires ou blanches (valeurs proches de $255$).   
Lecture de l'histogramme :
    Histogramme étroit (resserré) : Image terne et peu contrastée. Les anomalies se confondent avec le tissu sain.   
    Histogramme étalé (large) : Bon contraste. Les détails et la structure du sein se distinguent nettement.
"""


# partie 2: segmentation de l'image
def dynamic_expansion(img, new_min=0, new_max=255):
    """Expansion de dynamique (étirement linéaire de l'histogramme)."""
    i_min, i_max = float(img.min()), float(img.max())
    if i_max == i_min:
        return img.copy()
    expanded = (img - i_min) * ((new_max - new_min) / (i_max - i_min)) + new_min
    return np.clip(expanded, new_min, new_max).astype(np.uint8)

def histogram_equalization(img):
    """Égalisation globale basée sur la fonction de répartition cumulative (CDF)."""
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype("uint8")
    return cdf[img]

def histogram_specification(img, reference):
    """Mise en correspondance d'histogramme (Spécification)."""
    hist_src, _ = np.histogram(img.flatten(), 256, [0, 256])
    hist_ref, _ = np.histogram(reference.flatten(), 256, [0, 256])

    cdf_src = hist_src.cumsum() / hist_src.sum()
    cdf_ref = hist_ref.cumsum() / hist_ref.sum()

    lut = np.zeros(256, dtype=np.uint8)
    for g in range(256):
        diff = np.abs(cdf_src[g] - cdf_ref)
        lut[g] = np.argmin(diff)

    return lut[img]

#interpretation
"""
Expansion de dynamique : 
Étire la plage des niveaux de gris de [min, max] vers [0, 255].   
Résultat : Les zones sombres deviennent légèrement plus claires et l'image gagne un peu de lisibilité sans déformer la répartition originale

Égalisation d'histogramme :
Redistribue les niveaux de gris pour rendre l'histogramme uniforme.   
Résultat : Révèle les détails très sombres du tissu, mais peut surexposer certaines zones claires et amplifier le bruit.

Spécification d'histogramme :
Aligne la distribution des contrastes de l'image sur celle d'une image de référence[cite: 1].
Résultat et qu'il Harmonise le contraste entre plusieurs mammographies d'appareils différents

"""


#partie 3: reduction du bruit



"""===================================================================================================="""


img = load_image("mammogram.png")
analyze_histogram(img)


dynamic_img = dynamic_expansion(img)
analyze_histogram(dynamic_img, title="Image après Expansion de Dynamique")
equalized_img = histogram_equalization(img)
analyze_histogram(equalized_img, title="Image après Égalisation d'Histograme")
specified_img = histogram_specification(img, dynamic_img)
analyze_histogram(specified_img, title="Image après Spécification d'Histograme")


