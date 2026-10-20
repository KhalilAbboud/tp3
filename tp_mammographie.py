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
    print(f"--- Analyse Statistiques de {title} ---")
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
def add_gaussian_noise(img, mean=0, sigma=25, seed=42):
    """Ajoute un bruit gaussien à l'image."""
    np.random.seed(seed)
    gauss = np.random.normal(mean, sigma, img.shape)
    noisy = img.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(img, amount=0.05, seed=42):
    """Ajoute un bruit impulsionnel (sel et poivre)."""
    np.random.seed(seed)
    noisy = img.copy()
    num_salt = np.ceil(amount * img.size * 0.5)
    num_pepper = np.ceil(amount * img.size * 0.5)

    # Sel (pixels blancs)
    coords = [
        np.random.randint(0, i - 1, int(num_salt)) for i in img.shape
    ]  #
    noisy[tuple(coords)] = 255

    # Poivre (pixels noirs)
    coords = [
        np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape
    ]  #
    noisy[tuple(coords)] = 0
    return noisy


def mean_filter(img, ksize=3):
    """Filtre moyenneur fait maison."""
    pad = ksize // 2
    padded = np.pad(img, pad, mode="reflect")
    output = np.zeros_like(img, dtype=np.float32)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i : i + ksize, j : j + ksize]
            output[i, j] = np.mean(region)
    return output.astype(np.uint8)


def gaussian_filter(img, ksize=3, sigma=1.0):
    """Filtre gaussien fait maison."""
    pad = ksize // 2
    ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    kernel /= kernel.sum()

    padded = np.pad(img, pad, mode="reflect")
    output = np.zeros_like(img, dtype=np.float32)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i : i + ksize, j : j + ksize]
            output[i, j] = np.sum(region * kernel)
    return output.astype(np.uint8)


def median_filter(img, ksize=3):
    """Filtre médian fait maison."""
    pad = ksize // 2
    padded = np.pad(img, pad, mode="reflect")
    output = np.zeros_like(img, dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i : i + ksize, j : j + ksize]
            output[i, j] = np.median(region)
    return output

"""
filter mode:
[0,0,0,0
0,255,0,0
0,0,0,0
0,0,0,0] 
transforms to
[28.33,28.33,28.33,0
28.33,28.33,28.33,0
28.33,28.33,28.33,0
0,0,0,0]

filter guassian:
[15.94,31.88,15.94,0
31.88,63.75,31.88,0
15.94,31.88,15.94,0
0,0,0,0]
"""

#partie 4: binarisation de l'image
def binarize_image(img, threshold):
    """Binarisation de l'image selon un seuil donné."""
    binary = np.where(img >= threshold, 255, 0).astype(np.uint8)
    return binary

def erosion(img, kernel):
    """Érosion morphologique."""
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = np.pad(
        img,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=0,
    )
    output = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i : i + k_h, j : j + k_w]
            if np.all(region[kernel == 1] == 255):
                output[i, j] = 255
    return output

def dilation(img, kernel):
    """Dilatation morphologique."""
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = np.pad(
        img,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=0,
    )
    output = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i : i + k_h, j : j + k_w]
            if np.any(region[kernel == 1] == 255):
                output[i, j] = 255
    return output

def opening(img, kernel):
    """Ouverture morphologique (Érosion puis Dilatation)."""
    return dilation(erosion(img, kernel), kernel)


def closing(img, kernel):
    """Fermeture morphologique (Dilatation puis Érosion)."""
    return erosion(dilation(img, kernel), kernel)



#partie 5: Détection de bords et rehaussement
def sobel_edges(img):
    """Détection de contours par filtre de Sobel (Magnitude du gradient)."""
    # Calcul des gradients horizontal (dx) et vertical (dy)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Calcul de la magnitude du gradient : sqrt(Gx^2 + Gy^2)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalisation sur l'intervalle [0, 255]
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def laplacian_filter(img):
    """Filtrage laplacien pour accentuer les variations locales de contraste."""
    # Application du filtre Laplacien (dérivée seconde)[cite: 1]
    lap = cv2.Laplacian(img, cv2.CV_64F)

    # Valeur absolue et normalisation[cite: 1]
    return np.clip(np.abs(lap), 0, 255).astype(np.uint8)


#partie 6 : evaluation quantative
def mse(img1, img2):
    """Calcule l'Erreur Quadratique Moyenne (Mean Squared Error)."""
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)


def psnr(img1, img2):
    """Calcule le Rapport Signal sur Bruit de Crête (Peak Signal-to-Noise Ratio)."""
    err = mse(img1, img2)
    if err == 0:
        return float("inf")
    return 10.0 * np.log10((255.0**2) / err)


#partie 7: egalisation captive local




"""===================================================================================================="""


img = load_image("mammogram.png")
analyze_histogram(img)

# partie 2 resultat
dynamic_img = dynamic_expansion(img)
analyze_histogram(dynamic_img, title="Image après Expansion de Dynamique")
equalized_img = histogram_equalization(img)
analyze_histogram(equalized_img, title="Image après Égalisation d'Histograme")
specified_img = histogram_specification(img, dynamic_img)
analyze_histogram(specified_img, title="Image après Spécification d'Histograme")


# partie 3 resultats
guassian_noisy_img = add_gaussian_noise(img)
analyze_histogram(guassian_noisy_img, title="Image avec Bruit Gaussien")
salt_pepper_noisy_img = add_salt_and_pepper_noise(img)
analyze_histogram(salt_pepper_noisy_img, title="Image avec Bruit Sel et Poivre")
mean_filtered_img = mean_filter(guassian_noisy_img)
analyze_histogram(mean_filtered_img, title="Image après Filtre Moyenneur")
guassian_filtered_img = gaussian_filter(guassian_noisy_img)
analyze_histogram(guassian_filtered_img, title="Image après Filtre Gaussien")
median_filtered_img = median_filter(salt_pepper_noisy_img)
analyze_histogram(median_filtered_img, title="Image après Filtre Médian")


# partie 4 results
binarized_img = binarize_image(equalized_img, threshold=128)
analyze_histogram(binarized_img, title="Image après Binarisation")
erosion_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
eroded_img = erosion(binarized_img, erosion_kernel)
analyze_histogram(eroded_img, title="Image après Érosion")
dilated_img = dilation(binarized_img, erosion_kernel)
analyze_histogram(dilated_img, title="Image après Dilatation")

opened_img = opening(binarized_img, erosion_kernel)
analyze_histogram(opened_img, title="Image après Ouverture")
closed_img = closing(binarized_img, erosion_kernel)
analyze_histogram(closed_img, title="Image après Fermeture")



#partie5 results:
sobel_img = sobel_edges(equalized_img)
analyze_histogram(sobel_img, title="Image après Détection de Contours (Sobel)")
laplacian_img = laplacian_filter(equalized_img)
analyze_histogram(laplacian_img, title="Image après Filtrage Laplacien")



#partie 6 results: 
mse_value = mse(img, equalized_img)
psnr_value = psnr(img, equalized_img)



#partie 7 results:
