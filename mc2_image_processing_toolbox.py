import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu


def mean_intensity(img):
    assert len(img.shape) == 2, "L'image doit être en niveau de gris"
    n_pix = img.shape[0] * img.shape[1]
    B = 1 / n_pix * np.sum(img)
    return B


def contrast_michelson(img):
    assert len(img.shape) == 2, "L'image doit être en niveau de gris"
    cm = (img.max() - img.min()) / (img.max() + img.min())
    return cm


def contrast_rms(img):
    n_pix = img.shape[0] * img.shape[1]

    assert len(img.shape) == 2, "L'image doit être en niveau de gris"
    b = mean_intensity(img)
    c_rms = np.sqrt(1 / n_pix * np.sum((img - b) ** 2))
    return c_rms


T_rgd2lms = np.array(
    [
        [0.3904725, 0.54990437, 0.00890159],
        [0.07092586, 0.96310739, 0.00135809],
        [0.02314268, 0.12801221, 0.93605194],
    ]
)


def rgb2lms(img_rgb):
    img_lms = img_rgb @ T_rgd2lms.T
    return img_lms


def lms2rgb(img_lms):
    lms2rgb = np.linalg.inv(T_rgd2lms)
    img_rgb = img_lms @ lms2rgb.T
    return img_rgb


def binarisation(img, seuil=None, pick_seuil=False):
    if len(img.shape) == 2:
        print("L'image est en niveau de gris")
    else:
        print("L'image est en couleur :  conversion en niveau de gris")
        img = rgb2gray(img[:, :, :3])

    if seuil is not None:
        dyn_decimal = img.max() - img.min() + 1
        assert (
            seuil < dyn_decimal
        ), "Le seuil doit être inférieur à la dynamique de l'image"
        img_bin = np.zeros_like(img, dtype=np.uint8)
        img_bin[img > seuil] = 255

    elif seuil is None and pick_seuil:
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap="gray")
        plt.title("Image en niveau de gris")
        plt.subplot(1, 2, 2)
        plt.hist(img.ravel(), bins=np.linspace(0, 1, 256))
        plt.title(
            "Histogramme de l'image en niveau de gris : selectionner le seuil de binarisation"
        )
        seuil = plt.ginput(1, show_clicks=True)[0][0]
        plt.close()
        print(f"Seuil sélectionné : {seuil}")
        img_bin = np.zeros_like(img, dtype=np.uint8)
        img_bin[img > seuil] = 255

    else:
        img_bin = None

    s_opt = int(threshold_otsu(img))
    img_bin_opt = np.zeros_like(img, dtype=np.uint8)
    img_bin_opt[img > s_opt] = 255

    return seuil, img_bin, s_opt, img_bin_opt


def etalement_histo(img):
    if len(img.shape) == 2:
        print("L'image est en niveau de gris")
    else:
        print("L'image est en couleur :  conversion en niveau de gris")
        img = rgb2gray(img[:, :, :3])

    dyn_decimal = img.max() - img.min() + 1
    if dyn_decimal != 2:
        img = img / 255

    img_eta = 255 * (img - img.min()) / (img.max() - img.min())
    img_eta = img_eta.astype(np.uint8)

    # Plot transformation
    plt.figure(figsize=(12, 8))
    plt.scatter(img.ravel(), img_eta.ravel())
    plt.xlabel("Intensité originale (a)")
    plt.ylabel("Intensité transformée b=T(a)")
    plt.title("Transformation d'etalement de l'histogramme")

    return img_eta


def hist_cumul(img):
    count, bins_count = np.histogram(img, bins=range(256))
    return np.cumsum(count)


def egalisation_histo(img):
    if len(img.shape) == 2:
        print("L'image est en niveau de gris")
    else:
        print("L'image est en couleur :  conversion en niveau de gris")
        img = rgb2gray(img[:, :, :3])

    dyn_decimal = img.max() - img.min() + 1
    if dyn_decimal <= 2:  # Img in [0, 1]
        img = img * 255
        # img = img.astype(np.uint8)
        Hc = hist_cumul(img.astype(np.uint8))
        Hc = Hc / 255
    else:  # Img in [0, 255]
        Hc = hist_cumul(img)

    # Ugly loop
    Hc_mat = np.empty(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 255:
                Hc_mat[i, j] = Hc[254]
            else:
                Hc_mat[i, j] = Hc[img[i, j]]

    n_pix = img.shape[0] * img.shape[1]

    img_ega = 255 * Hc_mat / n_pix
    img_ega = img_ega.astype(np.uint8)

    # Plot transformation
    plt.figure(figsize=(12, 8))
    plt.scatter(img.ravel(), img_ega.ravel())
    plt.xlabel("Intensité originale (a)")
    plt.ylabel("Intensité transformée b=T(a)")
    plt.title("Transformation d'également de l'histogramme")

    return img_ega


def transform_img(img, T):
    img_t = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_t[i, j] = T[img[i, j]]

    return img_t


def profil_ligne(img, i):
    return img[i, :]


def profil_colonne(img, j):
    return img[:, j]
