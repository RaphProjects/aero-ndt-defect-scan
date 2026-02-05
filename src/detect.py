import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import WindowsPath


try:
    # Quand exécuté depuis la racine (Streamlit, tests)
    from src.autoencoder import Autoencoder
    from src.dataset import CleanPatchesDataset
    from src.annotation_parser import extract_defect_boxes
    from src.patch_extractor import get_jpg_path_from_xml
    from src.patch_extractor import is_patch_clean
except ModuleNotFoundError:
    # Quand exécuté directement depuis src/
    from autoencoder import Autoencoder
    from dataset import CleanPatchesDataset
    from annotation_parser import extract_defect_boxes
    from patch_extractor import get_jpg_path_from_xml
    from patch_extractor import is_patch_clean


def load_model(model_path):
    """Charge le modèle entraîné."""
    model = Autoencoder(HasAttention=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Mode évaluation (pas de dropout, etc.)
    return model

def preprocess_image(img):
    """
    Charge et prépare une image pour l'inférence.
    Retourne le tensor ET l'image PIL originale.
    """
    # 1. Charger l'image en niveaux de gris
    img = img.convert("L")

    # 2. Redimensionner en 50x50 si nécessaire
    if img.size != (50, 50):
        img = img.resize((50, 50))
    
    # 3. Convertir en tensor normalisé
    img_array = np.array(img)
    img_tensor = torch.from_numpy(img_array)
    img_tensor = img_tensor / 255.0
    if img_tensor.shape[0] != 1:
        img_tensor = img_tensor.unsqueeze(0)
        if img_tensor.shape[1] != 1:
            img_tensor = img_tensor.unsqueeze(0)

    # 4. Ajouter la dimension batch : (1, 1, 50, 50)
    return img_tensor,img

def compute_error_map(original, reconstructed):
    """
    Calcule la carte d'erreur entre l'original et la reconstruction.
    """
    # Différence absolue ou MSE pixel par pixel

    return np.abs(original.squeeze().numpy() - reconstructed.squeeze().numpy())

def detect_anomaly(model, image, threshold=0.05):
    """
    Détecte si une image contient une anomalie.
    
    Returns:
        - is_anomaly: bool
        - error_map: numpy array pour la heatmap
        - reconstruction: image reconstruite
    """
    model.eval()
    img_tensor,img = preprocess_image(image)
    with torch.no_grad():
        reconstruction = model(img_tensor)
    error_map = compute_error_map(img_tensor, reconstruction)
    is_anomaly = error_map.max() > threshold
    return is_anomaly, error_map, reconstruction

def cut_image(img):
    img = Image.open(img)
    cropped_imgs = []
    for i in range(0, 200, 50):
        for j in range(0, 200, 50):
            cropped_img = img.crop((i, j, i+50, j+50))
            cropped_imgs.append(cropped_img)
    return cropped_imgs


def detect_anomalies_test(model, annotation_path, imgFolder,threshold=0.05):

    n_clean_patches_clean_pred = 0
    n_defect_patches_defect_pred = 0

    n_clean_patches_defect_pred = 0
    n_defect_patches_clean_pred = 0

    defect_boxes = extract_defect_boxes(annotation_path)

    imgPath = get_jpg_path_from_xml(annotation_path,imgFolder)
    img = Image.open(imgPath).convert("L")

    for i in range(0, 200, 50):
        for j in range(0, 200, 50):
            patch = img.crop((i, j, i+50, j+50))
            patchbox = {"xmin": i, "ymin": j, "xmax": i+50, "ymax": j+50}
            hasDefect = not is_patch_clean(patchbox,defect_boxes)
            patch_anomaly, error_map, _ = detect_anomaly(model, patch, threshold)

            if patch_anomaly and hasDefect:
                n_defect_patches_defect_pred += 1
            elif patch_anomaly and not hasDefect:
                n_clean_patches_defect_pred += 1
            elif not patch_anomaly and hasDefect:
                n_defect_patches_clean_pred += 1
            elif not patch_anomaly and not hasDefect:
                n_clean_patches_clean_pred += 1


    return n_clean_patches_clean_pred, n_defect_patches_defect_pred, n_clean_patches_defect_pred, n_defect_patches_clean_pred


def detect_anomaly_full_image(model, image_path_or_image, threshold=0.05):
    """
    Analyse une image 200x200 complète.
    
    Returns:
        - is_anomaly: bool (True si au moins un patch est anormal)
        - full_heatmap: numpy array 200x200
    """
    if isinstance(image_path_or_image, WindowsPath):
        img = Image.open(image_path_or_image).convert("L")
    else:
        img = image_path_or_image
    full_heatmap = np.zeros((200, 200))
    reconstruction_full = np.zeros((200, 200))
    is_anomaly = False
    
    for i in range(0, 200, 50):
        for j in range(0, 200, 50):
            # Découper le patch
            patch = img.crop((i, j, i+50, j+50))
            
            # Analyser
            patch_anomaly, error_map, reconstruction = detect_anomaly(model, patch, threshold)
            
            # Placer dans la reconstruction complète
            reconstruction_full[j:j+50, i:i+50] = reconstruction
            # Placer dans la heatmap complète
            full_heatmap[j:j+50, i:i+50] = error_map
            
            if patch_anomaly:
                is_anomaly = True
    
    return is_anomaly, full_heatmap, reconstruction_full

def create_defect_mask(annotation_path, image_size=200):
    """
    Crée un masque binaire 200x200 où 1 = pixel défectueux.
    
    Args:
        annotation_path: chemin vers le fichier XML
        image_size: taille de l'image (200)
    
    Returns:
        numpy array (200, 200) avec 0 ou 1
    """
    # 1. Parser les bounding boxes (ta fonction extract_defect_boxes)
    boxes = extract_defect_boxes(annotation_path)
    # 2. Créer un masque de zéros
    mask = np.zeros((image_size, image_size))
    # 3. Remplir les rectangles avec des 1
    for box in boxes:
        mask[box["ymin"]:box["ymax"], box["xmin"]:box["xmax"]] = 1
    return mask

def evaluate_correlation(model, images_folder, annotations_folder, n_samples=10000):
    """
    Évalue la corrélation entre erreur de reconstruction et présence de défaut.
    
    Returns:
        - correlation: coefficient de Pearson 
        - all_y_true: liste des labels (0/1)
        - all_y_pred: liste des erreurs
    """
    all_y_true = []
    all_y_pred = []
    n_img_tested = 0
    while n_img_tested < n_samples and n_img_tested < 240*len(os.listdir(images_folder)):
        try:
            annotation_file = os.listdir(annotations_folder)[n_img_tested]
        except IndexError:
            break
        if annotation_file.endswith(".xml"):
            full_path = os.path.join(annotations_folder, annotation_file)
            defect_boxes = extract_defect_boxes(full_path)
            mask = create_defect_mask(full_path)
            is_anomaly, error_map = detect_anomaly_full_image(model, get_jpg_path_from_xml(full_path,images_folder))
            all_y_true.append(mask)
            all_y_pred.append(error_map)
            n_img_tested += 1
    
    # 4. Calculer la corrélation
    all_y_true_flat = np.concatenate([m.flatten() for m in all_y_true])
    all_y_pred_flat = np.concatenate([m.flatten() for m in all_y_pred])
    correlation = np.corrcoef(all_y_true_flat, all_y_pred_flat)[0, 1]
    return correlation, all_y_true_flat, all_y_pred_flat



def evaluate_by_category(model, images_folder, annotations_folder):
    """
    Évalue la corrélation par catégorie de défaut.
    """
    categories = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    
    results = {}
    
    for category in categories:
        all_y_true = []
        all_y_pred = []
        
        # Filtrer les fichiers de cette catégorie
        annotation_files = [f for f in os.listdir(annotations_folder) 
                          if f.startswith(category) and f.endswith(".xml")]
        
        for annotation_file in annotation_files[:240]:  # 240 par catégorie
            full_path = os.path.join(annotations_folder, annotation_file)
            mask = create_defect_mask(full_path)
            image_path = get_jpg_path_from_xml(full_path, images_folder)
            
            _, error_map, __ = detect_anomaly_full_image(model, image_path)
            
            all_y_true.append(mask.flatten())
            all_y_pred.append(error_map.flatten())
        
        # Calculer les métriques
        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        err_sain = y_pred[y_true == 0].mean()
        err_defaut = y_pred[y_true == 1].mean()
        ratio = err_defaut / err_sain
        
        results[category] = {
            "correlation": correlation,
            "err_sain": err_sain,
            "err_defaut": err_defaut,
            "ratio": ratio
        }
        
        print(f"{category:20} | Corr: {correlation:.4f} | Ratio: {ratio:.2f}")
    
    return results

    
def evaluate_patch_classification(model, images_folder, annotations_folder, threshold=0.05, n_samples=1000):

    n_clean_patches_clean_pred_total = 0
    n_defect_patches_defect_pred_total = 0

    n_clean_patches_defect_pred_total = 0
    n_defect_patches_clean_pred_total = 0

    n_img_tested = 0
    while n_img_tested < n_samples*4 and n_img_tested < (240*len(os.listdir(images_folder)))*4:
        try:
            annotation_file = os.listdir(annotations_folder)[n_img_tested]
        except IndexError:
            break
        if annotation_file.endswith(".xml"):
            full_path = os.path.join(annotations_folder, annotation_file)
            image_path = get_jpg_path_from_xml(full_path, images_folder)
            
            (
                n_clean_patches_clean_pred, n_defect_patches_defect_pred,
                n_clean_patches_defect_pred, n_defect_patches_clean_pred
            ) = detect_anomalies_test(model, full_path,images_folder, threshold)

            n_clean_patches_clean_pred_total += n_clean_patches_clean_pred
            n_defect_patches_defect_pred_total += n_defect_patches_defect_pred
            n_clean_patches_defect_pred_total += n_clean_patches_defect_pred            
            n_defect_patches_clean_pred_total += n_defect_patches_clean_pred
        
            n_img_tested += 4
        
    TN = n_clean_patches_clean_pred_total
    TP = n_defect_patches_defect_pred_total
    FN = n_defect_patches_clean_pred_total
    FP = n_clean_patches_defect_pred_total
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1

if __name__ == "__main__":

    model = load_model("models/autoencoder.pt")
    annotations_folder = "data/raw/NEU-DET/validation/annotations"
    images_folder = "data/raw/NEU-DET/validation/images"
    
    print("Évaluation par catégorie...")
    results = evaluate_by_category(model, images_folder, annotations_folder)

    print("Évaluation par patch...")
    results = evaluate_patch_classification(model, images_folder, annotations_folder)
    print(f"Accuracy : {results[0]:.4f}")
    print(f"Precision : {results[1]:.4f}")
    print(f"Recall : {results[2]:.4f}")
    print(f"F1 : {results[3]:.4f}")


