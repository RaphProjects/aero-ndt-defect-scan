try :
    from annotation_parser import extract_defect_boxes
except ModuleNotFoundError:
    from src.annotation_parser import extract_defect_boxes

import os
from pathlib import Path
from PIL import Image
def rectangles_overlap(patch_box, defect_box):
    """
    Vérifie si deux rectangles se chevauchent.
    
    Args:
        patch_box: dict avec xmin, ymin, xmax, ymax
        defect_box: dict avec xmin, ymin, xmax, ymax
        
    Returns:
        True si chevauchement, False sinon
    """
    x_overlap = patch_box["xmax"] >= defect_box["xmin"]  and  defect_box["xmax"] >= patch_box["xmin"]
    
    y_overlap = patch_box["ymax"] >= defect_box["ymin"]  and  defect_box["ymax"] >= patch_box["ymin"]
        
    return x_overlap and y_overlap

def is_patch_clean(patch_box, defect_boxes):
    """
    Vérifie si un patch ne chevauche AUCUN défaut.
    
    Args:
        patch_box: dict avec xmin, ymin, xmax, ymax
        defect_boxes: liste de dicts (résultat de parse_annotation)
        
    Returns:
        True si le patch est sain, False sinon
    """
    for defect_box in defect_boxes:
        if rectangles_overlap(patch_box, defect_box):
            return False
    return True


def count_clean_patches(annotations_folder, patch_size=100):
    """
    Analyse toutes les annotations et compte les patches sains potentiels.
    
    Args:
        annotations_folder: chemin vers le dossier des annotations XML
        patch_size: taille des patches (100 pour du 100x100)
        
    Returns:
        dict avec statistiques {"total_patches": int, "clean_patches": int, "defect_patches": int}
    """
    patches = []
    for i in range(0, 200, patch_size):
        for j in range(0, 200, patch_size):
            patches.append({"xmin": i, "ymin": j, "xmax": i+patch_size, "ymax": j+patch_size})
 
    
    n_clean_patches = 0
    n_defect_patches = 0
    n_patches = 0
    annotations_folder = Path(annotations_folder)
    for annotation_file in os.listdir(annotations_folder):
        if annotation_file.endswith(".xml"):
            full_path = os.path.join(annotations_folder, annotation_file)
            defect_boxes = extract_defect_boxes(full_path)
            for patch in patches:
                if is_patch_clean(patch, defect_boxes):
                    n_clean_patches += 1
                else:
                    n_defect_patches += 1
                n_patches += 1
    return {"total_patches": n_patches, "clean_patches": n_clean_patches, "defect_patches": n_defect_patches}


def get_jpg_path_from_xml(xml_path,images_folder):
    """
    Extrait le chemin vers le JPG d'un fichier XML.
    
    Args:
        xml_path: chemin vers le fichier XML - str
        
    Returns:
        chemin vers le JPG - str
    """
    path = Path(xml_path)
    # ex : data/raw/NEU-DET/train/annotations/crazing_28.xml -> data/raw/NEU-DET/train/images/crazing/crazing_28.jpg
    in_filename = path.stem
    categoryList = in_filename.split("_")[:-1]
    categoryStr = ""
    for i in categoryList:
        categoryStr += i + "_"
    category = categoryStr[:-1]
    jpg_path = Path(images_folder) / category / f"{in_filename}.jpg"
    return jpg_path



def extract_clean_patches(images_folder, annotations_folder, output_folder, patch_size=50):
    """
    Extrait et sauvegarde tous les patches sains.
    
    Args:
        images_folder: chemin vers les images (ex: data/raw/NEU-DET/train/images)
        annotations_folder: chemin vers les annotations XML
        output_folder: où sauvegarder les patches (ex: data/processed/clean_patches)
        patch_size: taille des patches
    """
    patches = []
    for i in range(0, 200, patch_size):
        for j in range(0, 200, patch_size):
            patches.append({"xmin": i, "ymin": j, "xmax": i+patch_size, "ymax": j+patch_size})
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    annotations_folder = Path(annotations_folder)
    for annotation_file in os.listdir(annotations_folder):
        if annotation_file.endswith(".xml"):
            full_path = os.path.join(annotations_folder, annotation_file)
            defect_boxes = extract_defect_boxes(full_path)
            img = Image.open(get_jpg_path_from_xml(full_path,images_folder))
            for patch,i in zip(patches, range(len(patches))):
                cropped_img = img.crop((patch["xmin"], patch["ymin"], patch["xmax"], patch["ymax"]))
                if is_patch_clean(patch, defect_boxes):
                    cropped_img.save(os.path.join(output_folder, annotation_file.replace(".xml", f"_patch_{i}.jpg")))
                

if __name__ == "__main__":
    annotation_path = "data/raw/NEU-DET/train/annotations"
    output_folder = "data/processed/clean_patches"
    images_folder = "data/raw/NEU-DET/train/images"
    extract_clean_patches(images_folder, annotation_path, output_folder, patch_size=50)
        