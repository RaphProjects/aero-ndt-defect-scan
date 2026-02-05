import xml.etree.ElementTree as ET

def extract_defect_boxes(annotation_file):
    """
    Extrait les bounding boxes d'un fichier XML.
    
    Args:
        xml_path: chemin vers le fichier XML
        
    Returns:
        liste de dictionnaires {"xmin": int, "ymin": int, "xmax": int, "ymax": int}
    """
    # 1. Charger le XML
    XMLfile = ET.parse(annotation_file)

    # 2. Trouver tous les <object>
    objects = XMLfile.findall(".//object")

    # 3. Pour chaque object, extraire les coordonnées de <bndbox>
    bounding_boxes = []
    for obj in objects:
        xmin = int(obj.find(".//bndbox/xmin").text)
        ymin = int(obj.find(".//bndbox/ymin").text)
        xmax = int(obj.find(".//bndbox/xmax").text)
        ymax = int(obj.find(".//bndbox/ymax").text)
        bounding_boxes.append({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})


    # 4. Retourner la list
    return bounding_boxes

if __name__ == "__main__":
    # Remplace par un vrai chemin vers un de tes fichiers XML
    test_path = "data/raw/NEU-DET/train/annotations/crazing_28.xml"
    result = extract_defect_boxes(test_path)
    print(result)