import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
# Autres imports nécessaires...

class CleanPatchesDataset(Dataset):
    """
    Dataset des patches sains pour entraîner l'autoencodeur.
    """
    
    def __init__(self, patches_folder, transform=None):
        """
        Args:
            patches_folder: chemin vers data/processed/clean_patches
            transform: transformations à appliquer (optionnel)
        """
        # 1. Lister tous les fichiers .jpg dans le dossier
        patches = []
        for file in os.listdir(patches_folder):
            if file.endswith(".jpg"):
                patches.append(os.path.join(patches_folder, file))
        # 2. Stocker la liste et les paramètres
        self.patches = patches
        self.transform = transform
        self.patches_folder = patches_folder
    
    def __len__(self):
        # Retourner le nombre de patches
        return len(self.patches)
    
    def __getitem__(self, idx):
        # 1. Charger l'image à l'index idx
        img = Image.open(self.patches[idx]).convert("L")  # "L" = Luminance (grayscale)
        # 2. Appliquer les transformations si définies
        if self.transform:
            img = self.transform(img)
        # 3. Retourner l'image (en tensor)
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array)
        img_tensor = img_tensor / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor




if __name__ == "__main__":
    dataset = CleanPatchesDataset("data/processed/clean_patches")
    print(f"Nombre de patches : {len(dataset)}")
    
    # Charger un patch
    sample = dataset[0]
    print(f"Type : {type(sample)}")
    print(f"Shape : {sample.shape}")
    print(f"Min/Max valeurs : {sample.min():.3f} / {sample.max():.3f}")