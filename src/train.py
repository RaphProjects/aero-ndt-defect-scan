import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from autoencoder import Autoencoder
from dataset import CleanPatchesDataset

# Initialisation de l'encodeur
def train():
    # 1. Hyperparametres
    batch_size = 64
    learning_rate = 0.0005
    epochs = 30

    # 2. Charger les données
    dataset = CleanPatchesDataset("data/processed/clean_patches")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Créer le modèle, la loss, l'optimiseur
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Entraîner l'encodeur
    for epoch in range(epochs):
        for batch,i in zip(dataloader, range(len(dataloader))):
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            if i % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    # 5. Sauvegarder le modèle
    torch.save(model.state_dict(), "models/autoencoder_attention.pt")

if __name__ == "__main__":
    train()