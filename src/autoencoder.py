import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    """
    Module d'attention spatiale.
    Prend une feature map (C, H, W), produit une carte d'attention (1, H, W).
    """
    def __init__(self, kernel_size=7):
        
        super(SpatialAttention, self).__init__()
        # Convolution qui prend 2 canaux (Avg + Max) et sort 1 canal
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=kernel_size//2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x est de forme (Batch, 128, 7, 7)
        # 1. Compression sur l'axe des canaux
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (Batch, 1, 7, 7)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (Batch, 1, 7, 7)
        
        # 2. Concaténation
        combined = torch.cat([avg_out, max_out], dim=1)  # (Batch, 2, 7, 7)
        # 3. Calcul de la carte d'attention
        attention = self.layers(combined)  # (Batch, 1, 7, 7)
        
        # 4. Application
        return x * attention


class Autoencoder(nn.Module):
    def __init__(self, HasAttention=False):
        self.HasAttention = HasAttention
        super(Autoencoder, self).__init__()
        
        # Encodeur
        self.encoder = nn.Sequential(
            # Conv1 : (1, 50, 50) → (32, 25, 25)
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Conv2 : (32, 25, 25) → (64, 13, 13)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Conv3 : (64, 13, 13) → (128, 7, 7)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Module d'attention
        if(self.HasAttention):
            self.attention = SpatialAttention()
        
        # Décodeur
        self.decoder = nn.Sequential(
            # Deconv1 : (128, 7, 7) → (64, 13, 13)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            # Deconv2 : (64, 13, 13) → (32, 25, 25)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            # Deconv3 : (32, 25, 25) → (1, 50, 50)
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if(self.HasAttention):
            encoded = self.encoder(x)
            attended = self.attention(encoded)
            decoded = self.decoder(attended)
            return decoded
        else :
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

if __name__ == "__main__":
    # Petit test pour vérifier que le fichier est bien à jour
    model = Autoencoder()
    x = torch.randn(1, 1, 50, 50)
    out = model(x)
    print(f"Test succès ! Output : {out.shape}")