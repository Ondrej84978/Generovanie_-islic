import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

# Hyperparametre
batch_size = 128
lr = 0.0002
latent_dim = 100      # dĺžka náhodného vektora z
num_epochs = 50       
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("Generovane_obrazky", exist_ok=True)

# Dátový loader (MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))   # [0,1] -> [-1,1]
])

dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Architektúra sietí
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 28 * 28),
            nn.Tanh()   # výstup v rozsahu [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(z.size(0), 1, 28, 28)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()   # pravdepodobnosť "real"
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Inicializácia modelov
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Na grafy si budeme ukladať priemerné straty za každú epochu
g_losses = []
d_losses = []

# Tréningová slučka
for epoch in range(num_epochs):
    epoch_g_loss = 0.0
    epoch_d_loss = 0.0

    for i, (imgs, _) in enumerate(dataloader):

        real_imgs = imgs.to(device)
        batch_size_curr = real_imgs.size(0)

        # Tréning Discriminátora
        discriminator.zero_grad()

        valid = torch.ones(batch_size_curr, 1, device=device)
        fake = torch.zeros(batch_size_curr, 1, device=device)

        # Loss na reálnych obrázkoch
        real_pred = discriminator(real_imgs)
        d_real_loss = criterion(real_pred, valid)

        # Loss na generovaných obrázkoch
        z = torch.randn(batch_size_curr, latent_dim, device=device)
        gen_imgs = generator(z)
        fake_pred = discriminator(gen_imgs.detach())
        d_fake_loss = criterion(fake_pred, fake)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Tréning Generatora
        generator.zero_grad()

        z = torch.randn(batch_size_curr, latent_dim, device=device)
        gen_imgs = generator(z)
        fake_pred = discriminator(gen_imgs)
        g_loss = criterion(fake_pred, valid)  # chceme, aby D veril, že sú reálne

        g_loss.backward()
        optimizer_G.step()

        # Ukladanie súčtov lossov pre priemerovanie
        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()

        if i % 200 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Batch [{i}/{len(dataloader)}] "
                f"D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}"
            )

    # priemer lossov za epochu
    epoch_g_loss /= len(dataloader)
    epoch_d_loss /= len(dataloader)
    g_losses.append(epoch_g_loss)
    d_losses.append(epoch_d_loss)

    print(f"===> Obrazok_cislic {epoch+1}/{num_epochs} | "
          f"avg D loss: {epoch_d_loss:.4f} | avg G loss: {epoch_g_loss:.4f}")

    # Uloženie generovaných obrázkov po každej epoche
    with torch.no_grad():
        z = torch.randn(64, latent_dim, device=device)
        gen_imgs = generator(z)
        gen_imgs = (gen_imgs + 1) / 2   # späť do [0,1]
        grid = make_grid(gen_imgs, nrow=8)
        save_image(grid, f"Generovane_obrazky/Obrazok_cislic_{epoch+1:03d}.png")

# Vykreslenie grafov lossov
epochs = range(1, num_epochs + 1)

# Graf Generatora lossu 
plt.figure(figsize=(8, 5))
plt.plot(epochs, g_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Generator Loss (G-loss)")
plt.grid(True)
plt.tight_layout()
plt.savefig("Graf_Generatora_Lossu.png")
plt.show()

# Graf Discriminatora lossu
plt.figure(figsize=(8, 5))
plt.plot(epochs, d_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Discriminator Loss (D-loss)")
plt.grid(True)
plt.tight_layout()
plt.savefig("Graf_Discriminatora_Lossu.png")
plt.show()

print("Grafy uložené: Graf_Generatora_Lossu.png a Graf_Discriminatora_Lossu.png")