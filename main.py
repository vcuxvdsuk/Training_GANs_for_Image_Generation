import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from generator import Generator  # Example: from a generator.py file
from discriminator import Discriminator

# Initialize W&B
wandb.init(project="gan-image-generation", name="gan-training-run")

# Config parameters
config = {
    "learning_rate": 0.0002,
    "batch_size": 64,
    "latent_dim": 100,
    "epochs": 100,
}
wandb.config.update(config)

# Instantiate models
latent_dim = config["latent_dim"]
generator = Generator(latent_dim)
discriminator = Discriminator()

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

# Define loss function
loss = nn.BCELoss()

# Define optimizers
optimizer_gen = optim.Adam(generator.parameters(), lr=config["learning_rate"], betas=(0.5, 0.999))
optimizer_disc = optim.Adam(discriminator.parameters(), lr=config["learning_rate"], betas=(0.5, 0.999))

# Define transformations
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# Load dataset
try:
    dataset = datasets.ImageFolder("path_to_dataset", transform=transform)
except FileNotFoundError:
    raise FileNotFoundError("Dataset not found. Check 'path_to_dataset'.")

dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# Training loop
for epoch in range(config["epochs"]):
    for real_images, _ in dataloader:
        real_images = real_images.to(device)

        # Train discriminator
        noise = torch.randn(real_images.size(0), latent_dim, 1, 1, device=device)  # Generate noise
        fake_images = generator(noise)

        # Real loss
        real_labels = torch.ones(real_images.size(0), 1, device=device)
        fake_labels = torch.zeros(fake_images.size(0), 1, device=device)
        real_loss = loss(discriminator(real_images), real_labels)
        
        # Fake loss
        fake_loss = loss(discriminator(fake_images.detach()), fake_labels)

        # Total discriminator loss
        d_loss = real_loss + fake_loss
        discriminator.zero_grad()
        d_loss.backward()
        optimizer_disc.step()

        # Train generator
        fake_labels_gen = torch.ones(fake_images.size(0), 1, device=device)
        g_loss = loss(discriminator(fake_images), fake_labels_gen)

        generator.zero_grad()
        g_loss.backward()
        optimizer_gen.step()

        # Log metrics to W&B
        wandb.log({
            "epoch": epoch,
            "discriminator_loss": d_loss.item(),
            "generator_loss": g_loss.item(),
        })

    # Generate and log images every 10 epochs
    if epoch % 10 == 0:
        sample_noise = torch.randn(16, latent_dim, 1, 1, device=device)
        with torch.no_grad():
            fake_samples = generator(sample_noise).cpu()
        wandb.log({"Generated Images": [wandb.Image(img) for img in fake_samples]})

# W&B sweep configuration
sweep_config = {
    "method": "grid",
    "parameters": {
        "learning_rate": {"values": [0.0001, 0.0002, 0.0005]},
        "batch_size": {"values": [32, 64, 128]},
        "latent_dim": {"values": [64, 100, 128]},
    }
}
sweep_id = wandb.sweep(sweep_config, project="gan-image-generation")

# Sweep training function
def train_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # Reinitialize models and optimizers with new hyperparameters
        generator = Generator(config.latent_dim).to(device)
        discriminator = Discriminator().to(device)
        optimizer_gen = optim.Adam(generator.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
        optimizer_disc = optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
        
        # Training loop (reuse the loop above)
        for epoch in range(config.epochs):
            # Train with the new config parameters
            pass  # Add training logic here

# Start the sweep agent
wandb.agent(sweep_id, train_sweep)

# Watch the models
wandb.watch(generator, log="all")
wandb.watch(discriminator, log="all")

# Save generator model
torch.save(generator.state_dict(), "generator.pth")
wandb.save("generator.pth")
