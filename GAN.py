import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
import glob
import os
import config as CFG
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder
import cv2

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 3 * 64 * 64)
        self.tanh = nn.Tanh()

    def forward(self, z):
        out = self.fc(z)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.tanh(out)
        out = out.view(-1, 3, 64, 64)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(256 * 8 * 8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)
        out = self.conv3(out)
        out = self.leaky_relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

class GANDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, '*.jpg'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

batch_size = 64
latent_dim = 100
num_epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = GANDataset('Datasets/Flicker-8k/Images/no_label', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(num_epochs):
    for i, image in enumerate(dataloader):
        real_images = image.to(device)
        batch_size = real_images.size(0)

        discriminator.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        real_outputs = discriminator(real_images)
        d_loss_real = criterion(real_outputs, real_labels)
        d_loss_real.backward()

        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        fake_outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)
        d_loss_fake.backward()

        discriminator_loss = d_loss_real + d_loss_fake
        discriminator_optimizer.step()

        generator.zero_grad()
        outputs = discriminator(fake_images)
        generator_loss = criterion(outputs, real_labels)
        generator_loss.backward()
        generator_optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], '
                  f'Discriminator Loss: {discriminator_loss.item():.4f}, '
                  f'Generator Loss: {generator_loss.item():.4f}')

torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')

if CFG.pretrained:
    generator.load_state_dict(torch.load('generator.pth'))
else:
    generator.apply(weights_init)

generator.eval()

data_transforms = transforms.Compose([
    transforms.Resize(CFG.size),
    transforms.CenterCrop(CFG.size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

original_dataset = ImageFolder(root='Datasets/Flicker-8k/Images', transform=data_transforms)
original_dataloader = DataLoader(original_dataset, batch_size=batch_size, shuffle=True, num_workers=CFG.num_workers)

augmented_images = []

with torch.no_grad():
    for batch_idx, (images, _) in enumerate(original_dataloader):
        images = images.to(device)
        generated_images = generator(images)
        augmented_images.append(generated_images.cpu())


augmented_images = torch.cat(augmented_images, dim=0)
vutils.save_image(augmented_images, 'augmented_images.png', normalize=True)

print("Augmented images saved successfully!")