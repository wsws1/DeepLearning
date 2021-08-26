import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import os

if not os.path.exists('./img'):
    os.mkdir('./img')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def to_img(x):
    # every data point from (-1,1)  to (0,1)
    out = 0.5 * (x + 1)
    # (0, 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out


batch_size = 128
num_epoch = 100
z_dimension = 100

# Image processing
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])
# MNIST dataset
mnist = datasets.MNIST(
    root='../data/', train=True, transform=img_transform,download=True)


# Data loader
dataloader = torch.utils.data.DataLoader(
    dataset=mnist, batch_size=batch_size, shuffle=True)

print(dataloader)
# Discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2), nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.dis(x)
        return x


# Generator
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 256), nn.ReLU(True), nn.Linear(256, 784), nn.Tanh())

    def forward(self, x):
        x = self.gen(x)
        return x


D = discriminator().to(device)
G = generator().to(device)

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

# Start training
for epoch in range(num_epoch):
    for i, (img, _) in enumerate(dataloader):
        num_img = img.size(0)
        # print(img.size())
        # =================train discriminator
        img = img.view(num_img, -1)
        real_img = img.to(device)
        real_label = torch.ones(num_img).to(device)
        fake_label = torch.zeros(num_img).to(device)

        # compute loss of real_img
        real_out = D(real_img).squeeze(1)
        # print(real_out.size(), real_label.size())
        d_loss_real = criterion(real_out, real_label)
        real_scores = real_out  # closer to 1 means better

        # compute loss of fake_img
        z = torch.randn(num_img, z_dimension).to(device)
        fake_img = G(z)
        fake_out = D(fake_img).squeeze(1)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out  # closer to 0 means better

        # bp and optimize
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ===============train generator
        # compute loss of fake_img
        z = torch.randn(num_img, z_dimension).to(device)
        fake_img = G(z)
        output = D(fake_img).squeeze(1)
        g_loss = criterion(output, real_label)

        # bp and optimize
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                  'D real: {:.6f}, D fake: {:.6f}'.format(
                      epoch, num_epoch, d_loss.item(), g_loss.item(),
                      real_scores.data.mean(), fake_scores.data.mean()))
    if epoch == 0:
        real_images = to_img(real_img.cpu().data)
        save_image(real_images, './img/real_images.png')

    fake_images = to_img(fake_img.cpu().data)
    save_image(fake_images, './img/fake_images-{}.png'.format(epoch + 1))

torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')