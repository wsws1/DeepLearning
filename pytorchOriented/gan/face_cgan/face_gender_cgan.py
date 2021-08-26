from torch.utils.data import Dataset, DataLoader
import cv2
import os
import random
import pylab
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torchvision
import glob
import torchvision.transforms as transforms

workspace_dir = '/mnt/03a7b3b6-14f4-4a74-958b-71dd6cdd09bc/dataset/lfwcrop_color/faces'
# print(glob.glob(workspace_dir + '/*.ppm'))
model_save_path = './models'
# build gender dict
gender_dict = {}

def get_gender_gict(label_path):
    gender_dict = {}
    with open(label_path + "male_names.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            male_img_name = line.split('.')[0]
            gender_dict[male_img_name] = 1
            # print(male_img_name)
    with open(label_path + "female_names.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            female_img_name = line.split('.')[0]
            gender_dict[female_img_name] = 0
            # print(male_img_name)

    return gender_dict
gender_dict = get_gender_gict('/mnt/03a7b3b6-14f4-4a74-958b-71dd6cdd09bc/dataset/lfwcrop_color/')

# print(gender_dict)
class FaceDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)
    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img_name = fname.split('/')[-1].split('.')[0]

        img_label = 1
        try:
            img_label = gender_dict[img_name]
        except Exception as e:
            pass
        # print(img_name, img_label)
        img = cv2.imread(fname)
        img = self.BGR2RGB(img) #because "torchvision.utils.save_image" use RGB
        img = self.transform(img)
        return img, img_label

    def __len__(self):
        return self.num_samples

    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)



def get_dataset(root):
    fnames = glob.glob(root + '/*.ppm')
    # resize the image to (64, 64)
    # linearly map [0, 1] to [-1, 1]
    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((64, 64)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3) ] )
    dataset = FaceDataset(fnames, transform)
    return dataset



def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    """
    input (N, in_dim)
    output (N, 3, 64, 64)
    """
    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(2, 2)

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())
        self.l1 = nn.Sequential(
            nn.Linear(in_dim , dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh())
        self.apply(weights_init)
    def forward(self, x, gender_labels):
        y = torch.cat((self.label_emb(gender_labels), x), -1)
        y = self.l1(y)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y

class Discriminator(nn.Module):
    """
    input (N, 3, 64, 64)
    output (N, )
    """
    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(2, 2)
        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2))
        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),
            conv_bn_lrelu(dim * 2, dim * 4),
            conv_bn_lrelu(dim * 4, dim * 8),
            # nn.Conv2d(dim * 8, 1, 4),
            # nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.Linear(dim * 8 * 16 + 2, 1),
            nn.Sigmoid()
        )
        self.apply(weights_init)
    def forward(self, x, gender_labels):

        y = self.ls(x)
        # print("y ", y.shape, " labels ", self.label_emb(gender_labels).shape)
        y = y.view(y.size(0), -1)
        # print("y ", y.shape, " labels ", self.label_emb(gender_labels).shape)
        y = torch.cat((y, self.label_emb(gender_labels)), -1)
        y = self.fc(y)

        # y = self.ls(x)
        return y


# hyperparameters
batch_size = 64
z_dim = 100
lr = 1e-4
n_epoch = 10
save_dir = os.path.join(workspace_dir+'/face_gender_cgan', 'logs')
os.makedirs(save_dir, exist_ok=True)

# model
G = Generator(in_dim=z_dim + 2).cuda()
D = Discriminator(in_dim=3).cuda()
G.train()
D.train()

# loss criterion
criterion = nn.BCELoss()

# optimizer
opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))


same_seeds(0)
# dataloader (You might need to edit the dataset path if you use extra dataset.)
dataset = get_dataset(workspace_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


# plt.imshow(dataset[10][0].numpy().transpose(1,2,0))
# for i in range(20):
#     plt.imshow(dataset[i][0].numpy().transpose(1, 2, 0))
#     print("gender: ",dataset[i][1])
#     pylab.show()

# for logging
z_sample = Variable(torch.randn(100, z_dim)).cuda()
print(len(dataloader))
print(len(glob.glob(workspace_dir + '/*')))

# Loss functions
adversarial_loss = torch.nn.MSELoss()
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

if __name__ == '__main__':
    step_n = 0
    for e, epoch in enumerate(range(n_epoch)):
        for i, data in enumerate(dataloader):
            imgs, r_label = data
            imgs = imgs.cuda()
            r_label = r_label.cuda()
            # print(r_label)

            bs = imgs.size(0)

            """ Train D """
            # Adversarial ground truths
            valid = Variable(FloatTensor(bs, 1).fill_(1.0), requires_grad=False).cuda()
            fake = Variable(FloatTensor(bs, 1).fill_(0.0), requires_grad=False).cuda()
            # label
            f_label = Variable(LongTensor(np.random.randint(0, 2, bs))).cuda()

            z = Variable(torch.randn(bs, z_dim)).cuda()
            r_imgs = Variable(imgs).cuda()
            f_imgs = G(z, f_label)



            # dis
            r_logit = D(r_imgs.detach(), r_label)
            f_logit = D(f_imgs.detach(), f_label)

            # compute loss
            # print(r_logit[:10])
            # print(r_logit.shape, valid.shape)
            r_loss = adversarial_loss(r_logit, valid)
            f_loss = adversarial_loss(f_logit, fake)
            loss_D = (r_loss + f_loss) / 2

            # update model
            D.zero_grad()
            loss_D.backward()
            opt_D.step()

            """ train G """


            # leaf
            z = Variable(torch.randn(bs, z_dim)).cuda()
            # Generate gender lables
            gen_labels = Variable(LongTensor(np.random.randint(0, 2, bs))).cuda()
            f_imgs = G(z, gen_labels)

            # dis
            f_logit = D(f_imgs, gen_labels)

            # compute loss
            loss_G = adversarial_loss(f_logit, valid)

            # update model
            G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # log
            if step_n % 200 == 0:
                print(
                    f'\rEpoch [{epoch + 1}/{n_epoch}] {i + 1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}',
                    end='')
                # f_imgs_sample = (G(z, gen_labels).data + 1) / 2.0
                # filename = os.path.join(save_dir, f'train Step_{step_n + 1}.jpg')

                # torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
                # print(f' | Save some samples to {filename}.')
            step_n = step_n + 1
            # # show generated image
            # grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
            # plt.imshow(grid_img)
            # plt.figure(figsize=(10, 10))
            # plt.imshow(grid_img.permute(1, 2, 0))
            # pylab.show()
        G.eval()
        f_imgs_sample = (G(z,gen_labels).data + 1) / 2.0
        filename = os.path.join(save_dir, f'Epoch_{epoch + 1:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
        print(f' | Save some samples to {filename}.')
        # show generated image
        # grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
        # cv2.imshow(grid_img)
        # plt.figure(figsize=(10, 10))
        # plt.imshow(grid_img.permute(1, 2, 0))
        # pylab.show()
        G.train()
        if (e + 1) % 5 == 0:
            torch.save(G.state_dict(), os.path.join(model_save_path, f'dcgan_g.pth'))
            torch.save(D.state_dict(), os.path.join(model_save_path, f'dcgan_d.pth'))

