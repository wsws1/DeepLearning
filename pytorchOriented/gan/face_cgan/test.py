import cv2
import torch
from face_gender_cgan import  Generator
import os
from torch.autograd import Variable
import torchvision
from matplotlib import pyplot as plt
import numpy as np

workspace_dir = './'
z_dim = 100
# load pretrained model
G = Generator(z_dim + 2)
G.load_state_dict(torch.load(os.path.join(workspace_dir+'models', 'dcgan_g.pth')))
G.eval()
G.cuda()
def draw_result(data, label,save_path):
    fig = plt.figure(figsize=(20, 20))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    data_size = len(data)

    for i in range(data_size):
        ax = plt.subplot(10, 10, i + 1)
        ax.axis("off")
        gender = 'men' if label[i].item() == 1 else 'women'
        ax.set_title(gender)
        plt.imshow(np.transpose(data[i].numpy(), (1, 2, 0)))

    plt.savefig(save_path)
    plt.show()

# generate images and save the result
n_output = 100
cuda = True if torch.cuda.is_available() else False
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
gen_labels = Variable(LongTensor(np.random.randint(0, 2, n_output))).cuda()
z_sample = Variable(torch.randn(n_output, z_dim)).cuda()
imgs_sample = (G(z_sample, gen_labels).data + 1) / 2.0
print(gen_labels)
save_dir = os.path.join(workspace_dir, 'logs')
filename = os.path.join(save_dir, f'result.jpg')
# torchvision.utils.save_image(imgs_sample, filename, nrow=10)
# show image
draw_result(imgs_sample.cpu(), gen_labels.cpu(), filename)
# grid_img = torchvision.utils.make_grid(imgs_sample.cpu(), nrow=10)
# plt.figure(figsize=(10,10))
# plt.suptitle(gen_labels)
# plt.imshow(grid_img.permute(1, 2, 0))
# plt.show()