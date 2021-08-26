import torch
import glob
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
from resnet import resnet
from inceptionMouble import InceptionNetSmall
from mobilenet_v1 import mobilenet_v1_small

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = mobilenet_v1_small()

net.load_state_dict(torch.load("/home/vipsl/ws-work/projects/DeepLearning/pytorchOriented/chapter6/models/mobilenet_v1/1.pth"))
im_list = glob.glob("../data/CIFAR10/TEST/*/*")

np.random.shuffle(im_list)

net.to(device)

label_name = ["airplane", "automobile", "bird",
              "cat", "deer", "dog",
              "frog", "horse", "ship", "truck"]

test_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

for im_path in im_list:
    net.eval()
    im_data = Image.open(im_path)

    inputs = test_transform(im_data)
    inputs = torch.unsqueeze(inputs, dim=0)

    inputs = inputs.to(device)
    outputs = net.forward(inputs)

    _, pred = torch.max(outputs.data, dim=1)

    print("real label:",im_path.split('/')[-2]," pred label:",label_name[pred.cpu().numpy()[0]])

    img = np.asarray(im_data)
    img = img[:, :, [1, 2, 0]]

    img = cv2.resize(img, (300, 300))
    cv2.imshow("im", img)
    cv2.waitKey()



