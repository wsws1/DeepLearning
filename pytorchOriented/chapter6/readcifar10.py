import pickle
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
label_name = ["airplane",
              "automobile",
              "bird",
              "cat",
              "deer",
              "dog",
              "frog",
              "horse",
              "ship",
              "truck"]

import glob
import numpy as np
import cv2
import os

train_list = glob.glob("../data/CIFAR10/test_batch*")
print(train_list)
save_path = "../data/CIFAR10/TEST"

for l in train_list:
    print(l)
    l_dict = unpickle(l)
    # print(l_dict.keys())
    for im_idx, im_data in enumerate(l_dict[b'data']):
        # print(im_idx)
        # print(im_data)
        im_label = l_dict[b'labels'][im_idx]
        im_name = l_dict[b'filenames'][im_idx]
        # print(im_label,im_data,im_name)
        im_label_name = label_name[im_label]
        im_data = im_data.reshape([3, 32, 32])
        im_data = np.transpose(im_data, (1, 2, 0))
        # cv2.imshow("im_data", cv2.resize(im_data, (200, 200)))
        # cv2.waitKey(0)

        img_label_path = "{}/{}".format(save_path, im_label_name)
        if not os.path.exists(img_label_path):
            os.mkdir(img_label_path)
        img_path = "{}/{}/{}".format(save_path, im_label_name,
                                     im_name.decode('UTF-8'))
        cv2.imwrite(img_path, im_data)



























