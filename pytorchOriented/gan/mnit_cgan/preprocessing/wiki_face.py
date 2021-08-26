import glob
import os
import numpy as np
import scipy.io as scio
import shutil

data_path = '/mnt/03a7b3b6-14f4-4a74-958b-71dd6cdd09bc/dataset/wiki/'
if not os.path.exists(data_path+'men'):
    os.mkdir(data_path+'men')
if not os.path.exists(data_path+'women'):
    os.mkdir(data_path+'women')
matFile = scio.loadmat(data_path + 'wiki.mat')
# print(type(matFile))

wiki_mat = matFile['wiki'][0][0]
print(type(wiki_mat))
# print(wiki_mat.shape)
# for i in range(10):
#     print(wiki_mat[i])
wiki_ids = wiki_mat[0][0]
print(wiki_ids)
wiki_gender = wiki_mat[3][0]
print(wiki_gender.size)
id_gender = dict()
print(wiki_ids.size)
for i in range(wiki_ids.size):
    # print(wiki_ids[i],wiki_gender[i])
    # print(type(wiki_ids[i]))
    id_gender[wiki_ids[i]] = wiki_gender[i]
imgs = wiki_mat[2][0]
print(len(imgs))
print(type(imgs[0][0]))
for i in range(imgs.size):
    full_img_path = data_path+imgs[i][0]
    gender = wiki_gender[i]
    img_name = full_img_path.split('/')[-1]
    if gender == 1.0:
        shutil.copy(full_img_path, data_path + 'men/')
    else:
        shutil.copy(full_img_path, data_path + 'women/')
