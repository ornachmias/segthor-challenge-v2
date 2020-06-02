import numpy as np
import torch
import os
import time
import cv2
import nibabel as nib
import pdb


def truncated_range(img):
    max_hu = 384
    min_hu = -384
    img[np.where(img > max_hu)] = max_hu
    img[np.where(img < min_hu)] = min_hu
    return (img - min_hu) / (max_hu - min_hu) * 255.


path = '../data/data_source/'
save_path = '../data/data_npy/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

files = os.listdir(path)
count = 0
print('begin processing data')

means = []
stds = []
total_imgs = []

for i, volume in enumerate(files):
    cur_file = os.path.join(path, volume)
    print(i, cur_file)
    cur_save_path = os.path.join(save_path, volume)
    if not os.path.exists(cur_save_path):
        os.makedirs(cur_save_path)
    img = nib.load(os.path.join(cur_file, volume + '.nii'))
    img = np.array(img.get_data())
    label = nib.load(os.path.join(cur_file, 'GT.nii'))
    label = np.array(label.get_data())
    img = truncated_range(img)

    for idx in range(img.shape[2]):
        if idx == 0 or idx == img.shape[2] - 1:
            continue
        # 2.5D data, using adjacent 3 images
        cur_img = img[:, :, idx - 1:idx + 2].astype('uint8')
        total_imgs.append(cur_img)
        cur_label = label[:, :, idx].astype('uint8')
        count += 1
        np.save(
            os.path.join(cur_save_path,
                         volume + '_' + str(idx) + '_image.npy'), cur_img)
        np.save(
            os.path.join(cur_save_path,
                         volume + '_' + str(idx) + '_label.npy'), cur_label)

    total_imgs = np.stack(total_imgs, 3) / 255.
    means.append(np.mean(total_imgs))
    stds.append(np.std(total_imgs))
    total_imgs = []

print('data mean is %f' % np.mean(means))
print('data std is %f' % np.std(stds))
print('total data size is %f' % count)
print('processing data end !')