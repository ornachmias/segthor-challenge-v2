from torchvision import transforms
from data_utils.torch_data import THOR_Data
import data_utils.transforms as tr
import numpy as np
from tqdm import tqdm
from fractions import Fraction
import matplotlib.pyplot as plt

def get_files_paths():
    paths = []
    for i in range(1, 41):
        if i < 10:
            i = '0' + str(i)
        else:
            i = str(i)

        paths.append("Patient_" + i)
    return paths


composed_transforms_tr = transforms.Compose([
        tr.Normalize(mean=(0.12, 0.12, 0.12), std=(0.018, 0.018, 0.018)),
        tr.ToTensor2(5)])
dataset = THOR_Data(
    transform=composed_transforms_tr,
    path='../data/data_npy/',
    file_list=get_files_paths(),
    otsu=0
)

# min_value = None
# max_value = None
#
# for i in tqdm(dataset):
#     curr_min = np.min(i['image'][1].numpy())
#     curr_max = np.max(i['image'][1].numpy())
#
#     if min_value is None or curr_min < min_value:
#         min_value = curr_min
#
#     if max_value is None or curr_max > max_value:
#         max_value = curr_max
# print('Min Value: {}, Max Value: {}'.format(min_value, max_value))
# Min Value: -6.666666507720947, Max Value: 48.88888931274414

hist = {
    0: {},
    1: {},
    2: {},
    3: {},
    4: {}
}

max_index = 0
curr_index = 0

for i in tqdm(dataset):
    if not curr_index < max_index and max_index != 0:
        break

    curr_image = i['image'][1]
    curr_mask = i['label_s']

    for j in range(5):
        curr_values = curr_image[curr_mask[j] == 1]
        if curr_values is not None and curr_values.size(0) != 0:
            curr_values, counts = np.unique(curr_values.reshape(-1), return_counts=True, axis=0)
            for v, c in zip(curr_values, counts):
                key = np.multiply(v, 10)
                key = round(key) / 10
                if key not in hist[j]:
                    hist[j][key] = 0

                hist[j][key] = hist[j][key] + c

    curr_index += 1



#
# dic_values = {
#     0: {'min': 0, 'max': 0},
#     1: {'min': 0, 'max': 0},
#     2: {'min': 0, 'max': 0},
#     3: {'min': 0, 'max': 0},
#     4: {'min': 0, 'max': 0}
# }
#
# max_index = 0
# curr_index = 0
#
# for i in tqdm(dataset):
#     if not curr_index < max_index and max_index != 0:
#         break
#
#     curr_image = i['image'][1]
#     curr_mask = i['label_s']
#
#     for j in range(5):
#         curr_values = curr_image[curr_mask[j] == 1]
#         if curr_values is not None and curr_values.size(0) != 0:
#             dic_values[j]['min'] = np.min(curr_values.numpy())
#             dic_values[j]['max'] = np.max(curr_values.numpy())
#
#     curr_index += 1
#
# for k in dic_values:
#     print('Class {}: Min Value: {}, Max Value:{}'.format(k, dic_values[k]['min'], dic_values[k]['max']))


# Class 0: Min Value: -6.666666507720947, Max Value:48.88888931274414
# Class 1: Min Value: 17.952070236206055, Max Value:28.409587860107422
# Class 2: Min Value: 9.019608497619629, Max Value:27.538127899169922
# Class 3: Min Value: -6.666666507720947, Max Value:9.455338478088379
# Class 4: Min Value: 15.11982536315918, Max Value:23.39869499206543