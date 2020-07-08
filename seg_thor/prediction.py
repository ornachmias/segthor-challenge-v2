from collections import namedtuple

from skimage.measure import label
import SimpleITK as sitk
import pickle
import os
import numpy as np
import cv2

from importlib import import_module
import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from data_utils.torch_data import THOR_Data, get_cross_validation_paths, get_global_alpha
import data_utils.transforms as tr
from models.loss_funs import DependentLoss
import postprocessing


def get_metrics(segmentation, mask, n_class=4):
    results = np.zeros((n_class, 5))
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    for i in range(n_class):
        cur_mask = (mask == i + 1).astype(np.int16)
        cur_segmentation = (segmentation == i + 1).astype(np.int16)
        segmentation_itk = sitk.GetImageFromArray(cur_segmentation)
        mask_itk = sitk.GetImageFromArray(cur_mask)
        overlap_measures_filter.Execute(segmentation_itk, mask_itk)
        hausdorff_distance_filter.Execute(segmentation_itk, mask_itk)
        results[i, 0] = overlap_measures_filter.GetJaccardCoefficient()
        results[i, 1] = overlap_measures_filter.GetDiceCoefficient()
        results[i, 2] = hausdorff_distance_filter.GetHausdorffDistance()
        results[i, 3] = overlap_measures_filter.GetFalseNegativeError()
        results[i, 4] = overlap_measures_filter.GetFalsePositiveError()
    return np.mean(results, 0)


def multi_scale(images, if_zoom=True, if_flip=True):
    total_images = []
    total_flip_flag = []
    total_zoom_flag = []
    zoom_scale = [512, 544, 576, 608, 640]  # , 672, 704, 736,768]
    if not if_zoom:
        zoom_scale = [512]
    flip_scale = [[1, -1], [1, 1]]
    if not if_flip:
        flip_scale = [[1, 1]]
    for cur_zoom in zoom_scale:
        for cur_flip in flip_scale:
            cur_images = images.transpose(0, 2, 3, 1)
            new_images = []
            for cur_idx in range(cur_images.shape[0]):
                new_image = cur_images[cur_idx]
                cur_zoom0 = cur_zoom / float(new_image.shape[0])
                cur_zoom1 = cur_zoom / float(new_image.shape[1])
                new_image = cv2.resize(
                    new_image,
                    None,
                    fx=cur_zoom1,
                    fy=cur_zoom0,
                    interpolation=cv2.INTER_LINEAR)

                new_image = np.ascontiguousarray(
                    new_image[::cur_flip[0], ::cur_flip[1]])
                new_images.append(new_image)
            total_zoom_flag.append((1 / cur_zoom0, 1 / cur_zoom1))
            new_images = np.stack(new_images, 0).transpose(0, 3, 1, 2)
            total_flip_flag.append(cur_flip)
            total_images.append(new_images)
    return total_images, total_flip_flag, total_zoom_flag


def recover(images, total_flip_flag, total_zoom_flag):
    total_labels = []
    for cur_images, flip_flag, zoom_flag in zip(images, total_flip_flag,
                                                total_zoom_flag):
        new_labels = []
        for idx in range(cur_images.shape[0]):
            cur_img = cur_images[idx].transpose(1, 2, 0)
            cur_zoom0 = zoom_flag[0]
            cur_zoom1 = zoom_flag[1]
            new_label = cv2.resize(
                cur_img.astype(np.float32),
                None,
                fx=cur_zoom1,
                fy=cur_zoom0,
                interpolation=cv2.INTER_LINEAR)
            new_label = np.ascontiguousarray(
                new_label[::flip_flag[0], ::flip_flag[1]])
            new_label = new_label.transpose(2, 0, 1)
            new_labels.append(new_label)
        new_labels = np.stack(new_labels, 0)
        total_labels.append(new_labels)
    total_labels = np.stack(total_labels, 0)
    return np.mean(total_labels, 0)


def getLargestCC(segmentation, connectivity):
    labels = label(segmentation, connectivity=connectivity)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg = list(zip(unique, counts))[1:]
    max_unique = 0
    max_count = 0
    for _unique, _count in list_seg:
        if max_count < _count:
            max_count = _count
            max_unique = _unique
    labels[np.where(labels != max_unique)] = 0
    return labels / max_unique


def postprocess_targets(patients, slices, predictions, targets):
    patients = [patient.int() for patient in patients]
    slices = [slice.int() for slice in slices]
    predictions = np.concatenate(predictions, 0)
    targets = np.concatenate(targets, 0)
    indexes = list(range(len(patients)))
    indexes.sort(key=lambda idx: (patients[idx], slices[idx]))
    targets = np.array([targets[idx] for idx in indexes])
    predictions = np.array([predictions[idx] for idx in indexes])
    start_idx = 0
    new_predictions = []
    for current_idx, patient in enumerate(patients):
        if patient != patients[start_idx] or current_idx == len(patients)-1:
            print(current_idx)
            print(start_idx)
            print(predictions.shape)
            sub_predictions = predictions[start_idx:current_idx + 1]
            print(sub_predictions.shape)
            sub_predictions = postprocessing.axis_based_denoise_method(sub_predictions, [1, 2, 3, 4])
            new_predictions.append(sub_predictions)
            start_idx = current_idx + 1

    return np.concatenate(np.array(new_predictions), 0), targets

def print_metrics(metrics):
    print('the JaccardCoefficient is --> %4f' % metrics[0])
    print('the DiceCoefficient    is --> %4f' % metrics[1])
    print('the HausdorffDistance  is --> %4f' % metrics[2])
    print('the FalseNegativeError is --> %4f' % metrics[3])
    print('the FalsePositiveError is --> %4f' % metrics[4])

#################################################################
# ######################testing config###########################
gpus = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

DEVICE = torch.device("cuda" if True else "cpu")
data_path = "../data/data_npy/"
custom_output = './custom_output/'
os.makedirs(custom_output, exist_ok=True)

test_flag = 0
train_files, test_files = get_cross_validation_paths(test_flag)
model_flag = 'MTL_WMCE'
net_name = 'ResUNet101'
loss_name = 'CombinedLoss'
if_fpf = True
saved_checkpoint = '63.ckpt'
dynamic_threshold = [0.2398, 0.2151, 0.1941, 0.1636]  # None
precise_net_path = os.path.join("results", net_name, str(test_flag), saved_checkpoint)
if_dependent = 1
if if_dependent == 1:
    alpha = get_global_alpha(train_files, data_path)
    alpha = torch.from_numpy(alpha).float().to(DEVICE)
    alpha.requires_grad = False
else:
    alpha = None

################################################################

from torch.nn import DataParallel

model = import_module('models.model_loader')
precise_net, loss = model.get_full_model(
    net_name, loss_name, n_classes=5, alpha=alpha)
c_loss = DependentLoss(alpha)
checkpoint = torch.load(precise_net_path)
precise_net.load_state_dict(checkpoint['state_dict'])
precise_net = precise_net.to(DEVICE)
precise_net = DataParallel(precise_net)
precise_net.eval()

################# first get the threshold #####################
composed_transforms_tr = transforms.Compose([
    tr.Normalize(mean=(0.12, 0.12, 0.12), std=(0.018, 0.018, 0.018)),
    tr.ToTensor2(5)
])
eval_dataset = THOR_Data(
    transform=composed_transforms_tr,
    path=data_path,
    file_list=test_files)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4)

predict_c = []
label_c = []
predictions = []
targets = []
patients = []
slices = []
for i, sample in enumerate(eval_loader):
    if i % 50 == 0:
        print(i)
    data = sample['image']
    target_c = sample['label_c']
    target_s = sample['label_s']
    for patient in sample["patient"]:
        patients.append(patient)
    for slc in sample["slice"]:
        slices.append(slc)

    if net_name == "ResUNet101Index":
        batch_size = data.size(0)
        h = data.size(2)
        w = data.size(3)
        data = torch.cat((data, sample['index'].float().unsqueeze(0).reshape((batch_size, 1, h, w))), dim=1)

    custom_output_dir = os.path.join(custom_output, str(i))
    os.makedirs(custom_output_dir, exist_ok=True)
    img_file_path = os.path.join(custom_output_dir, 'img.pkl')
    pickle.dump(np.asarray(sample['image']), open(img_file_path, 'wb'))
    gt_file_path = os.path.join(custom_output_dir, 'gt.pkl')
    pickle.dump(np.asarray(sample['label_s']), open(gt_file_path, 'wb'))

    label_c.append(target_c.numpy())
    targets.append(torch.argmax(target_s, 1).numpy())
    data = data.to(DEVICE)
    target_c = target_c.to(DEVICE)
    target_s = target_s.to(DEVICE)

    cur_img2_scales, total_flip_flag, total_zoom_flag = multi_scale(
        data.cpu().numpy(), if_zoom=True, if_flip=False)
    predict_slabel_scales = []
    predict_clabel_scales = []
    for cur_img2 in cur_img2_scales:
        cur_img2 = torch.from_numpy(cur_img2.astype('float32'))
        with torch.no_grad():
            data = cur_img2.to(DEVICE)
            output_s, output_c = precise_net(data)
            if model_flag != 'MTL_WMCE':
                c_p = output_c
            else:
                _, c_p = c_loss(output_c, target_c)
        predict_clabel_scales.append(c_p.cpu().numpy())
        predicted_label_s = torch.softmax(output_s, 1)
        predicted_label_s = predicted_label_s.cpu().numpy()
        predict_slabel_scales.append(predicted_label_s)
    predict_clabel = np.mean(np.array(predict_clabel_scales), 0)

    if dynamic_threshold == None:
        predict_clabel = (predict_clabel > 0.5).astype('uint8')
    else:
        for i in range(predict_clabel.shape[1]):
            predict_clabel[:, i] = (predict_clabel[:, i] > dynamic_threshold[i]).astype('uint8')
    recover_label = recover(predict_slabel_scales, total_flip_flag,
                            total_zoom_flag)
    predict_slabel = np.argmax(recover_label, 1)
    if model_flag != 'SM' and if_fpf:
        for i in range(predict_clabel.shape[1]):
            for j in range(predict_clabel.shape[0]):
                if predict_clabel[j, i] == 0:
                    predict_slabel[j][np.where(predict_slabel[j] == i + 1)] = 0
    predictions.append(predict_slabel.astype('uint8'))

    pred_file_path = os.path.join(custom_output_dir, 'pred.pkl')
    pickle.dump(predict_slabel.astype('uint8'), open(pred_file_path, 'wb'))

metric5 = get_metrics(np.concatenate(predictions, 0), np.concatenate(targets, 0))
print('Results before postprocess:')
print_metrics(metric5)
processed_predictions, targets = postprocess_targets(patients, slices, predictions, targets)
new_metric5 = get_metrics(processed_predictions, targets)

print('Results After postprocess:')
print_metrics(new_metric5)
