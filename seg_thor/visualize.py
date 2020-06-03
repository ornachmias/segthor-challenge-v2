import os
import time
from importlib import import_module

import cv2
import data_utils.transforms as tr
import numpy as np
import torch
from data_utils.torch_data import THOR_Data, get_cross_validation_paths, get_global_alpha
from models.loss_funs import DependentLoss
from skimage.measure import label
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision import transforms
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def draw_single_image(x, y, parent_axis):
    parent_axis.axis('off')
    rescaled = (255.0 / x.max() * (x - x.min())).astype(np.uint8)

    parent_axis.imshow(rescaled, cmap='gray')
    parent_axis.matshow(y, cmap='tab10', alpha=0.8)


def draw_multiple_images(gt_x, gt_y, pred_x, pred_y):
    number_of_plots = len(gt_x)
    fig = plt.figure(figsize=(3, number_of_plots))

    gs = GridSpec(number_of_plots, 2, width_ratios=[1, 1], wspace=0.0, hspace=0.0)
    title_ax1 = fig.add_subplot(gs[:, 0])
    title_ax1.set_title('ground_truth')

    title_ax2 = fig.add_subplot(gs[:, 1])
    title_ax2.set_title('prediction')

    for i in range(number_of_plots):
        ax1 = fig.add_subplot(gs[i, 0])
        draw_single_image(gt_x[i], gt_y[i], ax1)

        ax2 = fig.add_subplot(gs[i, 1])
        draw_single_image(pred_x[i], pred_y[i], ax2)

    for ax in fig.get_axes():
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax.axis('off')

    return fig


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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-nd', '--numpy_data', default='../data/data_npy')
    parser.add_argument('-o', '--output', default='../custom_output')
    parser.add_argument('-c', '--checkpoint', default='../data/checkpoints/14.ckpt')
    parser.add_argument('-m', '--model_name', default='ResUNet101')
    parser.add_argument('-n', '--samples_num', default=3)
    args = parser.parse_args()

    input = []
    gt = []
    pred = []

    gpus = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = args.numpy_data
    custom_output = args.output
    os.makedirs(custom_output, exist_ok=True)

    precise_net_path = args.checkpoint
    net_name = args.model_name
    samples_num = int(args.samples_num)

    test_flag = 0
    train_files, test_files = get_cross_validation_paths(test_flag)
    model_flag = 'MTL_WMCE'
    loss_name = 'CombinedLoss'
    if_fpf = True
    dynamic_threshold = [0.2398, 0.2151, 0.1941, 0.1636]  # None

    if_dependent = 1
    if if_dependent == 1:
        alpha = get_global_alpha(train_files, data_path)
        alpha = torch.from_numpy(alpha).float().to(DEVICE)
        alpha.requires_grad = False
    else:
        alpha = None


    model = import_module('models.model_loader')
    precise_net, loss = model.get_full_model(
        net_name, loss_name, n_classes=5, alpha=alpha)
    c_loss = DependentLoss(alpha)
    checkpoint = torch.load(precise_net_path, map_location=DEVICE)
    precise_net.load_state_dict(checkpoint['state_dict'])
    precise_net = precise_net.to(DEVICE)
    precise_net = DataParallel(precise_net)
    precise_net.eval()

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
        batch_size=1,
        shuffle=False,
        num_workers=4)

    predict_c = []
    label_c = []
    predictions = []
    targets = []

    for i, sample in enumerate(eval_loader):
        if i >= samples_num:
            break

        print('Sample {} in progress...'.format(i))
        data = sample['image']
        target_c = sample['label_c']
        target_s = sample['label_s']
        os.makedirs(custom_output, exist_ok=True)
        for x in data:
            input.append(np.asarray(x[1]).reshape((512, 512)))

        for x in target_s:
            gt.append(np.argmax(np.asarray(x), axis=0))

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

        for x in predict_slabel:
            pred.append(x.astype('uint8'))

    visualization_output = os.path.join(custom_output, str(int(time.time())))
    fig = draw_multiple_images(input, gt, input, pred)
    fig.savefig(visualization_output)

