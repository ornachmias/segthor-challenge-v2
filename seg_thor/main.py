import argparse
import csv
import logging
import os
import time
import uuid
from importlib import import_module

import data_utils.transforms as tr
import numpy as np
import torch
from data_utils.torch_data import THOR_Data, get_cross_validation_paths, get_global_alpha
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import setgpu, get_threshold, metric, segmentation_metrics
from stats import train_stats_path, eval_stats_path, stats_fields

###########################################################################

"""
                The main function of SegTHOR
                      Python 3
                    pytorch 1.1.0
                   author: Tao He
              Institution: Sichuan University
               email: taohe@stu.scu.edu.cn
"""
###########################################################################

parser = argparse.ArgumentParser(description='PyTorch SegTHOR Segmentation')
parser.add_argument(
    '--model_name',
    '-m_name',
    metavar='MODEL',
    default='DenseUNet161',
    help='model_name')
parser.add_argument(
    '--epochs',
    default=120,
    type=int,
    metavar='N',
    help='number of max epochs to run')
parser.add_argument(
    '--start-epoch',
    default=1,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=16,
    type=int,
    metavar='N',
    help='mini-batch size (default: 16)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.01,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum',
    default=0.9,
    type=float,
    metavar='M',
    help='momentum')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=0.00001,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-5)')
parser.add_argument(
    '--save_dir',
    default='SavePath/test/',
    type=str,
    metavar='SAVE',
    help='directory to save checkpoint (default: none)')
parser.add_argument(
    '--gpu',
    default='all',
    type=str,
    metavar='N',
    help='use gpu')
parser.add_argument(
    '--patient',
    default=10,
    type=int,
    metavar='N',
    help='the flat to stop training')
parser.add_argument(
    '--untest_epoch',
    default=10,
    type=int,
    metavar='N',
    help='number of untest_epoch, do not test for n epoch. just for saving time')
parser.add_argument(
    '--loss_name',
    default='CombinedLoss',
    type=str,
    metavar='N',
    help='the name of loss function')
parser.add_argument(
    '--data_path',
    default='../data/data_npy/',
    type=str,
    metavar='N',
    help='data path')
parser.add_argument(
    '--test_flag',
    default=0,
    type=int,
    metavar='0, 1, 2, 3',
    help='the test flag range in 0..9, 10..19, 20..29, 30..39 !')
parser.add_argument(
    '--n_class',
    default=5,
    type=int,
    metavar='n_class',
    help='number of classes')
parser.add_argument(
    '--if_dependent',
    default=1,
    type=int,
    metavar='1(True) or 0(False)',
    help='the flag to use WMCE')
parser.add_argument(
    '--if_closs',
    default=1,
    type=int,
    metavar='1(True) or 0(False)',
    help='if using multi-task learning')

DEVICE = torch.device("cuda" if True else "cpu")


def main(args):
    max_precision = 0.
    torch.manual_seed(123)
    cudnn.benchmark = True
    setgpu(args.gpu)
    data_path = args.data_path
    train_files, test_files = get_cross_validation_paths(args.test_flag)
    if args.if_dependent == 1:
        alpha = get_global_alpha(train_files, data_path)
        alpha = torch.from_numpy(alpha).float().to(DEVICE)
        alpha.requires_grad = False
    else:
        alpha = None
    model = import_module('models.model_loader')
    net, loss = model.get_full_model(
        args.model_name,
        args.loss_name,
        n_classes=args.n_class,
        alpha=alpha,
        if_closs=args.if_closs,
        class_weights=torch.FloatTensor([1.0, 5.0, 2.0, 3.0, 4.0]).to(DEVICE))
    start_epoch = args.start_epoch
    save_dir = args.save_dir
    logging.info(args)
    if args.resume:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        net.load_state_dict(checkpoint['state_dict'])

    net = net.to(DEVICE)
    loss = loss.to(DEVICE)
    if len(args.gpu.split(',')) > 1 or args.gpu == 'all':
        net = DataParallel(net)

    optimizer = torch.optim.SGD(
        net.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    init_lr = np.copy(args.lr)

    def get_lr(epoch):
        if args.lr < 0.0001:
            return args.lr
        if epoch > 0:
            args.lr = args.lr * 0.95
            logging.info('current learning rate is %f' % args.lr)
        return args.lr

    composed_transforms_tr = transforms.Compose([
        tr.RandomZoom((512, 512)),
        tr.RandomHorizontalFlip(),
        tr.Normalize(mean=(0.12, 0.12, 0.12), std=(0.018, 0.018, 0.018)),
        tr.ToTensor2(args.n_class)])
    train_dataset = THOR_Data(
        transform=composed_transforms_tr, path=data_path, file_list=train_files)
    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4)
    break_flag = 0.
    high_dice = 0.
    selected_thresholds = np.zeros((args.n_class - 1,))
    run_id = str(uuid.uuid4())
    cur_train_stats_path = train_stats_path.format(run_id)
    cur_eval_stats_path = eval_stats_path.format(run_id)
    with open(cur_train_stats_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(stats_fields)

    with open(cur_eval_stats_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(stats_fields)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, adaptive_thresholds = train(trainloader, net, loss, epoch,
                                                optimizer, get_lr,
                                                save_dir, cur_train_stats_path)
        if epoch < args.untest_epoch:
            continue
        break_flag += 1
        eval_dice, eval_precision = evaluation(args, net, loss, epoch, save_dir, test_files, selected_thresholds,
                                               cur_eval_stats_path)
        if max_precision <= eval_precision:
            selected_thresholds = adaptive_thresholds
            max_precision = eval_precision
            logging.info(
                '************************ dynamic threshold saved successful ************************** !'
            )
        if eval_dice >= high_dice:
            high_dice = eval_dice
            break_flag = 0
            if len(args.gpu.split(',')) > 1 or args.gpu == 'all':
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()
            torch.save({
                'epoch': epoch,
                'save_dir': save_dir,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
                'args': args
            }, os.path.join(save_dir, '%d.ckpt' % epoch))
            logging.info(
                '************************ model saved successful ************************** !'
            )
        if break_flag > args.patient:
            break
    # np.save(args.save_dir, np.array(adaptive_thresholds)) #save for plot


def train(data_loader, net, loss, epoch, optimizer, get_lr, save_dir, stats_path):
    start_time = time.time()
    net.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    total_train_loss = []
    class_predict = []
    class_target = []
    for i, sample in enumerate(data_loader):
        data = sample['image']
        target_c = sample['label_c']
        target_s = sample['label_s']
        data = data.to(DEVICE)
        target_c = target_c.to(DEVICE)
        target_s = target_s.to(DEVICE)
        output_s, output_c = net(data)
        optimizer.zero_grad()
        cur_loss, _, _, c_p = loss(output_s, output_c, target_s, target_c)
        with open(stats_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, i, cur_loss.item()])
        total_train_loss.append(cur_loss.item())
        class_target.append(target_c.detach().cpu().numpy())
        class_predict.append(c_p.detach().cpu().numpy())
        cur_loss.backward()
        optimizer.step()

    logging.info(
        'Epoch[%d], Batch [%d], total loss is %.6f, using %.1f s!' %
        (epoch, i, np.mean(total_train_loss), time.time() - start_time))
    total_train_class_predict = np.concatenate(class_predict, 0)
    total_train_class_target = np.concatenate(class_target, 0)
    adaptive_thresholds = get_threshold(total_train_class_predict,
                                        total_train_class_target, 0.995)
    cur_precision, _ = metric(total_train_class_predict, total_train_class_target,
                              adaptive_thresholds)
    logging.info(
        'Epoch[%d], [precision=%.4f, -->%.3f, -->%.3f, -->%.3f, -->%.3f]' %
        (epoch, np.mean(cur_precision), np.mean(cur_precision[0]), np.mean(cur_precision[1]),
         np.mean(cur_precision[2]), np.mean(cur_precision[3])))
    logging.info('the adaptive thresholds is [%.4f, %.4f, %.4f, %.4f]' %
                 (adaptive_thresholds[0], adaptive_thresholds[1],
                  adaptive_thresholds[2], adaptive_thresholds[3]))

    with open(stats_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, i, np.mean(total_train_loss), np.mean(cur_precision),
                         np.mean(cur_precision[0]), np.mean(cur_precision[1]), np.mean(cur_precision[2]),
                         np.mean(cur_precision[3])])

    return np.mean(total_train_loss), adaptive_thresholds


def evaluation(args, net, loss, epoch, save_dir, test_files, saved_thresholds, stats_path):
    cur_predict, cur_target, class_predict, class_target, eval_loss, start_time = run_on_evaluation_set(args,
                                                                                                        net,
                                                                                                        loss,
                                                                                                        epoch,
                                                                                                        save_dir,
                                                                                                        test_files,
                                                                                                        saved_thresholds,
                                                                                                        stats_path)

    return evaluate_test_results(cur_predict,
                                 cur_target,
                                 class_predict,
                                 class_target,
                                 saved_thresholds,
                                 epoch,
                                 start_time,
                                 eval_loss,
                                 stats_path)


def evaluate_test_results(cur_predict, cur_target, class_predict, class_traget, saved_thresholds, epoch, start_time,
                          eval_loss, stats_path):
    total_precision = []
    total_recall = []

    cur_precision, cur_recall = metric(
        class_predict, class_traget,
        saved_thresholds)
    total_precision.append(np.array(cur_precision))
    total_recall.append(np.array(cur_recall))
    TPVFs, dices, PPVs, FPVFs = segmentation_metrics(
        cur_predict, cur_target)
    logging.info(
        '***************************************************************************'
    )
    logging.info(
        'Esophagus --> Global dice is [%.5f], TPR is [%.5f], Precision is [%.5f] '
        % (dices[0], TPVFs[0], PPVs[0]))
    logging.info(
        'heart    --> Global dice is [%.5f], TPR is [%.5f], Precision is [%.5f] '
        % (dices[1], TPVFs[1], PPVs[1]))
    logging.info(
        'trachea  --> Global dice is [%.5f], TPR is [%.5f], Precision is [%.5f] '
        % (dices[2], TPVFs[2], PPVs[2]))
    logging.info(
        'aorta    --> Global dice is [%.5f], TPR is [%.5f], Precision is [%.5f] '
        % (dices[3], TPVFs[3], PPVs[3]))
    total_precision = np.stack(total_precision, 1)
    total_recall = np.stack(total_recall, 1)
    logging.info(
        'Epoch[%d], [precision=%.4f, -->%.3f, -->%.3f, -->%.3f, -->%.3f], using %.1f s!'
        % (epoch, np.mean(total_precision), np.mean(total_precision[0]),
           np.mean(total_precision[1]), np.mean(total_precision[2]), np.mean(total_precision[3]),
           time.time() - start_time))
    logging.info(
        'Epoch[%d], [recall=%.4f, -->%.3f, -->%.3f, -->%.3f, -->%.3f], using %.1f s!'
        % (epoch, np.mean(total_recall), np.mean(total_recall[0]),
           np.mean(total_recall[1]), np.mean(total_recall[2]), np.mean(total_recall[3]),
           time.time() - start_time))
    with open(stats_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, "NA", np.mean(eval_loss), np.mean(total_precision), np.mean(total_precision[0]),
                         np.mean(total_precision[1]), np.mean(total_precision[2]), np.mean(total_precision[3]),
                         np.mean(dices), dices[0], dices[1], dices[2], dices[3]])
    logging.info(
        'Epoch[%d], [total loss=%.6f], mean_dice=%.4f, using %.1f s!'
        % (epoch, np.mean(eval_loss), np.mean(dices),
           time.time() - start_time))
    logging.info(
        '***************************************************************************'
    )
    return np.mean(dices), np.mean(total_precision)


def run_on_evaluation_set(args, net, loss, epoch, save_dir, test_files, saved_thresholds, stats_path):
    net.eval()
    eval_loss = []
    start_time = time.time()

    composed_transforms_tr = transforms.Compose([
        tr.Normalize(mean=(0.12, 0.12, 0.12), std=(0.018, 0.018, 0.018)),
        tr.ToTensor2(args.n_class)
    ])
    eval_dataset = THOR_Data(
        transform=composed_transforms_tr,
        path=args.data_path,
        file_list=test_files
    )
    evalloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4)
    cur_target = []
    cur_predict = []
    class_predict = []
    class_target = []
    for i, sample in enumerate(evalloader):
        data = sample['image']
        target_c = sample['label_c']
        target_s = sample['label_s']
        data = data.to(DEVICE)
        target_c = target_c.to(DEVICE)
        target_s = target_s.to(DEVICE)
        with torch.no_grad():
            output_s, output_c = net(data)
            cur_loss, _, _, c_p = loss(output_s, output_c, target_s, target_c)
        eval_loss.append(cur_loss.item())
        with open(stats_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, i, cur_loss.item()])
        cur_target.append(torch.argmax(target_s, 1).cpu().numpy())
        cur_predict.append(torch.argmax(output_s, 1).cpu().numpy())
        class_target.append(target_c.cpu().numpy())
        class_predict.append(c_p.cpu().numpy())

    return np.concatenate(cur_predict, 0), np.concatenate(cur_target, 0), np.concatenate(class_predict,
                                                                                         0), np.concatenate(
        class_target, 0), eval_loss, start_time


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_dir = os.path.join(args.save_dir, args.model_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_dir = os.path.join(args.save_dir, str(args.test_flag))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s,%(lineno)d: %(message)s\n',
        datefmt='%Y-%m-%d(%a)%H:%M:%S',
        filename=os.path.join(args.save_dir, 'log.txt'),
        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    main(args)
