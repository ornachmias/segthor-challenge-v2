import os
import time

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import re
from skimage import filters


def get_cross_validation_paths(test_flag):
    """
    we use 4-fold cross-validation for testing
    we split data in order 0..9, 10..19, 20..29, 30..39
    Args:
        test_flag: range in [0, 3]
    """
    assert test_flag > -1 and test_flag < 4, 'the test flag is not in range !'
    train_files = []
    test_files = []
    test_nums = [i for i in range(test_flag * 10 + 1, test_flag * 10 + 11)]
    for i in range(1, 41):
        flag = False
        if i in test_nums:
            flag = True
        if i < 10:
            i = '0' + str(i)
        else:
            i = str(i)
        if flag:
            test_files.append("Patient_" + i)
        else:
            train_files.append("Patient_" + i)
    return train_files, test_files


def get_global_alpha(patient_ids, data_path):
    """
    given the patient ids, we count the data number of each class and then
    we calculate the global condition probability of each class
    the condition probability: P(B|A) --> P[A, B] in the code
    the return value is the alpha
    """
    total_patients = []
    for patient_id in os.listdir(data_path):
        total_patients.append(patient_id)
    n_class = 4  # four organs including Esophagus, heart, trachea, aorta
    events = np.zeros((n_class, n_class))
    P = np.zeros((n_class, n_class))
    for patient_id in patient_ids:
        if not patient_id in total_patients:
            assert 'there is a not found patient, please check the data !'
        cur_label_path = os.path.join(data_path, patient_id)
        for img_file in os.listdir(cur_label_path):
            if 'label.npy' in img_file:
                img_path = os.path.join(cur_label_path, img_file)
                cur_label = np.load(img_path)
                event = np.zeros((n_class, 1))
                for i in range(n_class):
                    if np.sum(cur_label == i + 1) > 0:
                        event[i] = 1
                events += np.dot(event, event.transpose(1, 0))
    for i in range(n_class):
        for j in range(n_class):
            if i == j:
                continue
            P[i, j] = events[i, j] / float(events[i, i])
    alpha = np.copy(P)
    for i in range(n_class):
        for j in range(n_class):
            if i == j:
                continue
            alpha[i, i] -= alpha[j, i]
        alpha[i, i] += n_class
    return alpha


class THOR_Data(Dataset):
    '''
        parameters: 
            transform: the data augmentation methods
            path: the processed data path (training or testing)
        functions:
    '''

    def __init__(self, transform=None, path=None, file_list=None, otsu=0):
        data_listdirs = os.listdir(path)
        data_files = []
        label_files = []
        for cur_listdir in data_listdirs:
            if not cur_listdir in file_list:
                continue
            cur_file_dir = os.path.join(path, cur_listdir)
            for cur_file_image in os.listdir(cur_file_dir):
                if 'image.npy' in cur_file_image:
                    data_files.append(os.path.join(cur_file_dir, cur_file_image))
                    label_files.append(
                        os.path.join(
                            cur_file_dir,
                            cur_file_image.split('image.npy')[0] + 'label.npy'))
        self.data_files = []
        self.label_files = []
        self.db = {}

        data_files = sorted(data_files, key=lambda x: self.get_slice_id(x))
        label_files = sorted(label_files, key=lambda x: self.get_slice_id(x))
        for i in range(len(data_files)):
            self.data_files.append(data_files[i])
            self.label_files.append(label_files[i])
        self.transform = transform
        self.run_otsu = otsu

        assert (len(self.data_files) == len(self.label_files))
        print('the data length is %d' % len(self.data_files))

    def get_slice_id(self, path):
        m = re.search('Patient_(\d+)_(\d+)_', path)
        if m:
            return int(m.group(1)), int(m.group(2))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        if index in self.db:
            return self.db[index]
        data_file = self.data_files[index]
        patient_id, slice_id = self.get_slice_id(data_file)
        slice_id = slice_id / 300

        _img = np.load(data_file)
        h = _img.shape[0]
        w = _img.shape[1]
        index_channel = np.ones((h, w)) * slice_id

        if self.run_otsu == 1:
            middle_img = _img[:, :, 1]
            if len(np.unique(middle_img)) > 1:
                val = filters.threshold_otsu(middle_img)
                _img[_img[:, :, 1] < val] = 0
        elif self.run_otsu == 2:
            num_thresholds = 10
            middle_img = np.copy(_img[:, :, 1])
            if len(np.unique(middle_img)) > num_thresholds:
                thresholds = filters.threshold_multiotsu(middle_img, classes=num_thresholds, nbins=1000)
                middle_img[_img[:, :, 1] < thresholds[0]] = 0
                middle_img[_img[:, :, 1] > thresholds[-1]] = 1

                for i in range(1, len(thresholds) - 1):
                    middle_img[thresholds[i] < _img[:, :, 1] <= thresholds[i + 1]] = i / num_thresholds

                _img[:, :, 1] = middle_img

        _img = Image.fromarray(_img)

        _target = np.load(self.label_files[index])
        _target = Image.fromarray(np.uint8(_target))
        patient, slice = self.get_slice_id(self.data_files[index])
        sample = {'image': _img, 'label': _target}
        if self.transform is not None:
            sample = self.transform(sample)
        sample.update({"patient": patient, "slice": slice, 'index': index_channel})
        self.db[index] = sample
        return sample

    def __str__(self):
        pass

    def get_patient_data(self, dir_name):
        result = []
        data_files = [x for x in self.data_files if dir_name in x]
        label_files = [x for x in self.label_files if dir_name in x]
        for i in range(len(data_files)):
            _img = np.load(data_files[i])
            _img = Image.fromarray(_img)
            _target = np.load(label_files[i])
            _target = Image.fromarray(np.uint8(_target))
            sample = {'image': _img, 'label': _target}
            if self.transform is not None:
                sample = self.transform(sample)

            result.append(sample)

        return result
