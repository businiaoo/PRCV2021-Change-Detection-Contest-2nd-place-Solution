import os
import random
from PIL import Image
import torch
import torch.utils.data as data

from utils import transforms as tr


def get_loaders(opt):   # train.py调用这个

    train_dataset = CDDloader(opt, 'train', aug=True)
    val_dataset = CDDloader(opt, 'val', aug=False)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.num_workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    return train_loader, val_loader


def get_eval_loaders(opt):    # eval.py调用这个
    dataset_name = "val"
    print("using dataset: {} set".format(dataset_name))
    eval_dataset = CDDloader(opt, dataset_name, aug=False)
    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=False,
                                              num_workers=opt.num_workers)
    return eval_loader


def get_infer_loaders(opt):
    infer_datast = CDDloadImageOnly(opt, '', aug=False)
    infer_loader = torch.utils.data.DataLoader(infer_datast,
                                               batch_size=opt.batch_size,
                                               shuffle=False,
                                               num_workers=opt.num_workers)
    return infer_loader


class CDDloader(data.Dataset):

    def __init__(self, opt, phase, aug=False):
        self.data_dir = str(opt.dataset_dir)
        self.dual_label = opt.dual_label   # 双标签
        self.phase = str(phase)
        self.aug = aug
        names = [i for i in os.listdir(os.path.join(self.data_dir, phase, 'Image1'))]
        self.names = []
        for name in names:
            if is_img(name):
                self.names.append(name)

        random.shuffle(self.names)

    def __getitem__(self, index):

        name = str(self.names[index])
        img1 = Image.open(os.path.join(self.data_dir, self.phase, 'Image1', name))
        img2 = Image.open(os.path.join(self.data_dir, self.phase, 'Image2', name))
        label_name = name.replace("tif", "png") if name.endswith("tif") else name   # for shengteng
        label1 = Image.open(os.path.join(self.data_dir, self.phase, 'label1', label_name))

        if self.dual_label:
            label2 = Image.open(os.path.join(self.data_dir, self.phase, 'label2', label_name))
        else:   # 如果是单标签的话，就让label2等于label1，然后按照双标签做数据增强
            label2 = label1

        if self.aug:
            img1, img2, label1, label2 = tr.with_augment_transforms([img1, img2, label1, label2])
        else:
            img1, img2, label1, label2 = tr.without_augment_transforms([img1, img2, label1, label2])

        return img1, img2, label1, label2, name

    def __len__(self):
        return len(self.names)


def is_img(name):
    img_format = ["jpg", "png", "jpeg", "bmp", "tif", "tiff", "TIF", "TIFF"]
    if "." not in name:
        return False
    if name.split(".")[-1] in img_format:
        return True
    else:
        return False


# 对于单标签和双标签这个函数都是这个，因为推理的时候没有标签，只需要加载图像
class CDDloadImageOnly(data.Dataset):

    def __init__(self, opt, phase, aug=False):
        self.data_dir = str(opt.dataset_dir)
        self.phase = str(phase)
        self.aug = aug
        names = [i for i in os.listdir(os.path.join(self.data_dir, phase, 'Image1'))]
        self.names = []
        for name in names:
            if is_img(name):
                self.names.append(name)
        # random.shuffle(self.names)  # test不需要洗牌

    def __getitem__(self, index):

        name = str(self.names[index])
        img1 = Image.open(os.path.join(self.data_dir, self.phase, 'Image1', name))
        img2 = Image.open(os.path.join(self.data_dir, self.phase, 'Image2', name))

        img1, img2 = tr.infer_transforms([img1, img2])

        return img1, img2, name

    def __len__(self):
        return len(self.names)

