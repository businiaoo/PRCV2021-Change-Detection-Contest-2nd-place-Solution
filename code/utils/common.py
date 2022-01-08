import os
import torch
import random
import cv2
import numpy as np
from pathlib import Path
import matplotlib
from tqdm import tqdm
import torch.nn.functional as F

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn

from shutil import copytree, ignore_patterns


def result_visual(img1, img2, label1, label2, out1, out2):
    # a, b, c, d, e, f 为[H,W,3]
    if len(img1.shape) < 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) < 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    if len(label1.shape) < 3:
        label1 = cv2.cvtColor(label1, cv2.COLOR_GRAY2BGR)
    if len(label2.shape) < 3:
        label2 = cv2.cvtColor(label2, cv2.COLOR_GRAY2BGR)
    if len(out1.shape) < 3:
        out1 = cv2.cvtColor(out1, cv2.COLOR_GRAY2BGR)
    if len(out2.shape) < 3:
        out2 = cv2.cvtColor(out2, cv2.COLOR_GRAY2BGR)
    row_white = np.ones((10, img1.shape[0], 3)) * 255
    column_white = np.ones((img1.shape[1] * 2 + 10, 10, 3)) * 255

    left_part = np.concatenate([img1, row_white, img2], axis=0)
    middle_part = np.concatenate([label1, row_white, label2], axis=0)
    right_part = np.concatenate([out1, row_white, out2], axis=0)

    out = np.concatenate([left_part, column_white, middle_part, column_white, right_part], axis=1)

    # out = cv2.resize(out, (1024, 1024))
    return out


def plot_results(result_paths, save_dir=None, names=None):  # result_paths可以是列表，用来在一张图上同时画两次训练的数据
    if not isinstance(result_paths, list):
        result_paths = [result_paths]

    fig, ax = plt.subplots(3, 3, figsize=(20, 20), tight_layout=True)

    for result_path in result_paths:
        assert result_path.endswith(".txt"), 'please check path: {}'.format(result_path)
        if save_dir is None:
            save_dir = result_path.replace(result_path.split(os.sep)[-1], '')

        ax = ax.ravel()
        s = ['lr', 'P', 'R', 'F1', 'mIOU', 'OA', 'best_metric', 'train_loss', 'val_loss']
        results = np.loadtxt(result_path, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9], skiprows=1, ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(n)
        for i in range(len(s)):
            y = results[i, x]
            if i == 6:      # 对于best_metric，从0.5截断，以便在曲线上更好的展示出来最终的变化
                y[y < 0.5] = 0.5
                # y[y < 0.65] = 0.65
            ax[i].plot(x, y, marker='', label=s[i], linewidth=2, markersize=8)
            ax[i].set_title(s[i], fontsize=20)

    if names is None:
        names = result_paths
    ax[6].legend(names, loc='best')
    fig.savefig(Path(save_dir) / 'results.jpg', dpi=400)
    plt.close()
    del fig, ax


def init_seed(seed=777):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from torch.backends import cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def check_dirs():
    print("\n"+"-"*30+"Check Dirs"+"-"*30)
    if not os.path.exists('./runs'):
        os.mkdir('./runs')
        os.mkdir('./runs/train')
        os.mkdir('./runs/eval')
    file_names = os.listdir('./runs/train')
    file_names = [int(i) for i in file_names] + [0]
    new_file_name = str(max(file_names) + 1)

    save_path = './runs/train/' + new_file_name
    every_ckp_save_path = os.path.join(save_path, 'every_ckp')
    best_ckp_save_path = os.path.join(save_path, 'best_ckp')
    os.mkdir(save_path)
    os.mkdir(every_ckp_save_path)
    os.mkdir(best_ckp_save_path)
    print("checkpoints & results are saved at: {}".format(save_path))

    result_save_path = os.path.join(save_path, "result.txt")

    # 将代码也复制一份备份起来，copytree会自动创建目标文件夹
    code_path = os.path.join(save_path, 'code')
    copytree('./', code_path,
             ignore=ignore_patterns('runs', 'pretrain', 'infer'))
    print("The code has been backed up! Backup path: {}".format(code_path))

    best_ckp_file = None

    return save_path, best_ckp_save_path, best_ckp_file, result_save_path, every_ckp_save_path


def check_eval_dirs():
    print("\n"+"-"*30+"Check Dirs"+"-"*30)
    if not os.path.exists('./runs'):
        os.mkdir('./runs')
        os.mkdir('./runs/train')
        os.mkdir('./runs/eval')
    file_names = os.listdir('./runs/eval')
    file_names = [int(i) for i in file_names] + [0]
    new_file_name = str(max(file_names) + 1)
    save_path = './runs/eval/' + new_file_name
    os.mkdir(save_path)

    # print(save_path)
    result_save_path = os.path.join(save_path, "eval_result.txt")
    print("results are saved at: {}".format(save_path))

    # 将代码也复制一份备份起来，copytree会自动创建目标文件夹
    code_path = os.path.join(save_path, 'code')
    copytree('./', code_path,
             ignore=ignore_patterns('runs', 'pretrain', 'infer'))
    print("The code has been backed up! Backup path: {}".format(code_path))

    return save_path, result_save_path


def compute_p_r_f1_miou_oa(tn_fp_fn_tps):   # 计算各种指标
    p, r, f1, miou, oa = [], [], [], [], []
    for tn_fp_fn_tp in tn_fp_fn_tps:
        tn, fp, fn, tp = tn_fp_fn_tp
        p_tmp = tp / (tp + fp)
        r_tmp = tp / (tp + fn)
        miou_tmp = 0.5 * tp / (tp + fp + fn) + 0.5 * tn / (tn + fp + fn)
        oa_tmp = (tp + tn) / (tp + tn + fp + fn)

        p.append(p_tmp)
        r.append(r_tmp)
        f1.append(2 * p_tmp * r_tmp / (r_tmp + p_tmp))
        miou.append(miou_tmp)
        oa.append(oa_tmp)

    return np.array(p), np.array(r), np.array(f1), np.array(miou), np.array(oa)


def get_metrics_offline(gt_dir, pred_dir, dual_label=True):
    """
    根据两个文件夹中的图像，离线计算指标，
    :param dual_label: 是否为双标签，
    :param gt_dir:  真值标签的文件路径
    :param pred_dir:    模型预测结果的文件路径
    """
    tn_fp_fn_tp1, tn_fp_fn_tp2 = np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])

    files = os.listdir(os.path.join(gt_dir, "label1"))
    names = []

    for file in files:
        if "." in file:
            if file.split(".")[-1] in ["jpg", "bmp", "png", "jpeg"]:
                names.append(file)

    for name in tqdm(names):
        label1 = cv2.imread(os.path.join(gt_dir, "label1", name))[..., 0]   # 以三通道的形式读取进来，再取其中一个通道
        cd_pred1 = cv2.imread(os.path.join(pred_dir, "label1", name))[..., 0]
        if dual_label:
            label2 = cv2.imread(os.path.join(gt_dir, "label2", name))[..., 2]
            cd_pred2 = cv2.imread(os.path.join(pred_dir, "label2", name))[..., 2]
            labels = [label1, label2]
            cd_preds = [cd_pred1, cd_pred2]
        else:
            labels = [label1]
            cd_preds = [cd_pred1]

        for i, (cd_pred, label) in enumerate(zip(cd_preds, labels)):
            tn = ((cd_pred == 0) & (label == 0)).sum()
            fp = ((cd_pred == 255) & (label == 0)).sum()
            fn = ((cd_pred == 0) & (label == 255)).sum()
            tp = ((cd_pred == 255) & (label == 255)).sum()
            assert tn+tp+fn+fp == label1.shape[0]*label1.shape[1], "wrong"

            if i == 0:
                tn_fp_fn_tp1 += [tn, fp, fn, tp]
            elif i == 1:
                tn_fp_fn_tp2 += [tn, fp, fn, tp]
    if dual_label:
        p, r, f1, miou, oa = compute_p_r_f1_miou_oa([tn_fp_fn_tp1, tn_fp_fn_tp2])
    else:
        p, r, f1, miou, oa = compute_p_r_f1_miou_oa([tn_fp_fn_tp1])
    print("P:{}\nR:{}\nF1:{}\nF1-mean:{}\nmIOU:{}\nmIOU-mean:{}\nOA:{}"
          .format(p, r, f1, f1.mean(), miou, miou.mean(), oa))


# 打印GPU相关信息
def gpu_info():
    print("\n" + "-" * 30 + "GPU Info" + "-" * 30)
    gpu_count = torch.cuda.device_count()
    x = [torch.cuda.get_device_properties(i) for i in range(gpu_count)]
    s = 'Using CUDA '
    c = 1024 ** 2  # bytes to MB
    if gpu_count > 0:
        print("Using GPU count: {}".format(torch.cuda.device_count()))
        for i in range(0, gpu_count):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g name='%s', memory=%dMB" % (s, i, x[i].name, x[i].total_memory / c))
    else:
        print("Using CPU !!!")


def fill_hole(cd_pred, device):
    cd_pred = cd_pred.cpu().numpy()
    bs, h, w = cd_pred.shape
    im_out = None
    for one_img in range(bs):
        # img = cd_pred[one_img, ...].astype(np.uint8)*255
        im_in = cv2.cvtColor(cd_pred[one_img, ...].astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
        mask = np.zeros((h + 2, w + 2), np.uint8)
        im_floodfill = im_in.copy()

        # floodFill函数中的seedPoint对应像素必须是背景
        seedpoint = (0, 0)
        isbreak = False
        for i in range(im_in.shape[0]):
            for j in range(im_in.shape[1]):
                if im_in[i][j].all() == 0:
                    seedpoint = (i, j)
                    isbreak = True
                    break
            if isbreak:
                break

        # 得到im_floodfill 255填充非孔洞值
        cv2.floodFill(im_floodfill, mask, seedpoint, [255, 255, 255])

        # 得到im_floodfill的逆im_floodfill_inv
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
        im_out_tmp = im_in | im_floodfill_inv
        a = im_out_tmp[:, :, 0] - im_in[:, :, 0] > 0
        im_out_tmp[a] = [0, 0, 255]

        if im_out is None:
            im_out = im_out_tmp

        else:
            im_out = np.concatenate([im_out, im_out_tmp], 0)

    im_out = im_out.astype(np.float32) / 255
    im_out = torch.tensor(im_out.astype(np.int64)).to(device)
    return im_out


class SaveResult:
    def __init__(self, result_save_path):
        self.result_save_path = result_save_path

    def prepare(self):
        # 为写日志做准备，先创建这个文件
        with open(self.result_save_path, "w")as f:
            f.write(('%-7s'+'%-12s' * 9) % (
                'epoch', 'lr', 'P', 'R', 'F1', 'mIOU', 'OA', 'best_metric', 'train_loss', 'val_loss') + "\n")

    def show(self, p, r, f1, miou, oa,
             refer_metric=np.array(0), best_metric=0, train_avg_loss=0, val_avg_loss=0, lr=0, epoch=0):
        # 将这些数据指标打印出来，写进txt中，并画在图中
        print(
            "lr:{}  P:{}  R:{}  F1:{}  mIOU:{} OA:{}\nrefer_metric-mean: {} best_metric: {}".format(
                lr, p, r, f1, miou, oa, round(refer_metric.mean(), 5), round(best_metric, 5)))
        with open(self.result_save_path, "a")as f:
            f.write(
                ('%-7s'+'%-12s' * 9) % (str(epoch), str(round(lr, 8)),
                                        str(round(float(p.mean()), 6)),
                                        str(round(float(r.mean()), 6)),
                                        str(round(float(f1.mean()), 6)),
                                        str(round(float(miou.mean()), 6)),
                                        str(round(float(oa.mean()), 6)),
                                        str(round(float(best_metric), 6)),
                                        str(round(train_avg_loss, 6)),
                                        str(round(val_avg_loss, 6))) + "\n")

        plot_results(self.result_save_path)

    def save_first_batch(self, batch_img1, batch_img2, batch_label1, batch_label2, name=0):  # img直接乘255
        img1_1 = ((batch_img1[0, ...].cpu().numpy()) * 255).astype(np.int8)
        img1_2 = ((batch_img2[0, ...].cpu().numpy()) * 255).astype(np.int8)
        label1_1 = batch_label1[0, ...].cpu().numpy()
        label1_1 = np.array([label1_1, label1_1, label1_1]).astype(np.int8) * 255
        label1_2 = batch_label2[0, ...].cpu().numpy()
        label1_2 = np.array([label1_2, label1_2, label1_2]).astype(np.int8) * 255

        out1 = np.concatenate([img1_1, img1_2], 1)
        out2 = np.concatenate([label1_1, label1_2], 1)
        out11 = np.concatenate([out1, out2], 2).astype(np.uint8).transpose(1, 2, 0)

        img1_1 = ((batch_img1[1, ...].cpu().numpy()) * 255).astype(np.int8)
        img1_2 = ((batch_img2[1, ...].cpu().numpy()) * 255).astype(np.int8)
        label1_1 = batch_label1[1, ...].cpu().numpy()
        label1_1 = np.array([label1_1, label1_1, label1_1]).astype(np.int8) * 255
        label1_2 = batch_label2[1, ...].cpu().numpy()
        label1_2 = np.array([label1_2, label1_2, label1_2]).astype(np.int8) * 255

        out1 = np.concatenate([img1_1, img1_2], 1)
        out2 = np.concatenate([label1_1, label1_2], 1)
        out22 = np.concatenate([out1, out2], 2).astype(np.uint8).transpose(1, 2, 0)
        out = np.concatenate([out11, out22], 1)

        # print(os.path.join(os.path.dirname(self.result_save_path), "sample.jpg"))
        cv2.imwrite(os.path.join(os.path.dirname(self.result_save_path), "train_sample_{}.jpg".format(str(name))), out)


class CosOneCycle:  # 自己定义的策略没有state_dict,需要看源码是怎么做的
    def __init__(self, optimizer, max_lr, epochs, min_lr=None, up_rate=0.3):  # max=0.0035, min=0.00035
        self.optimizer = optimizer

        self.max_lr = max_lr
        if min_lr is None:
            self.min_lr = max_lr / 10
        else:
            self.min_lr = min_lr
        self.final_lr = self.min_lr / 50

        self.new_lr = self.min_lr

        self.step_i = 0
        self.epochs = epochs
        self.up_rate = up_rate   # 学习率上升的比例
        assert up_rate < 0.5, "up_rate should be smaller than 0.5"

    def step(self):
        # 先增后减
        self.step_i += 1
        if self.step_i < (self.epochs*self.up_rate):
            self.new_lr = 0.5 * (self.max_lr - self.min_lr) * (
                        np.cos((self.step_i/(self.epochs*self.up_rate) + 1) * np.pi) + 1) + self.min_lr
        else:
            self.new_lr = 0.5 * (self.max_lr - self.final_lr) * (np.cos(
                ((self.step_i - self.epochs * self.up_rate) / (
                            self.epochs * (1 - self.up_rate))) * np.pi) + 1) + self.final_lr

        if len(self.optimizer.state_dict()['param_groups']) == 1:
            self.optimizer.param_groups[0]["lr"] = self.new_lr
        elif len(self.optimizer.state_dict()['param_groups']) == 2:  # for finetune
            self.optimizer.param_groups[0]["lr"] = self.new_lr / 10
            self.optimizer.param_groups[1]["lr"] = self.new_lr
        else:
            raise Exception('Error. You need to add a new "elif". ')
        # print(new_lr)

    def plot_lr(self):
        all_lr = []
        for i in range(self.epochs):
            all_lr.append(self.new_lr)
            self.step()
        fig = seaborn.lineplot(x=range(self.epochs), y=all_lr)
        fig = fig.get_figure()
        fig.savefig('./lr_schedule.jpg', dpi=200)
        self.step_i = 0
        self.new_lr = self.min_lr


class ScaleInOutput:
    def __init__(self, input_size=512):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size
        self.output_size = None

    def scale_input(self, imgs: tuple):
        assert isinstance(imgs, tuple), "Please check the input type. It should be a 'tuple'."
        imgs = list(imgs)
        self.output_size = imgs[0].shape[2:]

        for i, img in enumerate(imgs):
            imgs[i] = F.interpolate(img, self.input_size, mode='bilinear', align_corners=True)

        return tuple(imgs)

    def scale_output(self, outs: tuple):
        if type(outs) in [torch.Tensor]:
            outs = (outs,)
        assert isinstance(outs, tuple), "Please check the input type. It should be a 'tuple'."
        outs = list(outs)

        assert self.output_size is not None, \
            "Please call 'scale_input' function firstly, to make sure 'output_size' is not None"

        for i, out in enumerate(outs):
            outs[i] = F.interpolate(out, self.output_size, mode='bilinear', align_corners=True)

        return tuple(outs)


if __name__ == "__main__":
    # result_path = [
    #     '../runs/train/36/result.txt',
    #     '../runs/train/37/result.txt',
    # ]

    result_path = [
        '../runs/train/7/result.txt',
        '../runs/train/8/result.txt',
        '../runs/train/9/result.txt',
        '../runs/train/10/result.txt',
        '../runs/train/11/result.txt',
        # '../runs/train/7/result.txt',
    ]
    # result_path = '../runs/train/17/result.txt'
    save_dir = "./"
    # names = ['cswin_t_w64', 'cswin_s_w64', 'efficientnetv2_m_w40', 'cswin_b384_w96', 'cswin_b448_w96', 'cswin_b_w64']
    # plot_results(result_path, save_dir, names)
    plot_results(result_path, save_dir)

    # result_path = '../../others/FCS-cdgame/runs/eval/11/eval_result.txt'
    # plot_eval_results(result_path)

