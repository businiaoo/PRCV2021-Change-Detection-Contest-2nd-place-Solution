import os
import torch
import argparse
from tqdm import tqdm

from eval import eval_for_metric
from losses.get_losses import SelectLoss
from models.block.Drop import dropblock_step
from utils.dataloaders import get_loaders
from utils.common import check_dirs, init_seed, gpu_info, SaveResult, CosOneCycle, ScaleInOutput
from models.main_model import ChangeDetection, ModelEMA, ModelSWA


def train(opt):
    init_seed()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    gpu_info()  # 打印GPU信息
    save_path, best_ckp_save_path, best_ckp_file, result_save_path, every_ckp_save_path = check_dirs()

    save_results = SaveResult(result_save_path)
    save_results.prepare()

    train_loader, val_loader = get_loaders(opt)
    scale = ScaleInOutput(opt.input_size)

    model = ChangeDetection(opt).cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    criterion = SelectLoss(opt.loss)

    if opt.finetune:
        params = [{"params": [param for name, param in model.named_parameters()
                              if "backbone" in name], "lr": opt.learning_rate / 10},  # 微调backbone
                  {"params": [param for name, param in model.named_parameters()
                              if "backbone" not in name], "lr": opt.learning_rate}]  # 其它层正常学习
        print("Using finetune for model")
    else:
        params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=opt.learning_rate, weight_decay=0.001)
    if opt.pseudo_label:
        scheduler = CosOneCycle(optimizer, max_lr=opt.learning_rate/5, epochs=opt.epochs, up_rate=0)
    else:
        scheduler = CosOneCycle(optimizer, max_lr=opt.learning_rate, epochs=opt.epochs)  # 自己定义的onecycle

    best_metric = 0
    train_avg_loss = 0
    total_bs = 16
    accumulate_iter = max(round(total_bs / opt.batch_size), 1)
    print("Accumulate_iter={} batch_size={}".format(accumulate_iter, opt.batch_size))

    for epoch in range(opt.epochs):
        model.train()
        train_tbar = tqdm(train_loader)
        for i, (batch_img1, batch_img2, batch_label1, batch_label2, _) in enumerate(train_tbar):
            train_tbar.set_description("epoch {}, train_loss {}".format(epoch, train_avg_loss))
            if epoch == 0 and i < 20:
                save_results.save_first_batch(batch_img1, batch_img2, batch_label1, batch_label2, i)
            if opt.pseudo_label and epoch == 0:
                print("---Using Pseudo labels, skip the first epoch!---")
                break

            batch_img1 = batch_img1.float().cuda()
            batch_img2 = batch_img2.float().cuda()
            batch_label1 = batch_label1.long().cuda()
            batch_label2 = batch_label2.long().cuda()

            batch_img1, batch_img2 = scale.scale_input((batch_img1, batch_img2))   # 指定某个尺度进行训练
            outs = model(batch_img1, batch_img2)
            outs = scale.scale_output(outs)

            loss = criterion(outs, (batch_label1, batch_label2)) if model.dl else criterion(outs, (batch_label1,))
            train_avg_loss = (train_avg_loss * i + loss.cpu().detach().numpy()) / (i + 1)

            loss.backward()
            if ((i+1) % accumulate_iter) == 0:
                optimizer.step()
                optimizer.zero_grad()

            del batch_img1, batch_img2, batch_label1, batch_label2

        scheduler.step()
        dropblock_step(model)
        # swa.update(model)
        # swa.save(every_ckp_save_path)

        # 测试val_set上的指标
        # if swa.swa_model is not None:   # 评测swa的指标
        #     p, r, f1, miou, oa, val_avg_loss = eval_for_metric(swa.swa_model, val_loader, criterion, input_size=opt.input_size)
        #     print("swa—results: mIOU:{} OA:{}".format(miou.mean(), oa.mean()))
        p, r, f1, miou, oa, val_avg_loss = eval_for_metric(model, val_loader, criterion, input_size=opt.input_size)

        # refer_metric = f1
        refer_metric = miou
        underscore = "_"
        if refer_metric.mean() > best_metric:
            if best_ckp_file is not None:
                os.remove(best_ckp_file)
            best_ckp_file = os.path.join(
                best_ckp_save_path,
                underscore.join([opt.backbone, opt.neck, opt.head, 'epoch',
                                 str(epoch), str(round(float(refer_metric.mean()), 5))]) + ".pt")
            torch.save(model, best_ckp_file)
            best_metric = refer_metric.mean()

        # 写日志
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        save_results.show(p, r, f1, miou, oa, refer_metric, best_metric, train_avg_loss, val_avg_loss, lr, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Change Detection train')

    # 配置模型
    parser.add_argument("--backbone", type=str, default="tf_efficientnetv2_s_in21k_w40")
    parser.add_argument("--neck", type=str, default="fpn+aspp+fuse+drop")
    parser.add_argument("--head", type=str, default="fcn")
    parser.add_argument("--loss", type=str, default="bce+dice")

    # 配置训练参数
    parser.add_argument("--pretrain", type=str,
                        default="./runs/train/15/best_ckp/tf_efficientnetv2_s_in21k_w40_fpn+aspp+fuse+drop_fcn_swa_73_0.88683.pt")  # 预训练权重路径
    parser.add_argument("--cuda", type=str, default="7")
    parser.add_argument("--dataset-dir", type=str, default="/zq2/dataset/CD-dataset/whu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--input-size", type=int, default=352)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0035)
    parser.add_argument("--dual-label", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=True)
    parser.add_argument("--pseudo-label", type=bool, default=True)

    opt = parser.parse_args()
    print("\n" + "-" * 30 + "OPT" + "-" * 30)
    print(opt)

    train(opt)
