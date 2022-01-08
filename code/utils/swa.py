import time
from copy import deepcopy
import torch
import os
import sys
sys.path.append("../")


def swa(model_dir, average_list, save_dir):

    print("model_dir: {}".format(model_dir))
    print("average_list: {}".format(average_list))
    print("save_dir: {}".format(save_dir))

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model_names = ["epoch_"+str(i)+".pt" for i in average_list]
    print("model_names: {}".format(model_names))
    print("-"*100)

    model_paths = [os.path.join(model_dir, model_name) for model_name in model_names]
    models = [torch.load(model_path, map_location='cpu') for model_path in model_paths]
    model_keys = models[0].state_dict().keys()
    model_num = len(models)

    new_state_dict = deepcopy(models[0].state_dict())

    for model_key in model_keys:
        sum_weight = 0.0
        for m in models:
            sum_weight += m.state_dict()[model_key]
        avg_weight = sum_weight / model_num
        new_state_dict[model_key] = avg_weight

    new_model = deepcopy(models[int(model_num/2)])

    new_model.load_state_dict(new_state_dict)   # 这个是关键
    save_model_name = 'swa_' + str(average_list[0]) + '_' + str(len(average_list)) + '.pt'
    torch.save(new_model, os.path.join(save_dir, save_model_name))
    print("model is saved at {}".format(os.path.join(save_dir, save_model_name)))


# 优化上边的swa，更加快速，适用于比如：计算完1-5epoch的平均权重，再计算1-6的时候，直接加载之前计算好的1-5的平均权重，而不是重新从1-6再计算
def swa2(model_dir, average_list, save_dir, weight_name):

    print("model_dir: {}".format(model_dir))
    print("average_list: {}".format(average_list))
    print("save_dir: {}".format(save_dir))

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_model_name = weight_name + str(average_list[0]) + '_' + str(len(average_list)) + '.pt'
    last_save_model_name = weight_name + str(average_list[0]) + '_' + str(len(average_list)-1) + '.pt'
    last_exist = False   # 上一个文件是否存在

    if os.path.isfile(os.path.join(save_dir, last_save_model_name)):
        model_paths = [os.path.join(save_dir, last_save_model_name),
                       # os.path.join(model_dir, "epoch_"+str(average_list[-1])+".pt")]
                       os.path.join(model_dir, weight_name+str(average_list[-1])+".pt")]
        last_exist = True
        print("last_exist: {}".format(last_exist))
    else:
        # model_names = ["epoch_"+str(i)+".pt" for i in average_list]
        model_names = [weight_name+str(i)+".pt" for i in average_list]
        model_paths = [os.path.join(model_dir, model_name) for model_name in model_names]

    print("-" * 100)
    models = [torch.load(model_path, map_location='cpu') for model_path in model_paths]
    model_keys = models[0].state_dict().keys()
    model_num = len(average_list)

    new_state_dict = deepcopy(models[0].state_dict())

    for model_key in model_keys:
        sum_weight = 0.0
        if last_exist:
            sum_weight = models[0].state_dict()[model_key]*(model_num-1) + models[1].state_dict()[model_key]
        else:
            for m in models:
                sum_weight += m.state_dict()[model_key]
        avg_weight = sum_weight / model_num
        new_state_dict[model_key] = avg_weight

    new_model = deepcopy(models[0])

    new_model.load_state_dict(new_state_dict)   # 这个是关键

    torch.save(new_model, os.path.join(save_dir, save_model_name))
    print("model is saved at {}".format(os.path.join(save_dir, save_model_name)))


# 最终在retrain_for_swa.py调用，用这个版本的SWA
def weights_swa(last_swa_ckp_path, last_ckp_path, epoch):
    """

    :param last_swa_ckp_path:  最新的swa的模型
    :param last_ckp_path:    最新的正常训练的模型
    :param epoch:  到现在为止的迭代次数
    :return:   移动平均之后的模型权重
    """

    assert os.path.isfile(last_swa_ckp_path), "{} is not exist".format(last_swa_ckp_path)
    assert os.path.isfile(last_ckp_path), "{} is not exist".format(last_ckp_path)

    last_swa_ckp = torch.load(last_swa_ckp_path)   # GPU的权重加载进来在GPU上，CPU的在CPU上
    last_ckp = torch.load(last_ckp_path)

    model_keys = last_swa_ckp.state_dict().keys()

    new_state_dict = deepcopy(last_swa_ckp.state_dict())

    for model_key in model_keys:

        sum_weight = last_swa_ckp.state_dict()[model_key]*epoch + last_ckp.state_dict()[model_key]

        avg_weight = sum_weight / (epoch + 1)
        new_state_dict[model_key] = avg_weight

    new_model = deepcopy(last_swa_ckp)
    new_model.load_state_dict(new_state_dict)   # 这个是关键
    return new_model


if __name__ == "__main__":
    t0 = time.time()
    model_dir = "../runs/train/46/every_ckp"
    average_lists = []
    for i in range(110, 170):   # 这个代表长度
        average_lists.append([j for j in range(110, i+1)])
    print(average_lists)
    save_dir = "../runs/train/46/swa2_ckp"
    weight_name = "hrnet_w18_concat_fcs_epoch_"

    for average_list in average_lists:
        swa2(model_dir, average_list, save_dir, weight_name)
    print("Done!")
    print("cost time: {}".format(time.time()-t0))
