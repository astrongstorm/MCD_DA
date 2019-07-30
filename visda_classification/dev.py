import random
import torch
import math
import sys
import psutil
import gc
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as util_data
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import torch.utils.data as util_data
from data_list import ImageList
import pre_process as prep
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import seperate_data
from basenet import *
from mlp_network import MLP
import mlp_network

def get_dev_risk(weight, error):
    """
    :param weight: shape [N, 1], the importance weight for N source samples in the validation set
    :param error: shape [N, 1], the error value for each source sample in the validation set
    (typically 0 for correct classification and 1 for wrong classification)
    """
    N, d = weight.shape
    _N, _d = error.shape
    assert N == _N and d == _d, 'dimension mismatch!'
    weighted_error = weight * error  # weight correspond to Ntr/Nts, error correspond to validation error
    cov = np.cov(np.concatenate((weighted_error, weight), axis=1), rowvar=False)[0][1]
    var_w = np.var(weight, ddof=1)
    if cov == 0 and var_w == 0:
        cov = var_w = 0.00001
    if var_w == 0:
        var_w = cov # 1/(2.3* cov) or 1.6
    eta = - cov / var_w
    return np.mean(weighted_error) + eta * np.mean(weight) - eta


def get_weight(source_feature_path, target_feature_path,
               validation_feature_path):  # 这三个feature根据类别不同，是不一样的. source与target这里需注意一下数据量threshold 2倍的事儿
    """
    :param source_feature: shape [N_tr, d], features from training set
    :param target_feature: shape [N_te, d], features from test set
    :param validation_feature: shape [N_v, d], features from validation set
    :return:
    """
    print("Start calculating weight")

    print("Loading feature files")
    source_feature = np.load(source_feature_path)
    target_feature = np.load(target_feature_path)
    validation_feature_np = np.load(validation_feature_path)

    N_s, d = source_feature.shape
    N_t, _d = target_feature.shape
    if float(N_s) / N_t > 2:
        source_feature = random_select_src(source_feature, target_feature)
    else:
        source_feature = source_feature.copy()

    print('num_source is {}, num_target is {}, ratio is {}\n'.format(N_s, N_t, float(N_s) / N_t))  # check the ratio

    N_s, d = source_feature.shape
    target_feature = target_feature.copy()
    all_feature = np.concatenate((source_feature, target_feature))
    all_label = np.asarray([1] * N_s + [0] * N_t, dtype=np.int32)  # 1->source 0->target

    feature_for_train_np, feature_for_test_np, label_for_train_np, label_for_test_np = train_test_split(all_feature, all_label,
                                                                                            train_size=0.8)

    # here is train, test split, concatenating the data from source and target

    decays = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 0.0005]
    val_acc = []
    domain_classifiers = []

    # created the MLP network
    MLP = mlp_network.MLP(d, 2).cuda()
    loss_function = nn.CrossEntropyLoss()

    # convert all numpy to variables
    feature_for_train = Variable(torch.from_numpy(feature_for_train_np)).cuda()
    feature_for_test = Variable(torch.from_numpy(feature_for_test_np)).cuda()
    label_for_train = Variable(torch.from_numpy(label_for_train_np).long()).cuda()
    label_for_test = Variable(torch.from_numpy(label_for_test_np).long()).cuda()
    validation_feature = Variable(torch.from_numpy(validation_feature_np)).cuda()
    print("start training")
    for decay in decays:
        print("Decay: {}".format(decay))
        for ep in range(1, 1001):
            optimizer = torch.optim.Adam(MLP.parameters(), lr=0.001, weight_decay=decay)
            pred = MLP(feature_for_train)
            loss = loss_function(pred, label_for_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # check training accuracy
            if ep % 100 == 0:
                print("Check for testing set accuracy")
                pred_test = MLP(feature_for_test)
                predicted_test = torch.max(F.softmax(pred_test), dim=1)[1]
                pred_y = predicted_test.detach().cpu().numpy().squeeze()
                label_y = label_for_test.detach().cpu().numpy()
                accuracy = sum(pred_y == label_y) / label_y.size
                print("Accuracy is {}".format(accuracy))
        if not os.path.exists(source_feature_path.split("/")[0] + "/MLP_weight/"):
            os.makedirs(source_feature_path.split("/")[0] + "/MLP_weight/")
        pre_path = source_feature_path.split("/")[0] + "/MLP_weight/" + "MLP" + str(accuracy)
        pre_path = "MLP" + str(accuracy)
        path = pre_path.replace(".", "_") + ".pth"
        torch.save(MLP, path)
        domain_classifiers.append(path)
        val_acc.append(accuracy)

    index = val_acc.index(max(val_acc))
    path_2_load = domain_classifiers[index]

    Best_MLP = torch.load(path_2_load)

    out = Best_MLP(validation_feature)
    domain_out = out.detach().cpu().numpy()
    print("Domain out: {}".format(domain_out))
    return domain_out[:, :1] / domain_out[:, 1:] * N_s * 1.0 / N_t  # (Ntr/Nts)*(1-M(fv))/M(fv)

def random_select_src(source_feature, target_feature):
    # done with debugging
    """
    Select at most 2*Ntr data from source feature randomly
    :param source_feature: shape [N_tr, d], features from training set
    :param target_feature: shape [N_te, d], features from test set
    :return:
    """
    N_s, d = source_feature.shape
    N_t, _d = target_feature.shape
    items = [i for i in range(1, N_s)]
    random_list = random.sample(items, 2 * N_t - 1)
    new_source_feature = source_feature[0].reshape(1, d)
    for i in range(2 * N_t - 1):
        new_source_feature = np.concatenate((new_source_feature, source_feature[random_list[i]].reshape(1, d)))

    print("random_select:")
    print(new_source_feature.shape)
    return new_source_feature


def predict_loss(cls, y_pre):
    # done with debugging works fine
    """
    Calculate the cross entropy loss for prediction of one picture
    :param cls:
    :param y_pre:
    :return:
    """
    cls_torch = np.full(1, cls)
    pre_cls_torch = y_pre.double()
    target = torch.from_numpy(cls_torch).cuda()
    entropy = nn.CrossEntropyLoss()
    return entropy(pre_cls_torch, target).detach().cpu()

def get_label_list(args, target_list, feature_network_path, predict_network_path, num_layer, resize_size, crop_size, batch_size, use_gpu):
    """
    Return the target list with pesudolabel
    :param target_list: list conatinging all target file path and a wrong label
    :param predict_network: network to perdict label for target image
    :param resize_size:
    :param crop_size:
    :param batch_size:
    :return:
    """
    option = 'resnet' + args.resnet
    G = ResBase(option)
    F1 = ResClassifier(num_layer=num_layer)

    G.load_state_dict(torch.load(feature_network_path))
    F1.load_state_dict(torch.load(predict_network_path))
    if use_gpu:
        G.cuda()
        F1.cuda()
    G.eval()
    F1.eval()

    label_list = []
    dsets_tar = ImageList(target_list, transform=prep.image_train(resize_size=resize_size, crop_size=crop_size))
    dset_loaders_tar = util_data.DataLoader(dsets_tar, batch_size=batch_size, shuffle=False, num_workers=4)
    len_train_target = len(dset_loaders_tar)
    iter_target = iter(dset_loaders_tar)
    count = 0
    for i in range(len_train_target):
        input_tar, label_tar = iter_target.next()
        if use_gpu:
            input_tar, label_tar = Variable(input_tar).cuda(), Variable(label_tar).cuda()
        else:
            input_tar, label_tar = Variable(input_tar), Variable(label_tar)
        tar_feature = G(input_tar)
        predict_score = F1(tar_feature)
        _, pre_lab = torch.max(predict_score, 1)
        predict_label = pre_lab.detach()
        for num in range(len(predict_label.cpu())):
            if target_list[count][-3] == ' ':
                ind = -2
            else:
                ind = -3
            label_list.append(target_list[count][:ind])
            label_list[count] = label_list[count] + str(predict_label[num].cpu().numpy()) + "\n"
            count += 1
    return label_list


def cross_validation_loss(args, feature_network_path, predict_network_path, num_layer, src_list, target_path, val_list, class_num,
                          resize_size, crop_size, batch_size, use_gpu):
    """
    Main function for computing the CV loss
    :param feature_network:
    :param predict_network:
    :param src_cls_list:
    :param target_path:
    :param val_cls_list:
    :param class_num:
    :param resize_size:
    :param crop_size:
    :param batch_size:
    :return:
    """
    target_list_no_label = open(target_path).readlines()
    cross_val_loss = 0
    save_path = args.save
    # load network
    option = 'resnet' + args.resnet
    G = ResBase(option)
    F1 = ResClassifier(num_layer=num_layer)
    G.load_state_dict(torch.load(feature_network_path))
    F1.load_state_dict(torch.load(predict_network_path))
    if use_gpu:
        G.cuda()
        F1.cuda()
    G.eval()
    F1.eval()
    print("Loaded network")
    # add pesudolabel for target data
    print("Sperating target data")
    tar_list = []
    dsets_tar = ImageList(target_list_no_label,
                          transform=prep.image_train(resize_size=resize_size, crop_size=crop_size))
    dset_loaders_tar = util_data.DataLoader(dsets_tar, batch_size=batch_size, shuffle=False, num_workers=4)
    len_train_target = len(dset_loaders_tar)
    iter_target = iter(dset_loaders_tar)
    count = 0
    for i in range(len_train_target):
        input_tar, label_tar = iter_target.next()
        if use_gpu:
            input_tar, label_tar = Variable(input_tar).cuda(), Variable(label_tar).cuda()
        else:
            input_tar, label_tar = Variable(input_tar), Variable(label_tar)
        tar_feature = G(input_tar)
        predict_score = F1(tar_feature)
        _, pre_lab = torch.max(predict_score, 1)
        predict_label = pre_lab.detach()
        for num in range(len(predict_label.cpu())):
            if target_list_no_label[count][-3] == ' ':
                ind = -2
            else:
                ind = -3
            tar_list.append(target_list_no_label[count][:ind])
            tar_list[count] = tar_list[count] + str(predict_label[num].cpu().numpy()) + "\n"
            count += 1
    val_list = seperate_data.dimension_rd(val_list)
    print("Seperated")
    # load the dataloader for whole data
    prep_dict = prep.image_train(resize_size=resize_size, crop_size=crop_size)
    dsets_src = ImageList(src_list, transform=prep_dict)
    dset_loaders_src = util_data.DataLoader(dsets_src, batch_size=batch_size, shuffle=True, num_workers=4)
    dsets_val = ImageList(val_list, transform=prep_dict)
    dset_loaders_val = util_data.DataLoader(dsets_val, batch_size=batch_size, shuffle=True, num_workers=4)
    dsets_tar = ImageList(tar_list, transform=prep_dict)
    dset_loaders_tar = util_data.DataLoader(dsets_tar, batch_size=batch_size, shuffle=True, num_workers=4)

    # iterate through different classes
    for cls in range(class_num):
        # prepare source feature
        count_src = 0
        src_feature_de = np.array([])
        iter_src = iter(dset_loaders_src)
        while src_feature_de.size == 0:
            src_input, src_labels = iter_src.next()
            for i in range(len(src_labels)):
                if src_labels[i].item() == cls:
                    a, b, c = src_input[i].shape
                    if use_gpu:
                        src_pre_input = Variable(src_input[i]).cuda()
                    else:
                        src_pre_input = Variable(src_input[i])
                    src_input_final = src_pre_input.reshape(1, a, b, c)
                    if src_feature_de.size == 0:
                        src_feature = G(src_input_final)
                        src_feature_de = src_feature.detach().detach().cpu().numpy()
                    else:
                        src_feature_new = G(src_input_final)
                        src_feature_new_de = src_feature_new.detach().cpu().numpy()
                        src_feature_de = np.append(src_feature_de, src_feature_new_de, axis=0)
            count_src = count_src + 1
        for _ in range(len(dset_loaders_src) - count_src):
            src_input, src_labels = iter_src.next()
            for i in range(len(src_labels)):
                if src_labels[i].item() == cls:
                    a, b, c = src_input[i].shape
                    if use_gpu:
                        src_pre_input = Variable(src_input[i]).cuda()
                    else:
                        src_pre_input = Variable(src_input[i])
                    src_input_final = src_pre_input.reshape(1, a, b, c)
                    src_feature_new = G(src_input_final)
                    src_feature_new_de = src_feature_new.detach().cpu().numpy()
                    src_feature_de = np.append(src_feature_de, src_feature_new_de, axis=0)
        print("Pass Source for Class{}".format(cls + 1))
        print("Created feature: {}".format(src_feature_de.shape))

        # prepare target fature
        count_tar = 0
        tar_feature_de = np.array([])
        iter_tar = iter(dset_loaders_tar)
        while tar_feature_de.size == 0:
            tar_input, tar_labels = iter_tar.next()
            for i in range(len(tar_labels)):
                if tar_labels[i].item() == cls:
                    a, b, c = tar_input[i].shape
                    if use_gpu:
                        tar_pre_input = Variable(tar_input[i]).cuda()
                    else:
                        tar_pre_input = Variable(tar_input[i])
                    tar_input_final = tar_pre_input.reshape(1, a, b, c)
                    tar_feature = G(tar_input_final)
                    if tar_feature_de.size == 0:
                        tar_feature_de = tar_feature.detach().cpu().numpy()
                    else:
                        tar_feature_new_de = tar_feature.detach().cpu().numpy()
                        tar_feature_de = np.append(tar_feature_de, tar_feature_new_de, axis=0)
            count_tar = count_tar + 1
        for _ in range(len(dset_loaders_tar) - count_tar):
            tar_input, tar_labels = iter_tar.next()
            for i in range(len(tar_labels)):
                if tar_labels[i].item() == cls:
                    a, b, c = tar_input[i].shape
                    if use_gpu:
                        tar_pre_input = Variable(tar_input[i]).cuda()
                    else:
                        tar_pre_input = Variable(tar_input[i])
                    tar_input_final = tar_pre_input.reshape(1, a, b, c)
                    tar_feature_new = G(tar_input_final)
                    tar_feature_new_de = tar_feature_new.detach().cpu().numpy()
                    tar_feature_de = np.append(tar_feature_de, tar_feature_new_de, axis=0)
        print("Pass Target for Class: {}".format(cls + 1))
        print("Created feature: {}".format(tar_feature_de.shape))
        # prepare validation feature and errors

        count_val = 0
        val_feature_de = np.array([])
        iter_val = iter(dset_loaders_val)
        while val_feature_de.size == 0:
            val_input, val_labels = iter_val.next()
            for i in range(len(val_labels)):
                if val_labels[i].item() == cls:
                    a, b, c = val_input[i].shape
                    if use_gpu:
                        val_pre_input, val_labels_final = Variable(val_input[i]).cuda(), Variable(val_labels[i]).cuda()
                    else:
                        val_pre_input, val_labels_final = Variable(val_input[i]), Variable(val_labels[i])
                    val_input_final = val_pre_input.reshape(1, a, b, c)
                    val_feature = G(val_input_final)
                    pred_label = F1(val_feature)
                    w = pred_label.shape[1]
                    if val_feature_de.size == 0:
                        # feature and error
                        val_feature_de = val_feature.detach().cpu().numpy()
                        error = np.zeros(1)
                        error[0] = predict_loss(val_labels_final.item(), pred_label.reshape(1, w)).item()
                        error = error.reshape(1, 1)
                        print(error)
                    else:
                        # feature and error
                        val_feature_new_de = val_feature.detach().cpu().numpy()
                        val_feature_de = np.append(val_feature_de, val_feature_new_de, axis=0)
                        new_error = np.zeros(1)
                        new_error = new_error.reshape(1, 1)
                        new_error[0] = predict_loss(val_labels_final.item(), pred_label.reshape(1, w)).item()
                        error = np.append(error, new_error, axis=0)
            count_val = count_val + 1
        for _ in range(len(dset_loaders_val) - count_val):
            val_input, val_labels = iter_val.next()
            for i in range(len(val_labels)):
                if val_labels[i].item() == cls:
                    a, b, c = val_input[i].shape
                    if use_gpu:
                        val_pre_input, val_labels_final = Variable(val_input[i]).cuda(), Variable(val_labels[i]).cuda()
                    else:
                        val_pre_input, val_labels_final = Variable(val_input[i]), Variable(val_labels[i])
                    val_input_final = val_pre_input.reshape(1, a, b, c)
                    val_feature = G(val_input_final)
                    pred_label = F1(val_feature)
                    w = pred_label.shape[1]
                    val_feature_new_de = val_feature.detach().cpu().numpy()
                    val_feature_de = np.append(val_feature_de, val_feature_new_de, axis=0)
                    new_error = np.zeros(1)
                    new_error = new_error.reshape(1, 1)
                    new_error[0] = predict_loss(val_labels_final.item(), pred_label.reshape(1, w)).item()
                    error = np.append(error, new_error, axis=0)
        print("Pass Validation for Class: {}".format(cls + 1))
        print("Created error shape: {}".format(error.shape))
        print("Created feature: {}".format(val_feature_de.shape))
        # calculating the weight and the score for each class

        if not os.path.exists(args.save.split("/")[0] + "/feature_np/"):
            os.makedirs(args.save.split("/")[0] + "/feature_np/")

        np.save(args.save.split("/")[0]+ "/feature_np/" + str(cls) + "_" + "src_feature_de.npy", src_feature_de)
        np.save(args.save.split("/")[0]+ "/feature_np/" + str(cls) + "_" + "tar_feature_de.npy", tar_feature_de)
        np.save(args.save.split("/")[0]+ "/feature_np/" + str(cls) + "_" + "val_feature_de.npy", val_feature_de)
        src_feature_path = args.save.split("/")[0]+ "/feature_np/" + str(cls) + "_" + "src_feature_de.npy"
        tar_feature_path = args.save.split("/")[0]+ "/feature_np/" + str(cls) + "_" + "tar_feature_de.npy"
        val_feature_path = args.save.split("/")[0]+ "/feature_np/" + str(cls) + "_" + "val_feature_de.npy"
        weight = get_weight(src_feature_path, tar_feature_path, val_feature_path)
        cross_val_loss = cross_val_loss + get_dev_risk(weight, error) / class_num

    return cross_val_loss
