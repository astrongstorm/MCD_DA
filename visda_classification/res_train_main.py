from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from utils import *
from taskcv_loader import CVDataLoader
from basenet import *
import torch.nn.functional as F
import os
import source_risk
import dev
import dev_icml
import seperate_data as sep
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
# Training settings

parser = argparse.ArgumentParser(description='Visda Classification')
parser.add_argument('--batch-size', type=int, default=13, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP',
                    help='the name of optimizer')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_k', type=int, default=4, metavar='K',
                    help='how many steps to repeat the generator update')
parser.add_argument('--num-layer', type=int, default=2, metavar='K',
                    help='how many layers for classifier')
parser.add_argument('--name', type=str, default='board', metavar='B',
                    help='board dir')
parser.add_argument('--save', type=str, default='save/mcd', metavar='B',
                    help='board dir')
parser.add_argument('--train_path', type=str, default='', metavar='B',
                    help='directory of source datasets')
parser.add_argument('--val_path', type=str, default='', metavar='B',
                    help='directory of target datasets')
parser.add_argument('--resnet', type=str, default='101', metavar='B',
                    help='which resnet 18,50,101,152,200')
parser.add_argument('--load_network_path', type=str, default='',
                    help='whether load saved network')
parser.add_argument('--validation_method', type=str, choices=['Source_Risk', 'Dev', 'Dev_icml'])

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
train_path = args.train_path
val_path = args.val_path
num_k = args.num_k
num_layer = args.num_layer
batch_size = args.batch_size
save_path = args.save+'_'+str(args.num_k)

data_transforms = {
    train_path: transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    val_path: transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
dsets = {x: datasets.ImageFolder(os.path.join(x), data_transforms[x]) for x in [train_path,val_path]}
dset_sizes = {x: len(dsets[x]) for x in [train_path, val_path]}
dset_classes = dsets[train_path].classes

class_num = len(dset_classes)
print("Class number: {}".format(class_num))

print ('classes'+str(dset_classes))
use_gpu = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
train_loader = CVDataLoader()
train_loader.initialize(dsets[train_path],dsets[val_path],batch_size)
dataset = train_loader.load_data()
test_loader = CVDataLoader()
opt= args
test_loader.initialize(dsets[train_path],dsets[val_path],batch_size,shuffle=True)
dataset_test = test_loader.load_data()
option = 'resnet'+args.resnet
G = ResBase(option)
F1 = ResClassifier(num_layer=num_layer)
F2 = ResClassifier(num_layer=num_layer)
F1.apply(weights_init)
F2.apply(weights_init)
lr = args.lr

if args.load_network_path == '':
    test_load = False
else:
    test_load = True

if args.cuda:
    G.cuda()
    F1.cuda()
    F2.cuda()
if args.optimizer == 'momentum':
    optimizer_g = optim.SGD(list(G.features.parameters()), lr=args.lr,weight_decay=0.0005)
    optimizer_f = optim.SGD(list(F1.parameters())+list(F2.parameters()),momentum=0.9,lr=args.lr,weight_decay=0.0005)
elif args.optimizer == 'adam':
    optimizer_g = optim.Adam(G.features.parameters(), lr=args.lr,weight_decay=0.0005)
    optimizer_f = optim.Adam(list(F1.parameters())+list(F2.parameters()), lr=args.lr,weight_decay=0.0005)
else:
    optimizer_g = optim.Adadelta(G.features.parameters(),lr=args.lr,weight_decay=0.0005)
    optimizer_f = optim.Adadelta(list(F1.parameters())+list(F2.parameters()),lr=args.lr,weight_decay=0.0005)    
    
def train(num_epoch, option, num_layer, test_load, cuda):
    criterion = nn.CrossEntropyLoss().cuda()
    if not test_load:
        for ep in range(num_epoch):
            G.train()
            F1.train()
            F2.train()
            for batch_idx, data in enumerate(dataset):
                if batch_idx * batch_size > 30000:
                    break
                if args.cuda:
                    data1 = data['S']
                    target1 = data['S_label']
                    data2 = data['T']
                    target2 = data['T_label']
                    data1, target1 = data1.cuda(), target1.cuda()
                    data2, target2 = data2.cuda(), target2.cuda()
                # when pretraining network source only
                eta = 1.0
                data = Variable(torch.cat((data1, data2), 0))
                target1 = Variable(target1)
                # Step A train all networks to minimize loss on source
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()
                output = G(data)
                output1 = F1(output)
                output2 = F2(output)

                output_s1 = output1[:batch_size, :]
                output_s2 = output2[:batch_size, :]
                output_t1 = output1[batch_size:, :]
                output_t2 = output2[batch_size:, :]
                output_t1 = F.softmax(output_t1)
                output_t2 = F.softmax(output_t2)

                entropy_loss = - torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
                entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))

                loss1 = criterion(output_s1, target1)
                loss2 = criterion(output_s2, target1)
                all_loss = loss1 + loss2 + 0.01 * entropy_loss
                all_loss.backward()
                optimizer_g.step()
                optimizer_f.step()

                # Step B train classifier to maximize discrepancy
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()

                output = G(data)
                output1 = F1(output)
                output2 = F2(output)
                output_s1 = output1[:batch_size, :]
                output_s2 = output2[:batch_size, :]
                output_t1 = output1[batch_size:, :]
                output_t2 = output2[batch_size:, :]
                output_t1 = F.softmax(output_t1)
                output_t2 = F.softmax(output_t2)
                loss1 = criterion(output_s1, target1)
                loss2 = criterion(output_s2, target1)
                entropy_loss = - torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
                entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))
                loss_dis = torch.mean(torch.abs(output_t1 - output_t2))
                F_loss = loss1 + loss2 - eta * loss_dis + 0.01 * entropy_loss
                F_loss.backward()
                optimizer_f.step()
                # Step C train genrator to minimize discrepancy
                for i in range(num_k):
                    optimizer_g.zero_grad()
                    output = G(data)
                    output1 = F1(output)
                    output2 = F2(output)

                    output_s1 = output1[:batch_size, :]
                    output_s2 = output2[:batch_size, :]
                    output_t1 = output1[batch_size:, :]
                    output_t2 = output2[batch_size:, :]

                    loss1 = criterion(output_s1, target1)
                    loss2 = criterion(output_s2, target1)
                    output_t1 = F.softmax(output_t1)
                    output_t2 = F.softmax(output_t2)
                    loss_dis = torch.mean(torch.abs(output_t1 - output_t2))
                    entropy_loss = -torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
                    entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))

                    loss_dis.backward()
                    optimizer_g.step()
                if batch_idx % args.log_interval == 0:
                    print(
                        'Train Ep: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\t Dis: {:.6f} Entropy: {:.6f}'.format(
                            ep, batch_idx * len(data), 70000,
                                100. * batch_idx / 70000, loss1.data[0], loss2.data[0], loss_dis.data[0],
                            entropy_loss.data[0]))
                if batch_idx == 1 and ep > 1:
                    test(ep)
                    G.train()
                    F1.train()
                    F2.train()
    else:
        G_load = ResBase(option)
        F1_load = ResClassifier(num_layer=num_layer)
        F2_load = ResClassifier(num_layer=num_layer)

        F1_load.apply(weights_init)
        F2_load.apply(weights_init)
        G_path = args.load_network_path + 'G.pth'
        F1_path = args.load_network_path + 'F1.pth'
        F2_path = args.load_network_path + 'F2.pth'
        G_load.load_state_dict(torch.load(G_path))
        F1_load.load_state_dict(torch.load(F1_path))
        F2_load.load_state_dict(torch.load(F2_path))
        #
        # G_load = torch.load('whole_model_G.pth')
        # F1_load = torch.load('whole_model_F1.pth')
        # F2_load = torch.load('whole_model_F2.pth')
        if cuda:
            G_load.cuda()
            F1_load.cuda()
            F2_load.cuda()
        G_load.eval()
        F1_load.eval()
        F2_load.eval()
        test_loss = 0
        correct = 0
        correct2 = 0
        size = 0

        val = False
        for batch_idx, data in enumerate(dataset_test):
            if batch_idx * batch_size > 5000:
                break
            if args.cuda:
                data2 = data['T']
                target2 = data['T_label']
                if val:
                    data2 = data['S']
                    target2 = data['S_label']
                data2, target2 = data2.cuda(), target2.cuda()
            data1, target1 = Variable(data2, volatile=True), Variable(target2)
            output = G_load(data1)
            output1 = F1_load(output)
            output2 = F2_load(output)
            # print("Feature: {}\n Predict_value: {}".format(output, output2))
            test_loss += F.nll_loss(output1, target1).item()
            pred = output1.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target1.data).cpu().sum()
            pred = output2.data.max(1)[1]  # get the index of the max log-probability
            k = target1.data.size()[0]
            correct2 += pred.eq(target1.data).cpu().sum()

            size += k
        test_loss = test_loss
        test_loss /= len(test_loader)  # loss function already averages over batch size
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) ({:.0f}%)\n'.format(
            test_loss, correct, size,
            100. * correct / size, 100. * correct2 / size))
        value = max(100. * correct / size, 100. * correct2 / size)
        print("Value: {}".format(value))
        if args.cuda:
            use_gpu = True
        else:
            use_gpu = False
        # configure network path

        if (100. * correct /size) > (100. *correct2 /size):
            predict_network_path = F1_path
        else:
            predict_network_path = F2_path

        feature_network_path = G_path

        # configure the datapath for target and test
        source_path = 'chn_training_list.txt'
        target_path = 'chn_validation_list.txt'
        cls_source_list, cls_validation_list = sep.split_set(source_path, class_num)
        source_list = sep.dimension_rd(cls_source_list)

        if args.validation_method == 'Source_Risk':
            cv_loss = source_risk.cross_validation_loss(args, feature_network_path, predict_network_path, num_layer, cls_source_list,
                                                target_path, cls_validation_list,
                                                class_num, 256,
                                                224, batch_size,
                                                use_gpu)
        elif args.validation_method == 'Dev_icml':
            cv_loss = dev_icml.cross_validation_loss(args, feature_network_path, predict_network_path, num_layer, source_list,
                                                target_path, cls_validation_list,
                                                class_num, 256,
                                                224, batch_size,
                                                use_gpu)
        else:
            cv_loss = dev.cross_validation_loss(args, feature_network_path, predict_network_path, num_layer,
                                                     source_list,
                                                     target_path, cls_validation_list,
                                                     class_num, 256,
                                                     224, batch_size,
                                                     use_gpu)
            # target_list_no_label = open(target_path).readlines()
            # tar_cls_list = []
            # cross_val_loss = 0
            # save_path = args.save
            # # load network
            # print("Loaded network")
            # # add pesudolabel for target data
            # print("Sperating target data")
            # target_list = []
            # dsets_tar = ImageList(target_list_no_label,
            #                       transform=prep.image_train(resize_size=256, crop_size=224))
            # dset_loaders_tar = util_data.DataLoader(dsets_tar, batch_size=batch_size, shuffle=False, num_workers=4)
            # len_train_target = len(dset_loaders_tar)
            # iter_target = iter(dset_loaders_tar)
            # count = 0
            # for i in range(len_train_target):
            #     input_tar, label_tar = iter_target.next()
            #     if use_gpu:
            #         input_tar, label_tar = Variable(input_tar).cuda(), Variable(label_tar).cuda()
            #     else:
            #         input_tar, label_tar = Variable(input_tar), Variable(label_tar)
            #     tar_feature = G(input_tar)
            #     predict_score = F1(tar_feature)
            #     _, pre_lab = torch.max(predict_score, 1)
            #     predict_label = pre_lab.detach()
            #     for num in range(len(predict_label.cpu())):
            #         if target_list_no_label[count][-3] == ' ':
            #             ind = -2
            #         else:
            #             ind = -3
            #         target_list.append(target_list_no_label[count][:ind])
            #         target_list[count] = target_list[count] + str(predict_label[num].cpu().numpy()) + "\n"
            #         count += 1
            # # seperate the class
            # for i in range(class_num):
            #     tar_cls_list.append([j for j in target_list if int(j.split(" ")[1].replace("\n", "")) == i])
            # prep_dict = prep.image_train(resize_size=256, crop_size=224)
            # # load different class's image
            # for cls in range(class_num):
            #     print(cls)
            #     print("Loading source data")
            #     # prepare source feature
            #     dsets_src = ImageList(cls_source_list[cls], transform=prep_dict)
            #     dset_loaders_src = util_data.DataLoader(dsets_src, batch_size=batch_size, shuffle=True, num_workers=4)
            #
            #     iter_src = iter(dset_loaders_src)
            #     src_input = iter_src.next()[0]
            #     if use_gpu:
            #         src_input = Variable(src_input).cuda()
            #     else:
            #         src_input = Variable(src_input)
            #
            #     src_feature = G(src_input)
            #     src_feature_de = src_feature.detach().cpu().numpy()
            #     for count_src in range(len(dset_loaders_src) - 1):
            #         src_input = iter_src.next()[0]
            #         if use_gpu:
            #             src_input = Variable(src_input).cuda()
            #         else:
            #             src_input = Variable(src_input)
            #
            #         src_feature_new = G(src_input)
            #         src_feature_new_de = src_feature_new.detach().cpu().numpy()
            #         src_feature_de = np.append(src_feature_de, src_feature_new_de, axis=0)
            #
            #     np.save(os.path.join(save_path, 'src_feature_de.npy'), src_feature_de)
            #     print("Loading target data")
            #     # prepare target feature
            #     dsets_tar = ImageList(tar_cls_list[cls], transform=prep_dict)
            #     dset_loaders_tar = util_data.DataLoader(dsets_tar, batch_size=batch_size, shuffle=True, num_workers=4)
            #
            #     iter_tar = iter(dset_loaders_tar)
            #     tar_input = iter_tar.next()[0]
            #     if use_gpu:
            #         tar_input = Variable(tar_input).cuda()
            #     else:
            #         tar_input = Variable(tar_input)
            #
            #     tar_feature = G(tar_input)
            #     tar_feature_de = tar_feature.detach().cpu().numpy()
            #     for count_tar in range(len(dset_loaders_tar) - 1):
            #         tar_input = iter_tar.next()[0]
            #         if use_gpu:
            #             tar_input = Variable(tar_input).cuda()
            #         else:
            #             tar_input = Variable(tar_input)
            #
            #         tar_feature_new = G(tar_input)
            #         tar_feature_new_de = tar_feature_new.detach().cpu().numpy()
            #         tar_feature_de = np.append(tar_feature_de, tar_feature_new_de, axis=0)
            #
            #     np.save(os.path.join(save_path, 'tar_feature_de.npy'), tar_feature_de)
            #     print("Loading validation data")
            #     # prepare validation feature and predicted label for validation
            #     dsets_val = ImageList(cls_validation_list[cls], transform=prep_dict)
            #     dset_loaders_val = util_data.DataLoader(dsets_val, batch_size=batch_size, shuffle=True, num_workers=4)
            #
            #     iter_val = iter(dset_loaders_val)
            #     val_input, val_labels = iter_val.next()
            #     if use_gpu:
            #         val_input, val_labels = Variable(val_input).cuda(), Variable(val_labels).cuda()
            #     else:
            #         val_input, val_labels = Variable(val_input), Variable(val_labels)
            #
            #     val_feature = G(val_input)
            #     pred_label = F1(val_feature)
            #     val_feature_de = val_feature.detach().cpu().numpy()
            #
            #     w = pred_label[0].shape[0]
            #     error = np.zeros(1)
            #
            #     error[0] = dev.predict_loss(cls, pred_label[0].reshape(1, w)).item()
            #     error = error.reshape(1, 1)
            #     for num_image in range(1, len(pred_label)):
            #         new_error = np.zeros(1)
            #         single_pred_label = pred_label[num_image]
            #         w = single_pred_label.shape[0]
            #
            #         new_error[0] = dev.predict_loss(cls, single_pred_label.reshape(1, w)).item()
            #         new_error = new_error.reshape(1, 1)
            #         error = np.append(error, new_error, axis=0)
            #     for count_val in range(len(dset_loaders_val) - 1):
            #         val_input, val_labels = iter_val.next()
            #         if use_gpu:
            #             val_input, val_labels = Variable(val_input).cuda(), Variable(val_labels).cuda()
            #         else:
            #             val_input, val_labels = Variable(val_input), Variable(val_labels)
            #
            #         val_feature_new = G(val_input)
            #         val_feature_new_de = val_feature_new.detach().cpu().numpy()
            #         val_feature_de = np.append(val_feature_de, val_feature_new_de, axis=0)
            #         val_feature = G(val_input)
            #         pred_label = F1(val_feature)
            #
            #         for num_image in range(len(pred_label)):
            #             new_error = np.zeros(1)
            #             single_pred_label = pred_label[num_image]
            #             w = single_pred_label.shape[0]
            #             new_error[0] = dev.predict_loss(cls, single_pred_label.reshape(1, w)).item()
            #             new_error = new_error.reshape(1, 1)
            #             # cls should be a value, new_labels should be a [[x]] tensor format, the input format required by predict_loss
            #             error = np.append(error, new_error, axis=0)
            #         # error should be a (N, 1) numpy array, the input format required by get_dev_risk
            #     np.save(os.path.join(save_path, 'val_feature_de.npy'), val_feature_de)
            #     print("Finish printing")
            #     # print(src_feature_de.shape)
            #     # print(tar_feature_de.shape)
            #     # print(val_feature_de.shape)
            #
            #     # print(cls)
            #     # weight = dev.get_weight(os.path.join(save_path, 'src_feature_de.npy'),
            #     #                     os.path.join(save_path, 'tar_feature_de.npy'),
            #     #                     os.path.join(save_path, 'val_feature_de.npy'))
            #     # cv_loss = cross_val_loss + dev.get_dev_risk(weight, error) / class_num
        print(cv_loss)

def test(epoch):
    G.eval()
    F1.eval()
    F2.eval()
    test_loss = 0
    correct = 0
    correct2 = 0
    size = 0

    val = False
    for batch_idx, data in enumerate(dataset_test):
        if batch_idx*batch_size > 5000:
            break
        if args.cuda:
            data2 = data['T']
            target2 = data['T_label']
            if val:
                data2 = data['S']
                target2 = data['S_label']
            data2, target2 = data2.cuda(), target2.cuda()
        data1, target1 = Variable(data2, volatile=True), Variable(target2)
        output = G(data1)
        output1 = F1(output)
        output2 = F2(output)
        test_loss += F.nll_loss(output1, target1).data[0]
        pred = output1.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target1.data).cpu().sum()
        pred = output2.data.max(1)[1] # get the index of the max log-probability
        k = target1.data.size()[0]
        correct2 += pred.eq(target1.data).cpu().sum()

        size += k
    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) ({:.0f}%)\n'.format(
        test_loss, correct, size,
        100. * correct / size,100.*correct2/size))
    # if 100. * correct / size > 67 or 100. * correct2 / size > 67:
    value = max(100. * correct / size,100. * correct2 / size)
    if not val and value > 60:
        # torch.save(F1.state_dict(), save_path+'_'+args.resnet+'_'+str(value)+'_'+'F1.pth')
        # torch.save(F2.state_dict(), save_path+'_'+args.resnet+'_'+str(value)+'_'+'F2.pth')
        # torch.save(G.state_dict(), save_path+'_'+args.resnet+'_'+str(value)+'_'+'G.pth')
        print(save_path)
        torch.save(F1.state_dict(),  save_path + '_' + str(epoch) + 'F1.pth')
        torch.save(F2.state_dict(), save_path + '_' + str(epoch) + 'F2.pth')
        torch.save(G.state_dict(), save_path + '_' + str(epoch) + 'G.pth')
        torch.save(F1, 'whole_model_F1.pth')
        torch.save(F2, 'whole_model_F2.pth')
        torch.save(G, 'whole_model_G.pth')


#for epoch in range(1, args.epochs + 1):
train(args.epochs+1, option, num_layer, test_load, args.cuda)
