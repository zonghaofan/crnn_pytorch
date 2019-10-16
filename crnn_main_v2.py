#coding:utf-8
"""
fzh created on 2019/10/15
crnn模型训练程序
"""
from __future__ import print_function
from torch.utils.data import DataLoader
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
# from warpctc_pytorch import CTCLoss
import os
import utils
# import dataset
import crnn as crnn
import re
import params
from dataset_v2 import baiduDataset

# def init_args():
#     args = argparse.ArgumentParser()
#     args.add_argument('--trainroot', help='path to dataset', default='./to_lmdb/train')
#     args.add_argument('--valroot', help='path to dataset', default='./to_lmdb/train')
#     args.add_argument('--cuda', action='store_true', help='enables cuda', default=False)

#     return args.parse_args()

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def val(net, val_loader, criterion, epoch, max_i=1000):
    print('================Start val=================')
    for p in crnn.parameters():
        p.requires_grad = False
    net.eval()
    i = 0
    n_correct = 0
    n_all = 0
    loss_avg = utils.averager()

    for i_batch, (image, index) in enumerate(val_loader):
        image = image.to(device)
        print('image.shape:',image.shape)
        label = utils.get_batch_label(val_dataset, index)
        # [41,batch,nclass]
        preds = crnn(image)
        batch_size = image.size(0)
        # index = np.array(index.data.numpy())
        label_text, label_length = converter.encode(label)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, label_text, preds_size, label_length) / batch_size
        loss_avg.add(cost)
        # [41,batch]
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        # preds = preds.transpose(1, 0).reshape(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        print('label:',label[:2])
        print('sim_preds:',sim_preds[:2])
        # print(list(zip(sim_preds, label)))

        n_all += len(label)
        for pred, target in list(zip(sim_preds, label)):
            if pred == target:
                n_correct += 1

        if (i_batch+1)%params.displayInterval == 0:
            print('[%d/%d][%d/%d]' %
                      (epoch, params.epochs, i_batch, len(val_loader)))
        if i_batch == max_i:
            break
    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, label):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
    #
    # print('n_correct:',n_correct)
    # accuracy = n_correct / float(max_i * params.val_batchSize)
    accuracy = n_correct / n_all
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
    return accuracy

def train(crnn, train_loader, criterion, epoch):
    for p in crnn.parameters():
        p.requires_grad = True
    crnn.train()
    #loss averager
    loss_avg = utils.averager()
    for i_batch, (image, index) in enumerate(train_loader):
        #[b,c,h,w] [32,1,32,160]
        image = image.to(device)
        print('image.shape:',image.shape)
        batch_size = image.size(0)
        #['xxx','xxxx',...batch]
        label = utils.get_batch_label(dataset, index)
        #[41,batch,nclass]
        preds = crnn(image)
        # print('preds.shape',preds.shape)
        # index = np.array(index.data.numpy())
        #[, , ,]    [len(lable[0]),len(lable[1]),...]
        label_text, label_length = converter.encode(label)
        # print('label_text:', len(label_text))
        # print('label_length:', label_length)
        #[41,41,41,...]*batch
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        # print('preds.shape, label_text.shape, preds_size.shape, label_length.shape',preds.shape, label_text.shape, preds_size.shape, label_length.shape)
        # torch.Size([41, 32, 6736]) torch.Size([320]) torch.Size([320]) torch.Size([320])
        cost = criterion(preds, label_text, preds_size, label_length) / batch_size
        # print('cost:',cost)
        crnn.zero_grad()
        cost.backward()
        optimizer.step()

        loss_avg.add(cost)

        if (i_batch+1) % params.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, params.epochs, i_batch, len(train_loader), loss_avg.val()))
            loss_avg.reset()

def main(crnn, train_loader, val_loader, criterion, optimizer):

    crnn = crnn.to(device)
    criterion = criterion.to(device)
    for i,epoch in enumerate(range(params.epochs)):
        # if i<1:
            train(crnn, train_loader, criterion, epoch)
            # # ## max_i: cut down the consuming time of testing, if you'd like to validate on the whole testset, please set it to len(val_loader)
            accuracy = val(crnn, val_loader, criterion, epoch, max_i=1000)
            for p in crnn.parameters():
                p.requires_grad = True
            # if accuracy > params.best_accuracy:
            torch.save(crnn.state_dict(), '{0}/crnn_Rec_done_{1}_{2}.pth'.format(params.experiment, epoch, accuracy))
            torch.save(crnn.state_dict(), '{0}/crnn_best.pth'.format(params.experiment))
            print("is best accuracy: {0}".format(accuracy > params.best_accuracy))


def backward_hook(self, grad_input, grad_output):
    for g in grad_input:
        g[g != g] = 0   # replace all nan/inf in gradients to zero

if __name__ == '__main__':
    # args = init_args()
    # manualSeed = random.randint(1, 10000)  #fix seed
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    manualSeed = 10
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # store model path
    if not os.path.exists('./expr'):
        os.mkdir('./expr')
    # read train set
    dataset = baiduDataset("./data_chinese_tra/images_add_fake", "./data_chinese_tra/label_add_fake/train_add_fake.txt",
                           params.alphabet, False, (params.imgW, params.imgH))
    val_dataset = baiduDataset("./data_chinese_tra/images_add_fake", "./data_chinese_tra/label_add_fake/val_add_fake.txt",
                               params.alphabet, False, (params.imgW, params.imgH))

    train_loader = DataLoader(dataset, batch_size=params.batchSize, shuffle=True, num_workers=params.workers)
    # shuffle=True, just for time consuming.
    val_loader = DataLoader(val_dataset, batch_size=params.val_batchSize, shuffle=True, num_workers=params.workers)
    converter = utils.strLabelConverter(dataset.alphabet)
    nclass = len(params.alphabet) + 1
    print('nclass:',nclass)

    nc = 1
    criterion = torch.nn.CTCLoss(reduction='sum')
    # criterion = CTCLoss()

    # cnn and rnn
    crnn = crnn.CRNN(32, nc, nclass, params.nh)

    crnn.apply(weights_init)
    if params.crnn != '':
        print('loading pretrained model from %s' % params.crnn)
        crnn.load_state_dict(torch.load(params.crnn))

    # setup optimizer
    if params.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=params.lr,
                               betas=(params.beta1, 0.999))
    elif params.adadelta:
        optimizer = optim.Adadelta(crnn.parameters(), lr=params.lr)
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=params.lr)

    crnn.register_backward_hook(backward_hook)
    main(crnn, train_loader, val_loader, criterion, optimizer)
