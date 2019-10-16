#coding:utf-8
"""
fzh created on 2019/10/15
修改最后一层节点个数，在做fine tune
"""
import torch
from torch import nn

def change_shape_of_coco_wt_lstm():
    load_from = './trained_models/mixed_second_finetune_acc97p7.pth'
    save_to = './trained_models/pretrained_model.pth'

    model = torch.load(load_from)
    for key,values in model.items():
        print(key)
    print(model['rnn.1.embedding.weight'].shape)
    print(model['rnn.1.embedding.bias'].shape)

    # # print(wt['module.rnn.1.embedding.bias'].shape)
    #
    model['rnn.1.embedding.weight'] = nn.init.kaiming_normal_(torch.empty(5146, 512),
                                               mode='fan_in', nonlinearity='relu')
    model['rnn.1.embedding.bias'] = torch.rand(5146)

    torch.save(model, save_to)
if __name__ == '__main__':
    change_shape_of_coco_wt_lstm()