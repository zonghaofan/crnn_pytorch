#coding:utf-8
"""
fzh created on 2019/10/15
crnn模型的推理过程

"""
import numpy as np
import sys, os
import time
import cv2
# sys.path.append(os.getcwd())

# crnn packages
import torch
from torch.autograd import Variable
from . import utils
from . import crnn
from . import chinese_tra_alphabets
from . import params
import torch.nn as nn

str1 = chinese_tra_alphabets.alphabet
sys.path.append(os.getcwd())
model_path = os.path.dirname(os.path.abspath(__file__))+'/crnn_best.pth'
def del_repeat(a):
    # a = [1, 0, 0, 2, 2, 2]
    opt = [1] * len(a)
    for i in range(len(a)):
        if a[i] == 0:
            opt[i] = 0
        elif a[i] == a[i - 1]:
            opt[i] = 0

    return np.nonzero(np.array(opt))
def crnn_recognition(cropped_image, model,alphabet):
    converter = utils.strLabelConverter(alphabet)
    image = (np.reshape(cropped_image, (params.imgH, params.imgW, 1))).transpose(2, 0, 1)
    # print('image.shape:', image.shape)
    image = image.astype(np.float32) / 255.
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image.sub_(params.mean).div_(params.std)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    # print('image.shape:',image.shape)

    model.eval()
    with torch.no_grad():
        preds = model(image)
    # print('preds.shape',preds.shape)
    # print(preds[:,:,2])
    # print('preds.max(2):',preds.max(2))
    value, preds = preds.max(2)
    # print('preds.shape:',preds.shape)
    # insert get probality
    probas = 10 ** value
    preds = preds.transpose(1, 0).contiguous().view(-1)
    # print('preds:',preds.data)
    mask = del_repeat(preds.data.view(-1).cpu().numpy().tolist())
    # print('probas[mask]:',probas[mask])
    final_probas = probas[mask].data.view(-1).cpu().numpy()

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    # print(preds_size)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    return sim_pred, final_probas

def word_reco_main(image,model):
    alphabet = str1
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print('image.shape:', image.shape)
    ori_h, ori_w, = image.shape
    if ori_h > ori_w:
        image = np.rot90(image)
    new_h, new_w = image.shape
    ratio = params.imgH / new_h
    resize_w = new_w * ratio
    image_resize = cv2.resize(image, (int(resize_w), params.imgH))
    # cv2.imwrite('hah.jpg', image_resize)
    if resize_w < params.imgW:  # 200
        length = (params.imgW - resize_w) / 2
        image = np.pad(np.array(image_resize), ((0, 0), (int(length), int(length))),
                       'constant', constant_values=(225, 225))
    image = cv2.resize(image, (params.imgW, params.imgH))
    # cv2.imwrite('hah.jpg', image)
    sim_pred = crnn_recognition(image, model,alphabet)
    # 释放GPU缓存
    torch.cuda.empty_cache()
    return sim_pred
def load_model():
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    alphabet = str1
    nclass = len(alphabet) + 1
    # print('nclass:', nclass)
    # crnn network
    model = crnn.CRNN(32, 1, nclass, 256)
    # print('torch.cuda.is_available():',torch.cuda.is_available())
    if torch.cuda.is_available():
        model = model.cuda()
        # model = nn.DataParallel(model)
    # print('============model info============')
    print('loading pretrained model from {0}'.format(model_path))
    # 导入已经训练好的crnn模型
    model.load_state_dict(torch.load(model_path))
    return model
if __name__ == '__main__':
    image_path = './test_images/test_jietu3.png'
    ## read an image
    image = cv2.imread(image_path)
    started = time.time()
    model = load_model()
    rec_res = word_reco_main(image, model)
    print('results: {}'.format(rec_res))
    finished = time.time()
    print('elapsed time: {0}'.format(finished - started))



    
