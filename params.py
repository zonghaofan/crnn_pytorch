# import alphabets
from . import chinese_tra_alphabets

random_sample = True
best_accuracy = 50
keep_ratio = False
adam = False
adadelta = False
saveInterval = 2
valInterval = 400
n_test_disp = 10
displayInterval = 1
experiment = './expr'
# alphabet = alphabets.alphabet
alphabet = chinese_tra_alphabets.alphabet
# crnn = ''
# crnn = 'trained_models/mixed_second_finetune_acc97p7.pth'
crnn = 'trained_models/pretrained_model.pth'
beta1 = 0.5
lr = 0.0001
epochs = 300
nh = 256
imgW = 200#160
imgH = 32
val_batchSize = 16
batchSize = 32
workers = 2
std = 0.193
mean = 0.588
