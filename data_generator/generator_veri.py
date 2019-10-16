from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import glob
import numpy as np
from to_dictionary import to_dictionary
import os
import cv2
from chinese_tra_alphabets import alphabet
from nespaper_semantics import seg_str
#dict_1 = to_dictionary('../char_std_5990.txt', 'gbk')
#dict_2 = to_dictionary('../text_info_results.txt', 'utf-8')
#dict_3 = to_dictionary('info.txt', 'utf-8')


#print(len(info_str))
# print(dict_1)
# print(dict_2)
# print(dict_3)

'''
1. 从文字库随机选择10个字符
2. 生成图片
3. 随机使用函数
'''

# 从字库中随机选择n个字符
def sto_choice_from_info_str(info_str,quantity):
    start = random.randint(0, len(info_str)-(quantity+1))
    end = start + quantity
    random_word = info_str[start:end]

    return random_word

def random_word_color():
    # font_color_choice = [[54,54,54],[54,54,54],[105,105,105]]
    font_color_choice = [[10, 10, 10], [5, 5, 5], [0, 0, 0]]
    font_color = random.choice(font_color_choice)
    print('font_color：', font_color)
    noise = np.array([random.randint(0,2),random.randint(0,2),random.randint(0,2)])
    font_color = (np.array(font_color) + noise).tolist()

    return tuple(font_color)

# 生成一张图片
def create_an_image(bground_path, width, height):
    bground_list = os.listdir(bground_path)
    bground_choice = random.choice(bground_list)
    bground = Image.open(bground_path+bground_choice)
    #print('background:',bground_choice)
    # print('bground.size[0],bground.size[1]:',bground.size[0],bground.size[1])
    x, y = random.randint(0,bground.size[0]-width), random.randint(0, bground.size[1]-height)
    # print('x,y',x,y)
    bground = bground.crop((x, y, x+width, y+height))
    # print('bground.size',bground.size)
    return bground

# 选取作用函数
def random_choice_in_process_func():
    pass

# 模糊函数
def darken_func(image):
    #.SMOOTH
    #.SMOOTH_MORE
    #.GaussianBlur(radius=2 or 1)
    # .MedianFilter(size=3)
    # 随机选取模糊参数
    filter_ = random.choice(
                            [ImageFilter.SMOOTH,
                            ImageFilter.SMOOTH_MORE,
                            ImageFilter.GaussianBlur(radius=1.3)]
                            )
    image = image.filter(filter_)
    #image = img.resize((290,32))

    return image

# 旋转函数
def rotate_func():
    pass

# 噪声函数
def random_noise_func():
    pass

# 字体拉伸函数
def stretching_func():
    pass

# 随机选取文字贴合起始的坐标, 根据背景的尺寸和字体的大小选择
def random_x_y(bground_size, font_size):
    width, height = bground_size
    print('bground_size:',bground_size)
    print('font_size:',font_size)
    # 为防止文字溢出图片，x，y要预留宽高
    # x = random.randint(0, width - font_size * 10)
    # y = random.randint(0, int((height-font_size)/4))
    """====notice notice===="""
    #10个字要减140 9个字要减100 8个字要减80 7个字要减40 6个字要减20 5个字以下不用减
    x = random.randint(3, int((width - font_size) / 2))
    y = random.randint(10, height - font_size * 7)

    return x, y

def random_font_size():
    font_size = random.randint(22,25)

    return font_size

def random_font(font_path):
    font_list = os.listdir(font_path)
    random_font = random.choice(font_list)

    return font_path + random_font
def add_white_noise(image):
    # print(image.shape)
    rows, cols, dims = image.shape

    random_index=random.randint(10,100)
    for i in range(random_index):
        x = np.random.randint(2, rows)
        y = np.random.randint(2, cols)

        if random.getrandbits(1):
            image[x-1:x+1, y-1:y+1, :] = 180
        else:
            image[x, y, :] = 180
    return image

def main(infostr,save_path, num,words_lenght,width,height):
    # 随机选取5个字符
    random_word_ori = sto_choice_from_info_str(infostr,words_lenght)
    # print('random_word:',random_word)
    random_word=''.join([i+'\n' for i in random_word_ori if i not in [':','(',')','｢','｣']])
    # print('random_word:', random_word)
    # 生成一张背景图片，已经剪裁好，宽高为280*32
    raw_image = create_an_image('./background/',width,height)

    # 随机选取字体大小
    font_size = random_font_size()
    # 随机选取字体
    font_name = random_font('./font/')
    # 随机选取字体颜色
    font_color = random_word_color()
    #
    # 随机选取文字贴合的坐标 x,y
    draw_x, draw_y = random_x_y(raw_image.size, font_size)

    # 将文本贴到背景图片
    font = ImageFont.truetype(font_name, font_size)
    # print('font:',font)
    draw = ImageDraw.Draw(raw_image)
    draw.text((draw_x, draw_y), random_word, fill=font_color, font=font)
    raw_image = raw_image.rotate(0.3)
    image = add_white_noise(np.array(raw_image)[...,::-1])
    # 随机选取作用函数和数量作用于图片
    # random_choice_in_process_func()
    # raw_image = darken_func(raw_image)

    # 保存文本信息和对应图片名称
    # with open(save_path[:-1]+'.txt', 'a+', encoding='utf-8') as file:
    # file.write('10val/' + str(num)+ '.png ' + random_word + '\n')
    # raw_image.save(save_path+str(num)+'.png')
    img_name=save_path+'/'+'fake_'+str(words_lenght)+'_'+str(num)+'.jpg'
    cv2.imwrite(img_name,image)
    with open(img_name.replace('.jpg','.txt'), 'w', encoding='utf-8') as file:
        [file.write(val) for val in random_word_ori]
def make_fake_data():
    info_str = seg_str
    print(len(info_str))
    words_lenght = 3
    output_path = 'data_set/fake_one_'+str(words_lenght)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # 图片标签
    # file  = open('data_set/val_set.txt', 'w', encoding='utf-8')
    # print(file)
    if words_lenght<6:
        width=32
        height=200
    elif 6<=words_lenght<7:
        width = 32
        height = 220

    elif 7<=words_lenght<8:
        width = 32
        height = 240
    elif 8 <= words_lenght < 9:
        width = 32
        height = 280
    elif 9 <= words_lenght < 10:
        width = 32
        height = 300
    else:
        width = 32
        height = 340
    total = 1000
    for num in range(0, total):
        main(info_str,output_path, num,words_lenght,width,height)
        if num % 1000 == 0:
            print('[%d/%d]' % (num, total))
#根据标注的报纸制作假的语义库
def make_fake_word_library():
    import pandas as pd
    train_path = './label/train.txt'
    val_path = './label/val.txt'

    train_names = np.array(pd.read_csv(train_path, header=None))
    val_names = np.array(pd.read_csv(val_path, header=None))

    Semantics_list=[]
    for i,train_name in enumerate(train_names):
            words = train_name[0].split(' ')[-1]
            Semantics_list.append(words)

    for i,val_name in enumerate(val_names):
            words = val_name[0].split(' ')[-1]
            Semantics_list.append(words)
    print(len(Semantics_list))
    print(Semantics_list[:2])
    Semantics_str=''.join(Semantics_list)
    print(len(Semantics_str))
    print(Semantics_str)
if __name__ == '__main__':
    #用alphat制作的假数据
    make_fake_data()
    # make_fake_word_library()


