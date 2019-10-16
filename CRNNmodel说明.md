## CRNNmodel说明

python crnn_main_v2.py  进行训练

数据存放路径：data_chinese_tra/images_add_fake

​							data_chinese_tra/label_add_fake

python test_word_reco.py进行测试



data_generator是用做数据扩充文件夹，做３到１０个字的扩充数据，流程是用标注的文字label,连成一个串，随机截取３到１０个字符．

python generator_hori.py　根据给的背景background文件夹和字体font文件夹，扩充横条字（目前还没用）

python generator_veri.py　根据给的背景background文件夹和字体font文件夹，扩充竖条字（目前已使用），

但是目前是手动修改生成的字符个数，可以进一步修改成任意长度．

