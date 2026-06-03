#encoding=gbk
import os.path
from tqdm import  tqdm
import onnx
import onnxruntime as ort
import numpy as np
from imutils import paths
import cv2
from tools.ocr_dicts import en_dict
#路径配置
onnx_path=r'../inference/rec_onnx/best-smi.onnx'
pic_path=r'D:\E\github_zl\OCRDataSetV1\rec\test'
# pic_path=r'./test'
# pic_path=r'./imgs'
w,h=384,48

#onnx模型载入
model = onnx.load(onnx_path)
onnx.checker.check_model(model)
session = ort.InferenceSession(onnx_path,providers=['CPUExecutionProvider'])

imgpaths=list(paths.list_images(pic_path))

total_num=len(imgpaths)
right_num=0
error_strs=[]
error_nums=0
for imgpath in tqdm(imgpaths):
    img=cv2.imread(imgpath)
    filename=imgpath.split(os.path.sep)[-1]
    img=cv2.resize(img,(w,h))
    img=np.array([np.transpose(img,(2,0,1))], dtype=np.uint8)  # HWC→CHW，uint8                         #除方差
    #模型推理
    out = session.run(None,input_feed = { 'input' : img })

    preds_idx=np.argmax(out[0],axis=2)[0]# 第2个0，是因为我不是多批次输入，只有一张的原因
    preds_prob=np.max(out[0],axis=2)[0]


    top_data=preds_idx[1:]   #第二个元素到最后一个元素
    bottom_data=preds_idx[:-1]  #第一个元素到最后第二个元素

    filter_data=[preds_idx[0]]  #第一个元素不受影响
    indexs=[0]  #存放filter_data的索引值

    #去除重复元素
    index=0
    for topdata,bottomdata in zip(top_data,bottom_data):
        index+=1
        if(topdata!=bottomdata):   #不相同则保留
            filter_data.append(topdata) #记录对比不相同的topdata
            indexs.append(index)        #对应索引

    char_list=[]
    probs=[]

    #去除id为0的元素，id为0是空格
    for id, index in zip(filter_data,indexs):
        if(id !=0):
            char_list.append(en_dict[id])
            probs.append(preds_prob[index])
    text = ''.join(char_list)

    str_result='({})-({})-({})'.format(filename,text,np.mean(probs))
    gt=filename.split('_')[0]

    if ('@@@' in gt):  # 针对':'进行特殊处理
        gt = gt.replace('@@@', ':')
        if ('---' in gt):
            gt = gt.replace('---', '/')
    if ('---' in gt):  # 针对'/'进行特殊处理
        gt = gt.replace('---', '/')
        if ('@@@' in gt):
            gt = gt.replace('@@@', '：')

    if(gt==text):
        right_num+=1
    else:
        error_nums+=1
        error_strs.append(str_result)


if(len(error_strs)>0):
    print('预测错误的有：')
    for error_str in error_strs:
        print(error_str)
print('准确率：{}'.format(right_num/total_num))
print('错误个数：{}'.format(error_nums))










