#encoding=gbk
import os.path
from tools.ocr_dicts import en_dict
import onnx
import onnxruntime as ort
import numpy as np
from imutils import paths
import cv2

#路径配置
onnx_path=r'../inference/rec_onnx/best-smi.onnx'
pic_path=r'./imgs'
w,h=384,48

#onnx模型载入
model = onnx.load(onnx_path)
onnx.checker.check_model(model)
session = ort.InferenceSession(onnx_path,providers=['CPUExecutionProvider'])

imgpaths=list(paths.list_images(pic_path))

for imgpath in imgpaths:
    img=cv2.imread(imgpath)
    filename=imgpath.split(os.path.sep)[-1]
    img=cv2.resize(img,(w,h),interpolation=cv2.INTER_LINEAR)
    # 预处理已融合进模型，直接保持 uint8 喂给模型
    img=np.array([np.transpose(img,(2,0,1))], dtype=np.uint8)  # HWC→CHW，uint8
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

    print('({})-({})-({})'.format(filename,text,np.mean(probs)))








