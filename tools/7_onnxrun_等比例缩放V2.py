#encoding=gbk
import os.path

import onnx
import onnxruntime as ort
import numpy as np
from imutils import paths
import cv2

#路径配置
onnx_path=r'../inference/rec_onnx/best-smi.onnx'
# pic_path=r'../inference/rec_onnx'
# pic_path=r'./outocr'
pic_path=r'./imgs'
scale_size_w,scale_size_h=480,48

#字典
en_dict=['blank','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '', '']
#onnx模型载入
model = onnx.load(onnx_path)
onnx.checker.check_model(model)
session = ort.InferenceSession(onnx_path,providers=['CPUExecutionProvider'])

imgpaths=list(paths.list_images(pic_path))

for imgpath in imgpaths:
    imgO=cv2.imread(imgpath)
    filename=imgpath.split(os.path.sep)[-1]

    try:
        h, w = imgO.shape[:2]
    except:
        continue

    img_ratio=h/w
    out_h=int(img_ratio*scale_size_w)
    if(out_h<scale_size_h):
        resize_img=cv2.resize(imgO,(scale_size_w,out_h))
        img = cv2.copyMakeBorder(resize_img, 0, scale_size_h-out_h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        out_w=int(scale_size_h/img_ratio)
        resize_img = cv2.resize(imgO, (out_w, scale_size_h))
        img = cv2.copyMakeBorder(resize_img, 0, 0, 0, scale_size_w-out_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))





    # img=cv2.resize(img,(w,h))
    img=np.array(img).astype(np.float32)  # 注意输入type一定要np.float32
    img-=127.5                             #减均值
    img/=127.5                              #除方差
    img=np.array([np.transpose(img,(2,0,1))])
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








