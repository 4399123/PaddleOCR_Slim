# -*- coding: utf-8 -*-
import os.path
from tools.ocr_dicts import ch_dict
import onnx
import onnxruntime as ort
import numpy as np
from imutils import paths
import cv2

# 路径配置
onnx_path = r'../inference/rec_onnx/best-smi.onnx'
pic_path  = r'./imgs_ch'
w, h = 384, 48


def imread_unicode(path):
    """支持中文路径的图像读取"""
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# onnx 模型加载
model = onnx.load(onnx_path)
onnx.checker.check_model(model)
session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

imgpaths = list(paths.list_images(pic_path))

for imgpath in imgpaths:
    img = imread_unicode(imgpath)               # 支持中文路径
    filename = imgpath.split(os.path.sep)[-1]
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = np.array([np.transpose(img, (2, 0, 1))], dtype=np.uint8)  # HWC→CHW uint8

    # 模型推理
    out = session.run(None, input_feed={'input': img})

    preds_idx  = np.argmax(out[0], axis=2)[0]
    preds_prob = np.max(out[0],    axis=2)[0]

    # CTC 去重：相邻相同的只保留一个
    top_data    = preds_idx[1:]   # 第2个元素到最后一个元素
    bottom_data = preds_idx[:-1]  # 第1个元素到倒数第二个元素

    filter_data = [preds_idx[0]]  # 第一个元素不受影响
    indexs      = [0]             # 对应 filter_data 的索引值

    index = 0
    for topdata, bottomdata in zip(top_data, bottom_data):
        index += 1
        if topdata != bottomdata:          # 不完全相同则保留
            filter_data.append(topdata)    # 记录对比不相同的 topdata
            indexs.append(index)           # 对应索引

    char_list = []
    probs     = []

    # 去掉 id 为 0 的元素（id=0 是空格/blank）
    for id, index in zip(filter_data, indexs):
        if id != 0:
            char_list.append(ch_dict[id])  # 使用中文字典
            probs.append(preds_prob[index])
    text = ''.join(char_list)

    print('({})-({})-({:.4f})'.format(filename, text, np.mean(probs) if probs else 0.0))








