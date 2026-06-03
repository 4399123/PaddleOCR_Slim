# -*- coding: utf-8 -*-
import os.path
from tqdm import tqdm
import onnx
import onnxruntime as ort
import numpy as np
from imutils import paths
import cv2
from tools.ocr_dicts import ch_dict

# 路径配置
onnx_path = r'../inference/rec_onnx/best-smi.onnx'
pic_path  = r'D:\E\github_zl\OCRDataSetV1\rec\test'
# pic_path = r'./test'
# pic_path = r'./imgs'
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

total_num  = len(imgpaths)
right_num  = 0
error_strs = []
error_nums  = 0

for imgpath in tqdm(imgpaths):
    img = imread_unicode(imgpath)               # 支持中文路径
    filename = imgpath.split(os.path.sep)[-1]
    img = cv2.resize(img, (w, h))
    img = np.array([np.transpose(img, (2, 0, 1))], dtype=np.uint8)  # HWC→CHW uint8

    # 模型推理
    out = session.run(None, input_feed={'input': img})

    preds_idx  = np.argmax(out[0], axis=2)[0]
    preds_prob = np.max(out[0],    axis=2)[0]

    # CTC 去重：相邻相同的只保留一个
    top_data    = preds_idx[1:]
    bottom_data = preds_idx[:-1]

    filter_data = [preds_idx[0]]
    indexs      = [0]

    index = 0
    for topdata, bottomdata in zip(top_data, bottom_data):
        index += 1
        if topdata != bottomdata:
            filter_data.append(topdata)
            indexs.append(index)

    char_list = []
    probs     = []

    for id, index in zip(filter_data, indexs):
        if id != 0:
            char_list.append(ch_dict[id])      # 使用中文字典
            probs.append(preds_prob[index])
    text = ''.join(char_list)

    str_result = '({})-({})-({:.4f})'.format(filename, text, np.mean(probs) if probs else 0.0)
    gt = filename.split('_')[0]

    # 文件名中的特殊字符还原
    if '@@@' in gt:    # ':' 的替代编码
        gt = gt.replace('@@@', ':')
    if '---' in gt:    # '/' 的替代编码
        gt = gt.replace('---', '/')

    if gt == text:
        right_num += 1
    else:
        error_nums += 1
        error_strs.append(str_result)

if error_strs:
    print('预测错误样本：')
    for s in error_strs:
        print(s)
print('准确率：{:.4f}'.format(right_num / total_num))
print('错误数量：{}'.format(error_nums))










