#encoding=gbk
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2

#路径配置
onnx_path=r'../inference/rec_onnx/best-smi.onnx'
pic_path=r'../inference/rec_onnx/11.png'
w,h=320,48



#onnx模型载入
model = onnx.load(onnx_path)
onnx.checker.check_model(model)
session = ort.InferenceSession(onnx_path,providers=['CPUExecutionProvider'])

img=cv2.imread(pic_path)
img=cv2.resize(img,(w,h))
img=np.array(img).astype(np.float32)  # 注意输入type一定要np.float32
img-=127.5                             #减均值
img/=127.5                              #除方差
img=np.array([np.transpose(img,(2,0,1))])
#模型推理
out = session.run(None,input_feed = { 'input' : img })

preds_idx=np.argmax(out[0],axis=2)[0]
preds_prob=np.max(out[0],axis=2)[0]


dict_path=r'../inference/rec_onnx/en_dict.txt'

with open(dict_path,'r') as f:
    lines=f.readlines()
    new_lines=['blank']
    for line in lines:
        line1=line.strip()
        new_lines.append(line1)

result_list = []
ignored_tokens=[0]
selection = np.ones(len(preds_idx), dtype=bool)
selection[1:] = preds_idx[1:] != preds_idx[:-1]
for ignored_token in ignored_tokens:
    selection &= preds_idx != ignored_token

char_list = [new_lines[text_id]for text_id in preds_idx[selection]]
if preds_prob is not None:
    conf_list = preds_prob[selection]
else:
    conf_list = [1] * len(selection)
if len(conf_list) == 0:
    conf_list = [0]

text = ''.join(char_list)

result_list.append((text, np.mean(conf_list).tolist()))
print(result_list)


