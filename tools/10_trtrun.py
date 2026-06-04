#encoding=utf-8
"""
TensorRT 推理脚本
  - 输入：由 ONNX 转换得到的 TensorRT engine 文件 (.trt / .engine)
  - 模型已在 ONNX 阶段融合预处理 (uint8 输入，HWC->CHW，归一化等在模型内部完成)
  - 解码逻辑与 8_onnxrun.py 保持一致 (CTC greedy + 去重 + 去 blank)

用法：
    1. 将通过 trtexec 等工具得到的 .trt / .engine 放到 trt_path 指向的位置
    2. 修改 pic_path 指向待识别图片文件夹
    3. python tools/10_trtrun.py
"""
import os
import os.path
import sys

# 确保可以从 tools 目录直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from imutils import paths

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  自动初始化 CUDA 上下文

from tools.ocr_dicts import en_dict


# --------------------------- 路径与参数配置 ---------------------------
trt_path = r'../inference/rec_onnx/best-smi.engine'   # TensorRT engine 路径
pic_path = r'./imgs'                                # 测试图片目录
w, h = 384, 48                                      # 与 ONNX 推理保持一致

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# --------------------------- 工具函数 ---------------------------
def load_engine(engine_path: str) -> trt.ICudaEngine:
    """加载序列化后的 TensorRT engine"""
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f'engine 文件不存在: {engine_path}')
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError('engine 反序列化失败，请检查 TensorRT 版本是否匹配')
    return engine


def _np_dtype(trt_dtype):
    """trt dtype -> numpy dtype"""
    mapping = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT8: np.int8,
        trt.DataType.INT32: np.int32,
        trt.DataType.BOOL: np.bool_,
    }
    # UINT8 在新版本 TensorRT 才有
    if hasattr(trt.DataType, 'UINT8'):
        mapping[trt.DataType.UINT8] = np.uint8
    return mapping[trt_dtype]


class TRTRunner:
    """封装 TensorRT 推理：申请显存 -> H2D -> execute -> D2H"""

    def __init__(self, engine: trt.ICudaEngine, input_shape):
        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = cuda.Stream()

        # 兼容老接口 (binding 索引) 与新接口 (tensor name)
        self._use_tensor_api = hasattr(engine, 'num_io_tensors')

        self.inputs = []   # [{name, host, device, shape, dtype}]
        self.outputs = []
        self.bindings = []  # 给 execute_v2 使用

        if self._use_tensor_api:
            io_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        else:
            io_names = [engine.get_binding_name(i) for i in range(engine.num_bindings)]

        # 先设置输入动态 shape (如果是动态)
        for name in io_names:
            if self._is_input(name):
                if self._use_tensor_api:
                    self.context.set_input_shape(name, tuple(input_shape))
                else:
                    idx = engine.get_binding_index(name)
                    self.context.set_binding_shape(idx, tuple(input_shape))

        # 分配每个 IO 的 host / device 内存
        for name in io_names:
            shape = self._get_shape(name)
            dtype = _np_dtype(self._get_dtype(name))
            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            io = {
                'name': name,
                'host': host_mem,
                'device': device_mem,
                'shape': tuple(shape),
                'dtype': dtype,
            }
            if self._is_input(name):
                self.inputs.append(io)
            else:
                self.outputs.append(io)

            # 新接口需要绑定地址
            if self._use_tensor_api:
                self.context.set_tensor_address(name, int(device_mem))

    # ----- 兼容接口 -----
    def _is_input(self, name):
        if self._use_tensor_api:
            return self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        idx = self.engine.get_binding_index(name)
        return self.engine.binding_is_input(idx)

    def _get_shape(self, name):
        if self._use_tensor_api:
            return self.context.get_tensor_shape(name)
        idx = self.engine.get_binding_index(name)
        return self.context.get_binding_shape(idx)

    def _get_dtype(self, name):
        if self._use_tensor_api:
            return self.engine.get_tensor_dtype(name)
        idx = self.engine.get_binding_index(name)
        return self.engine.get_binding_dtype(idx)

    def infer(self, input_array: np.ndarray):
        # 拷贝输入到 host pinned memory
        np.copyto(self.inputs[0]['host'], input_array.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # 执行推理
        if self._use_tensor_api:
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # 拷回输出
        results = []
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        for out in self.outputs:
            results.append(out['host'].reshape(out['shape']))
        return results


def ctc_greedy_decode(preds: np.ndarray, dict_list):
    """与 8_onnxrun.py 完全一致的 CTC 贪心解码"""
    preds_idx = np.argmax(preds, axis=2)[0]
    preds_prob = np.max(preds, axis=2)[0]

    top_data = preds_idx[1:]
    bottom_data = preds_idx[:-1]

    filter_data = [preds_idx[0]]
    indexs = [0]

    index = 0
    for topdata, bottomdata in zip(top_data, bottom_data):
        index += 1
        if topdata != bottomdata:
            filter_data.append(topdata)
            indexs.append(index)

    char_list = []
    probs = []
    for cid, idx in zip(filter_data, indexs):
        if cid != 0:
            char_list.append(dict_list[cid])
            probs.append(preds_prob[idx])
    text = ''.join(char_list)
    mean_prob = float(np.mean(probs)) if probs else 0.0
    return text, mean_prob


# --------------------------- 主流程 ---------------------------
def main():
    engine = load_engine(trt_path)
    runner = TRTRunner(engine, input_shape=(1, 3, h, w))

    print(f'[INFO] 加载 engine 成功: {trt_path}')
    print(f'[INFO] 输入: name={runner.inputs[0]["name"]}, shape={runner.inputs[0]["shape"]}, dtype={runner.inputs[0]["dtype"]}')
    for out in runner.outputs:
        print(f'[INFO] 输出: name={out["name"]}, shape={out["shape"]}, dtype={out["dtype"]}')

    img_paths = list(paths.list_images(pic_path))
    if not img_paths:
        print(f'[WARN] 在 {pic_path} 中未找到图片')
        return

    for img_path in img_paths:
        filename = img_path.split(os.path.sep)[-1]
        img = cv2.imread(img_path)
        if img is None:
            print(f'[WARN] 读取失败: {img_path}')
            continue

        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        # 预处理已融合进模型，保持 uint8 喂给模型
        inp = np.transpose(img, (2, 0, 1))[None, ...].astype(runner.inputs[0]['dtype'])

        outputs = runner.infer(inp)
        # 第一个输出对应 ONNX 中的概率张量，shape: [1, T, num_classes]
        preds = outputs[0]
        text, mean_prob = ctc_greedy_decode(preds, en_dict)
        print('({})-({})-({})'.format(filename, text, mean_prob))


if __name__ == '__main__':
    main()
