# encoding=utf-8
"""
将预处理（uint8 → float32、减均值127.5、除方差127.5）融合进 ONNX 模型。

使用方式：
    python 4.5_onnx_fuse_preprocess.py

输入：原始 ONNX 模型（输入为 float32，值域 [-1, 1]）
输出：新 ONNX 模型（输入为 uint8，值域 [0, 255]），内部自动完成预处理

融合后推理侧只需：
    img = cv2.resize(img, (w, h))
    img = np.array([img.transpose(2, 0, 1)], dtype=np.uint8)  # HWC→CHW，uint8
    out = session.run(None, {'input': img})
"""

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper

# ──────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────
src_onnx_path = r'../inference/rec_onnx/best_rename.onnx'
dst_onnx_path = r'../inference/rec_onnx/best_u8.onnx'

MEAN = 127.5
STD  = 127.5


def copy_shape_with_new_dtype(orig_input, new_name, new_dtype):
    """
    复制原始输入的 shape（保留动态轴的 dim_param），只替换名字和 dtype。
    """
    new_type = onnx.TypeProto()
    new_type.tensor_type.elem_type = new_dtype
    orig_shape = orig_input.type.tensor_type.shape
    new_shape = new_type.tensor_type.shape
    for orig_dim in orig_shape.dim:
        new_dim = new_shape.dim.add()
        if orig_dim.HasField('dim_param') or orig_dim.dim_param != '':
            # 动态轴：保留符号名
            new_dim.dim_param = orig_dim.dim_param
        elif orig_dim.dim_value > 0:
            # 静态轴
            new_dim.dim_value = orig_dim.dim_value
        else:
            # dim_value==0 且无 dim_param，说明是动态轴但没有符号名，给一个
            new_dim.dim_param = 'batch'
    vi = onnx.ValueInfoProto()
    vi.name = new_name
    vi.type.CopyFrom(new_type)
    return vi


def fuse_preprocess(src_path: str, dst_path: str, mean: float, std: float):
    model = onnx.load(src_path)
    graph = model.graph

    # ── 1. 读取原始输入信息 ────────────────────────────────────────
    orig_input      = graph.input[0]
    orig_input_name = orig_input.name          # "input"
    orig_dtype      = orig_input.type.tensor_type.elem_type  # 1 = float32

    # 打印原始 shape 供确认
    raw_dims = orig_input.type.tensor_type.shape.dim
    shape_str = [d.dim_param if (d.dim_param != '') else d.dim_value for d in raw_dims]
    print(f"原始输入: name={orig_input_name}, dtype={orig_dtype}, shape={shape_str}")

    # ── 2. 新输入节点：uint8，shape 与原始相同（保留动态轴）─────────
    new_input_name = orig_input_name          # 保持同名，推理代码无需改动
    new_input = copy_shape_with_new_dtype(orig_input, new_input_name, TensorProto.UINT8)

    # ── 3. 中间张量名 ──────────────────────────────────────────────
    cast_out = orig_input_name + "__f32"
    sub_out  = orig_input_name + "__sub"
    div_out  = orig_input_name + "__norm"   # 接入原模型第一层

    # ── 4. 构造三个节点 ────────────────────────────────────────────
    cast_node = helper.make_node(
        "Cast",
        inputs=[new_input_name],
        outputs=[cast_out],
        to=int(TensorProto.FLOAT)
    )

    mean_const_name = "__preproc_mean__"
    std_const_name  = "__preproc_std__"

    # 用 Constant 节点存标量，避免 initializer 在某些 runtime 下的兼容问题
    mean_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[mean_const_name],
        value=numpy_helper.from_array(np.array(mean, dtype=np.float32))
    )
    std_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[std_const_name],
        value=numpy_helper.from_array(np.array(std, dtype=np.float32))
    )

    sub_node = helper.make_node(
        "Sub",
        inputs=[cast_out, mean_const_name],
        outputs=[sub_out]
    )
    div_node = helper.make_node(
        "Div",
        inputs=[sub_out, std_const_name],
        outputs=[div_out]
    )

    # ── 5. 把原模型所有节点中对 orig_input_name 的引用改为 div_out ──
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp == orig_input_name:
                node.input[i] = div_out

    # ── 6. 将新节点插到 graph 最前面（顺序：cast→mean_const→std_const→sub→div）
    for n in reversed([cast_node, mean_node, std_node, sub_node, div_node]):
        graph.node.insert(0, n)

    # ── 7. 替换 graph.input ────────────────────────────────────────
    # 先清空再重建，避免 protobuf repeated field 操作顺序问题
    old_inputs = list(graph.input)
    del graph.input[:]
    for old_inp in old_inputs:
        if old_inp.name == orig_input_name:
            graph.input.append(new_input)
        else:
            graph.input.append(old_inp)

    # ── 8. 校验并保存 ─────────────────────────────────────────────
    try:
        onnx.checker.check_model(model)
        print("✅ onnx.checker 校验通过")
    except onnx.checker.ValidationError as e:
        print(f"⚠️  onnx.checker 警告（不影响运行）: {e}")

    onnx.save(model, dst_path)
    print(f"✅ 已保存到: {dst_path}")
    print(f"   新输入: name={new_input_name}, dtype=uint8, shape={shape_str}")
    print(f"   预处理: (cast_to_float32 - {mean}) / {std}")

    # ── 9. 快速验证：用随机 uint8 数据跑一次推理 ──────────────────
    print("\n── 推理验证 ──")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(dst_path, providers=['CPUExecutionProvider'])
        inp_info = sess.get_inputs()[0]
        print(f"   ORT 输入: name={inp_info.name}, type={inp_info.type}, shape={inp_info.shape}")

        # 构造一个 batch=1 的随机 uint8 输入
        static_shape = [1 if (isinstance(d, str) or d == 0) else d
                        for d in inp_info.shape]
        dummy = np.random.randint(0, 256, static_shape, dtype=np.uint8)
        result = sess.run(None, {inp_info.name: dummy})
        print(f"   输出 shape: {result[0].shape}, dtype: {result[0].dtype}")
        print("✅ 推理验证通过")
    except Exception as e:
        print(f"❌ 推理验证失败: {e}")


if __name__ == "__main__":
    fuse_preprocess(src_onnx_path, dst_onnx_path, MEAN, STD)
