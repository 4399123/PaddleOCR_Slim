# PaddleOCR - 光学字符识别系统

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

PaddleOCR 是一个基于 PaddlePaddle 深度学习框架的高效、实用的光学字符识别（OCR）系统，支持中文、英文及多种语言的文字检测与识别。

## 功能特性

- **多语言支持**：支持中文、英文、日文、韩文、阿拉伯文、印地文等20+种语言
- **完整OCR流程**：文本检测 → 方向分类 → 文本识别
- **多个模型版本**：PP-OCRv4、PP-OCRv3、PP-OCRv2、PP-OCR
- **表格识别**：PP-Structure 和 PP-StructureV2 支持表格结构识别
- **灵活部署**：支持 PaddlePaddle 推理和 ONNX 格式模型
- **批处理支持**：支持单张图像、PDF 文件和批量图像处理

## 项目结构

```
.
├── paddleocr.py                 # 主入口文件
├── ppocr/                       # 核心 OCR 模块
│   ├── modeling/                # 模型定义（检测、识别、分类）
│   ├── data/                    # 数据处理、增强
│   ├── losses/                  # 损失函数
│   ├── metrics/                 # 评估指标
│   ├── utils/                   # 工具函数
│   └── ext_op/                  # 扩展操作
├── ppstructure/                 # 表格识别模块
├── tools/                       # 推理工具
│   ├── infer/                   # 推理接口
│   └── 9_onnxrun_中文.py        # ONNX 模型推理示例
├── configs/                     # 模型配置文件
├── inference/                   # 预导出的推理模型
└── output/                      # 输出结果目录
```

## 快速开始

### 安装依赖

```bash
pip install paddlepaddle onnxruntime opencv-python numpy pillow
```

### 基础使用

```python
from paddleocr import PaddleOCR

# 初始化 OCR（自动下载模型）
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

# 执行 OCR 识别
result = ocr.ocr('path/to/image.jpg', cls=True)

# 输出结果格式: [[[x1,y1,x2,y2,...], [text, confidence]], ...]
for line in result:
    for word_info in line:
        print(word_info)
```

### ONNX 推理示例

参考 `tools/9_onnxrun_中文.py` 进行 ONNX 格式模型推理：

```bash
cd tools
python 9_onnxrun_中文.py
```

## 支持的模型版本

### OCR 模型
| 版本 | 检测 | 识别 | 分类 | 语言支持 |
|------|------|------|------|---------|
| PP-OCRv4 | ✓ | ✓ | ✓ | 20+ 语言 |
| PP-OCRv3 | ✓ | ✓ | ✓ | 20+ 语言 |
| PP-OCRv2 | ✓ | ✓ | ✓ | 中文 |
| PP-OCR  | ✓ | ✓ | ✓ | 多语言 |

### 结构识别模型
| 版本 | 表格识别 | 版面分析 | 语言支持 |
|------|---------|---------|---------|
| PP-StructureV2 | ✓ | ✓ | 中英文 |
| PP-Structure   | ✓ | - | 英文 |

## 主要 API

### PaddleOCR 类

```python
# 初始化
ocr = PaddleOCR(
    use_angle_cls=True,           # 是否使用方向分类
    lang='ch',                    # 语言（ch/en/...）
    ocr_version='PP-OCRv4',       # 模型版本
    use_gpu=True                  # 是否使用 GPU
)

# 执行 OCR
result = ocr.ocr(
    img,                          # 图像路径、numpy数组、bytes
    det=True,                     # 是否执行文本检测
    rec=True,                     # 是否执行文本识别
    cls=True,                     # 是否执行方向分类
    bin=False,                    # 是否二值化
    inv=False,                    # 是否反色
    alpha_color=(255,255,255)     # 透明部分填充颜色
)
```

### 绘制结果

```python
from paddleocr import draw_ocr

# 绘制 OCR 结果到图像
image = draw_ocr(image, result)
cv2.imwrite('output.jpg', image)
```

## 配置说明

配置文件位于 `configs/rec/PP-OCRv4/` 目录：

- `ch_PP-OCRv4_rec.yml` - 中文识别标准配置
- `ch_PP-OCRv4_rec_hgnet.yml` - 中文识别 HGNet 优化版
- `ch_PP-OCRv4_rec_distill.yml` - 中文识别蒸馏模型
- `ch_PP-OCRv4_rec_ampO2_ultra.yml` - 超轻量版本配置

## 模型下载

预导出的模型文件位于 `inference/` 目录：

- `ch_PP-OCRv4_rec_export/` - Paddle 格式推理模型
- `rec_onnx/` - ONNX 格式推理模型

## 性能对比

不同模型版本的性能表现（仅供参考）：

| 模型 | 检测 ACC | 识别 ACC | 推理速度 |
|------|---------|---------|---------|
| PP-OCRv4 | 88.3% | 76.8% | 快 |
| PP-OCRv3 | 85.0% | 73.0% | 中等 |
| PP-OCRv2 | 82.5% | 70.0% | 较快 |

## 环境要求

- Python 3.6+
- PaddlePaddle 2.0+
- OpenCV
- NumPy
- Pillow

## 常见问题

**Q: 如何加速推理？**
A: 可以使用 GPU（`use_gpu=True`）、ONNX 格式模型或选择轻量化版本。

**Q: 如何支持自定义语言？**
A: 需要准备对应语言的字典和训练相应的模型。

**Q: 推理模型在哪里下载？**
A: 首次使用时会自动下载到 `~/.paddleocr/` 目录。

## 许可证

Apache License 2.0

## 相关资源

- [PaddleOCR 官方文档](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddlePaddle 框架](https://www.paddlepaddle.org.cn/)

---

**版本**: 2.7.0.3  
**最后更新**: 2026年
