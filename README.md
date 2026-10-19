构建深度学习图像分类模型识别 37 种猫狗品种 猫狗品种分类数据集的训练及应用
@[toc]
以下文字及代码仅供参考学习。
**数据集描述：
猫狗品种分类数据集
含37个不同类别的猫狗品种
每个品种下包含上千张该类猫或狗的图像**
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/09fc2a2ed68c4e33a04388419aecde3b.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c838e1e1f34a4d7a94694c260d140f9d.png)
非常适合用于训练一个高精度的 **细粒度图像分类模型**，可用于宠物识别 App、智能宠物医疗系统、宠物社交平台、宠物店自动化服务等场景。

以下是 **从环境搭建 → 数据准备 → 模型训练 → 推理部署 → 应用拓展** 的完整技术方案与应用场景指南。

---

## ✅ 一、项目目标

构建深度学习图像分类模型，能够：
- 准确识别 37 种猫狗品种
- 支持单图/批量/视频流输入
- 可部署于 Web、移动端或边缘设备

---

## ✅ 二、系统环境搭建

### 1. 安装 Anaconda（推荐）

前往 [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution) 下载并安装。

### 2. 创建虚拟环境

```bash
conda create -n pet_classifier python=3.9
conda activate pet_classifier
```

### 3. 安装依赖项

```bash
# 安装 PyTorch（GPU 版）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他必要库
pip install ultralytics opencv-python numpy matplotlib tqdm scikit-learn pandas pillow tensorboard streamlit gradio flask

# 验证 GPU
python -c "import torch; print(torch.cuda.is_available())"  # 应输出 True
```

---

## ✅ 三、数据集结构与预处理

### 推荐目录结构

```
pet_dataset/
├── train/
│   ├── golden_retriever/
│   ├── siamese_cat/
│   ├── german_shepherd/
│   └── ...（共 37 类）
├── val/
│   ├── golden_retriever/
│   └── ...
└── test/
    ├── golden_retriever/
    └── ...
```

> ✅ 你的数据已经是按类别分文件夹组织？直接使用！  
> ❌ 不是？用脚本自动分类。

### 数据划分脚本（8:1:1）

```python
# split_data.py
import os
import random
import shutil
from pathlib import Path

data_dir = 'raw_pets'  # 原始数据目录
output_dir = 'pet_dataset'

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

# 获取所有子类（品种）
classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

for cls in classes:
    cls_path = os.path.join(data_dir, cls)
    images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    n = len(images)
    train_files = images[:int(0.8*n)]
    val_files = images[int(0.8*n):int(0.9*n)]
    test_files = images[int(0.9*n):]

    for files, subset in zip([train_files, val_files, test_files], ['train', 'val', 'test']):
        dst_dir = os.path.join(output_dir, subset, cls)
        os.makedirs(dst_dir, exist_ok=True)
        for f in files:
            src = os.path.join(cls_path, f)
            dst = os.path.join(dst_dir, f)
            shutil.copy(src, dst)

print("✅ 数据划分完成：8:1:1")
```

---

## ✅ 四、使用 YOLOv8 进行图像分类训练（推荐）

Ultralytics YOLOv8 支持 `classify` 模式，适合多类别图像分类任务。

### 1. 创建 `data.yaml`

```yaml
# data.yaml
train: ./pet_dataset/train
val: ./pet_dataset/val
test: ./pet_dataset/test

# 类别数量
nc: 37

# 类别名称（自动从文件夹名读取，也可手动指定）
names:
  - golden_retriever
  - siamese_cat
  - german_shepherd
  - bulldog
  - persian_cat
  - husky
  - ...（共 37 个）
```

> ✅ 提示：可用脚本自动提取类别名：
>
> ```python
> print([d for d in os.listdir('pet_dataset/train')])
> ```

---

### 2. 训练代码（调用官方预训练模型）

```python
# train.py
from ultralytics import YOLO

# 加载官方预训练分类模型（基于 ImageNet）
model = YOLO('yolov8m-cls.pt')  # 推荐 medium 模型

# 开始训练
results = model.train(
    data='data.yaml',
    epochs=100,                 # 图像分类通常 50~150 足够
    batch=32,                   # 显存允许下尽量大
    imgsz=224,                  # 分类模型标准输入尺寸
    optimizer='AdamW',
    lr0=0.001,
    weight_decay=0.0005,
    momentum=0.9,

    # 数据增强
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,

    # 学习率调度
    cos_lr=True,

    # 早停
    patience=20,

    # 保存
    project='runs/classify',
    name='pet_breed_37',
    save=True,
    save_period=10,
    exist_ok=False,

    # 缓存
    cache=True
)
```

> 📌 模型选择建议：
> - `yolov8s-cls.pt`：轻量级，适合移动端
> - `yolov8m-cls.pt`：平衡精度与速度（推荐）
> - `yolov8l-cls.pt`：高精度，适合服务器端

---

## ✅ 五、推理与评估代码

### 1. 单图推理

```python
model = YOLO('runs/classify/pet_breed_37/weights/best.pt')
results = model('dog.jpg')

# 获取预测结果
probs = results[0].probs.data.cpu().numpy()
top5_idx = probs.argsort()[-5:][::-1]
names = model.names

for i in top5_idx:
    print(f"{names[i]}: {probs[i]:.4f}")
```

### 2. 批量推理

```python
results = model.predict(
    source='pet_dataset/test',
    save=True,
    project='runs/predict/classify',
    name='test_results',
    imgsz=224
)
```

### 3. 模型评估

```python
metrics = model.val(data='data.yaml', split='test')
print(f"Top-1 Accuracy: {metrics.top1:.4f}")
print(f"Top-5 Accuracy: {metrics.top5:.4f}")
```

---

## ✅ 六、模型导出（ONNX / TensorRT）

```python
# 导出为 ONNX（支持 Web / 移动端）
model.export(format='onnx', dynamic=True, opset=13, imgsz=224)

# 导出为 TensorRT（高性能部署）
model.export(format='engine', half=True, dynamic=True, imgsz=224)
```

---

## ✅ 七、应用场景与拓展

| 应用场景 | 技术实现 |
|--------|--------|
| **宠物识别 App** | 用户拍照 → 返回品种 + 置信度 |
| **智能宠物医院** | 自动识别品种 → 推荐常见疾病护理方案 |
| **宠物社交平台** | 上传照片自动打标签，增强互动 |
| **宠物店智能导购** | 识别顾客宠物品种 → 推荐适配商品 |
| **流浪动物救助** | 快速识别品种 → 匹配领养信息 |

---

## ✅ 八、进阶功能建议

### 1. 增加“猫/狗”一级分类器
先判断是猫还是狗，再进入对应子模型，提升准确率。

### 2. 构建 Web 服务（Flask + Gradio）

```python
import gradio as gr
from ultralytics import YOLO

model = YOLO('best.pt')

def classify_pet(img):
    results = model(img)
    probs = results[0].probs.data.cpu().numpy()
    return {model.names[i]: float(probs[i]) for i in range(len(probs))}

gr.Interface(fn=classify_pet, inputs="image", outputs="label").launch()
```

### 3. 添加品种百科知识库
结合 Wikipedia 或宠物数据库，返回品种特征、寿命、习性等。

### 4. 视频流实时识别
用 OpenCV 实时识别摄像头中的宠物品种。

---

💡
